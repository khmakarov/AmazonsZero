#database.py
import sqlite3
import json
import lz4.frame
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict


class AmazonsDatabase:

    def __init__(self, db_path: str = "/home/khmakarov/AmazonsZero/data/game/amazons.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构"""
        with self._get_connection() as conn:
            conn.executescript(
                '''
                CREATE TABLE IF NOT EXISTS games (
                    game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    result TEXT CHECK(result IN ('黑胜', '白胜', '未结束')),
                    total_steps INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS moves (
                    move_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER NOT NULL,
                    step_number INTEGER NOT NULL,
                    board_state BLOB NOT NULL,  -- 压缩后的棋盘状态
                    action TEXT,  -- JSON格式: {"from": 12, "to": 28, "block": 35}
                    player INTEGER CHECK(player IN (0, 1)),
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                );

                CREATE INDEX IF NOT EXISTS idx_games_time ON games(timestamp);
                CREATE INDEX IF NOT EXISTS idx_moves_game ON moves(game_id);
            '''
            )

    def _get_connection(self):
        """获取数据库连接（启用WAL模式提升并发性能）"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def save_games(self, games_data):
        """
        批量插入多个对局数据
        games_data: List[Tuple(episode_data, result)]
        """
        with self._get_connection() as conn:
            game_ids = []
            batch_moves = []

            # 首先生成所有game记录
            for episode_data, result in games_data:
                cursor = conn.execute(
                    '''INSERT INTO games (result, total_steps)
                    VALUES (?, ?)''', (result, len(episode_data))
                )
                game_id = cursor.lastrowid
                game_ids.append(game_id)

                # 准备步骤数据
                for step_idx, (state, action) in enumerate(episode_data):
                    state = lz4.frame.compress(state.tobytes())
                    batch_moves.append((game_id, step_idx, state, json.dumps(action) if action else None, 1 if step_idx % 2 else 0))

            # 批量插入所有步骤数据
            if batch_moves:
                conn.executemany(
                    '''
                    INSERT INTO moves 
                    (game_id, step_number, board_state, action, player)
                    VALUES (?, ?, ?, ?, ?)
                ''', batch_moves
                )

            return game_ids

    def load_game(self, game_id: int) -> List[Tuple[object, tuple]]:
        """
        加载指定对局数据
        返回格式：[(state, action), ...]
        """
        with self._get_connection() as conn:
            # 获取步骤数据（按顺序排序）
            rows = conn.execute(
                '''
                SELECT board_state, action 
                FROM moves 
                WHERE game_id = ? 
                ORDER BY step_number
            ''', (game_id, )
            ).fetchall()

            episode_data = []
            for state_bytes, action_json in rows:
                decompressed = lz4.frame.decompress(state_bytes)
                state = np.frombuffer(decompressed, dtype=np.int8)
                state = state.reshape(8, 8, 5)
                action = json.loads(action_json) if action_json else None
                episode_data.append((state, action))

            return episode_data

    def query_games(self, limit: int = 100, result_filter: str = None, start_date: str = None) -> List[Dict]:
        """查询最近的若干对局元数据"""
        query = '''
            SELECT game_id, timestamp, result, total_steps
            FROM games
            WHERE 1=1
        '''
        params = []

        if result_filter:
            query += " AND result = ?"
            params.append(result_filter)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [{"game_id": row[0], "timestamp": row[1], "result": row[2], "steps": row[3]} for row in cursor.fetchall()]
