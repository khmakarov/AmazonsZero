import sqlite3
import json
import zlib
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from Amazons import GameCore


class AmazonsDatabase:

    def __init__(self, db_path: str = "amazons.db"):
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
                    total_steps INTEGER NOT NULL,
                    additional_info TEXT  -- 可扩展字段
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

    # --------- 核心操作方法 ---------
    def save_game(
        self,
        episode_data: List[Tuple[object, tuple]],  # (state, action)
        result: str
    ) -> int:
        """
        保存完整对局到数据库
        返回插入的game_id
        """
        with self._get_connection() as conn:
            # 插入元数据
            cursor = conn.execute('''
                INSERT INTO games (result, total_steps)
                VALUES (?, ?)
            ''', (result, len(episode_data)))
            game_id = cursor.lastrowid

            # 批量插入步骤数据
            batch_data = []
            for step_idx, (state, action) in enumerate(episode_data):
                batch_data.append((game_id, step_idx, self._serialize_state(state), json.dumps(action) if action else None, state.current_player))

            conn.executemany(
                '''
                INSERT INTO moves 
                (game_id, step_number, board_state, action, player)
                VALUES (?, ?, ?, ?, ?)
            ''', batch_data
            )

            return game_id

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
            for board_state, action_json in rows:
                state = self._deserialize_state(board_state)
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

    # --------- 序列化方法 ---------
    def _serialize_state(self, state) -> bytes:
        """序列化游戏状态为压缩字节流"""
        state_data = {"board": np.array(state.get_state()).tolist(), "black": int(state.black), "white": int(state.white), "blocks": int(state.blocks), "current_player": state.current_player}
        return zlib.compress(json.dumps(state_data).encode())

    def _deserialize_state(self, data: bytes):
        """从字节流重建游戏状态"""
        state_data = json.loads(zlib.decompress(data).decode())

        # 重建GameCore实例
        state = GameCore()
        state.current_player = state_data["current_player"]
        state.black = state_data["black"]
        state.white = state_data["white"]
        state.blocks = state_data["blocks"]

        return state
