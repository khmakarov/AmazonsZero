#database.py
import sqlite3
import json
import lz4.frame
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict


class AmazonsDatabase:

    def __init__(self, db_path: str = "/home/khmakarov/AmazonsZero/data/game/v.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构（包含级联更新支持）"""
        with self._get_connection() as conn:
            conn.executescript(
                '''
                CREATE TABLE IF NOT EXISTS games (
                    game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    result TEXT CHECK(result IN ('黑胜', '白胜', '未结束')),
                    total_steps INTEGER NOT NULL,
                    iteration INTEGER NOT NULL DEFAULT 0,
                    phase INTEGER CHECK(phase IN (0, 1)) NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS moves (
                    move_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER NOT NULL,
                    step_number INTEGER NOT NULL,
                    board_state BLOB NOT NULL,
                    action TEXT,
                    player INTEGER CHECK(player IN (0, 1)),
                    top5_pi TEXT,  -- 新增字段
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                        ON UPDATE CASCADE
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_games_time ON games(timestamp);
                CREATE INDEX IF NOT EXISTS idx_moves_game ON moves(game_id);
            '''
            )

    def _get_connection(self):
        """获取数据库连接（启用WAL和外键）"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")  # 启用外键支持
        return conn

    def save_games(self, games_data):
        """
        批量插入多个对局数据
        games_data: List[Tuple(episode_data, result, iteration, phase)]
        """
        with self._get_connection() as conn:
            game_ids = []
            batch_moves = []

            # 首先生成所有game记录
            for episode_data, result, iteration, phase in games_data:
                cursor = conn.execute(
                    '''INSERT INTO games (result, total_steps, iteration, phase)
                    VALUES (?, ?, ?, ?)''', ("黑胜" if result == 1 else "白胜", len(episode_data) - 1, iteration, phase)
                )
                game_id = cursor.lastrowid
                game_ids.append(game_id)

                # 准备步骤数据
                for step_idx, (state, action, top5_pi) in enumerate(episode_data):
                    state = lz4.frame.compress(state.tobytes())
                    batch_moves.append((game_id, step_idx, state, json.dumps(action) if action else None, 1 if step_idx % 2 else 0, json.dumps(top5_pi)))

            # 批量插入所有步骤数据
            if batch_moves:
                conn.executemany(
                    '''
                    INSERT INTO moves 
                    (game_id, step_number, board_state, action, player, top5_pi)
                    VALUES (?, ?, ?, ?, ?, ?)
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
                SELECT board_state, action, top5_pi 
                FROM moves 
                WHERE game_id = ? 
                ORDER BY step_number
            ''', (game_id, )
            ).fetchall()

            episode_data = []
            for state_bytes, action_json, top5_json in rows:
                decompressed = lz4.frame.decompress(state_bytes)
                state = np.frombuffer(decompressed, dtype=np.int8)
                state = state.reshape(8, 8, 5)
                action = json.loads(action_json) if action_json else None
                top5_pi = json.loads(top5_json) if top5_json else None
                episode_data.append((state, action, top5_pi))

            return episode_data

    def query_games(self, limit: int = 100, result_filter: str = None, start_date: str = None, iteration: int = None, phase: int = None) -> List[Dict]:
        """增强查询功能（新增对iteration和phase的过滤）"""
        query = '''
            SELECT game_id, timestamp, result, total_steps, iteration, phase
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
        if iteration is not None:
            query += " AND iteration = ?"
            params.append(iteration)
        if phase is not None:
            query += " AND phase = ?"
            params.append(phase)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [{"game_id": row[0], "timestamp": row[1], "result": row[2], "steps": row[3], "iteration": row[4], "phase": row[5]} for row in cursor.fetchall()]

    def delete_games_range(self, start_id: int, end_id: int) -> None:
        """
        删除指定范围内的游戏并重新编号后续ID
        操作流程：
        1. 删除指定范围内的游戏记录（级联删除关联的移动记录）
        2. 更新后续游戏的game_id以填补空缺
        3. 重置自增ID计数器
        """
        with self._get_connection() as conn:
            conn.execute("BEGIN TRANSACTION;")
            try:
                # 删除指定范围内的游戏记录
                conn.execute("DELETE FROM games WHERE game_id BETWEEN ? AND ?;", (start_id, end_id))

                # 计算需要填补的数量
                deleted_count = end_id - start_id + 1

                # 更新后续游戏的game_id（触发级联更新）
                conn.execute("UPDATE games SET game_id = game_id - ? WHERE game_id > ?;", (deleted_count, end_id))

                # 获取当前最大ID（处理空表情况）
                max_id = conn.execute("SELECT MAX(game_id) FROM games;").fetchone()[0] or 0

                # 更新自增序列
                conn.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = 'games';", (max_id, ))

                conn.execute("COMMIT;")
            except Exception as e:
                conn.execute("ROLLBACK;")
                raise
