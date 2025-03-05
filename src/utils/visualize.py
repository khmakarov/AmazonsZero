import tkinter as tk
from tkinter import ttk
import numpy as np
from src.utils.database import AmazonsDatabase


class AmazonsVisualizer:

    def __init__(self, game_id: int):
        self.root = tk.Tk()
        self.root.title("亚马逊棋对弈回放")
        self.db = AmazonsDatabase()
        self.history = self.db.load_game(game_id)
        self.current_step = 0
        self.cell_size = 70  # 单元格尺寸
        self.piece_radius = 25  # 棋子半径
        self.block_radius = 25  # 障碍半径
        self.left_margin = 40  # 左侧边距（行标签区域）
        self.top_margin = 40  # 顶部边距（列标签区域）
        self.board_size = self.cell_size * 8

        # 初始化画布（增加边距空间）
        self.canvas = tk.Canvas(self.root, width=self.left_margin + self.board_size + 20, height=self.top_margin + self.board_size + 20, bg="#DEB887")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # 控制面板
        control_frame = ttk.Frame(self.root)
        control_frame.grid(row=1, column=0, sticky="ew")

        # 步数标签
        self.step_label = ttk.Label(control_frame, text="步数：0/0")
        self.step_label.pack()

        # 进度条
        self.slider = ttk.Scale(control_frame, from_=0, to=len(self.history) - 1, command=self.on_slider_changed)
        self.slider.pack(fill="x", padx=10)

        # 控制按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=5)

        ttk.Button(btn_frame, text="<< 开始", command=self.first_step).pack(side="left")
        ttk.Button(btn_frame, text="< 上一步", command=self.prev_step).pack(side="left")
        ttk.Button(btn_frame, text="下一步 >", command=self.next_step).pack(side="left")
        ttk.Button(btn_frame, text="结尾 >>", command=self.last_step).pack(side="left")

        # 初始绘制
        self.update_step_label()
        self.draw_board()

    @classmethod
    def from_recent_game(cls):
        """快速加载最近对局"""
        db = AmazonsDatabase()
        games = db.query_games(limit=1)
        if not games:
            raise ValueError("没有找到历史对局")
        return cls(games[0]["game_id"])

    # ---------- 核心绘制方法 ----------
    def draw_board(self):
        """绘制棋盘基础网格和坐标标签"""
        self.canvas.delete("all")

        # 绘制棋盘网格（偏移边距）
        for i in range(9):
            pos = i * self.cell_size
            # 水平线
            self.canvas.create_line(self.left_margin, self.top_margin + pos, self.left_margin + self.board_size, self.top_margin + pos)
            # 垂直线
            self.canvas.create_line(self.left_margin + pos, self.top_margin, self.left_margin + pos, self.top_margin + self.board_size)

        # 绘制坐标标签
        self._draw_column_labels()  # 列标签（A-H）
        self._draw_row_labels()  # 行标签（1-8）

        # 绘制棋子状态
        self.draw_state()

    def _draw_column_labels(self):
        """在顶部边距绘制列标签（A-H）"""
        for x in range(8):
            label_x = self.left_margin + x * self.cell_size + self.cell_size // 2
            label_y = self.top_margin // 2
            self.canvas.create_text(label_x, label_y, text=chr(65 + x), anchor="center", font=("Arial", 14, "bold"), fill="#2F4F4F")

    def _draw_row_labels(self):
        """在左侧边距绘制行标签（1-8）"""
        for y in range(8):
            label_x = self.left_margin // 2
            label_y = self.top_margin + y * self.cell_size + self.cell_size // 2
            self.canvas.create_text(label_x, label_y, text=str(y + 1), anchor="center", font=("Arial", 14, "bold"), fill="#2F4F4F")

    def draw_state(self):
        """绘制当前状态的棋子和障碍"""
        state = self.history[self.current_step][0]
        grid = np.array(state.get_state())

        for y in range(8):
            for x in range(8):
                cell = grid[y][x]
                # 计算实际绘制坐标（考虑边距）
                cx = self.left_margin + x * self.cell_size + self.cell_size // 2
                cy = self.top_margin + y * self.cell_size + self.cell_size // 2

                # 绘制障碍（蓝色圆形）
                if cell[2] == 1:
                    self.canvas.create_oval(cx - self.block_radius, cy - self.block_radius, cx + self.block_radius, cy + self.block_radius, fill="#4169E1", outline="black")

                # 绘制黑棋
                if cell[0] == 1:
                    self.canvas.create_oval(cx - self.piece_radius, cy - self.piece_radius, cx + self.piece_radius, cy + self.piece_radius, fill="black", width=2)

                # 绘制白棋
                if cell[1] == 1:
                    self.canvas.create_oval(cx - self.piece_radius, cy - self.piece_radius, cx + self.piece_radius, cy + self.piece_radius, fill="white", width=2)

        # 高亮上一步动作
        if self.current_step > 0 and self.history[self.current_step - 1][1] is not None:
            self._highlight_action(self.history[self.current_step - 1][1])

    def _highlight_action(self, action):
        """高亮显示动作轨迹"""
        from_pos, to_pos, block_pos = action

        # 转换坐标（考虑边距）
        def pos_to_coord(pos):
            x = pos % 8
            y = pos // 8
            return (self.left_margin + x * self.cell_size + self.cell_size // 2, self.top_margin + y * self.cell_size + self.cell_size // 2)

        # 绘制移动箭头
        fx, fy = pos_to_coord(from_pos)
        tx, ty = pos_to_coord(to_pos)
        self.canvas.create_line(fx, fy, tx, ty, arrow=tk.LAST, fill="#FF4500", width=3, arrowshape=(12, 15, 6))

        # 标记障碍位置
        bx, by = pos_to_coord(block_pos)
        self.canvas.create_oval(bx - self.block_radius, by - self.block_radius, bx + self.block_radius, by + self.block_radius, outline="#FFD700", width=3)

    # ---------- 控制逻辑 ----------
    def update_step_label(self):
        self.step_label.config(text=f"步数：{self.current_step+1}/{len(self.history)}")

    def on_slider_changed(self, value):
        new_step = int(float(value))
        if new_step != self.current_step:
            self.current_step = new_step
            self.update_step_label()
            self.draw_board()

    def first_step(self):
        self.current_step = 0
        self.slider.set(0)
        self.update_step_label()
        self.draw_board()

    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.slider.set(self.current_step)
            self.update_step_label()
            self.draw_board()

    def next_step(self):
        if self.current_step < len(self.history) - 1:
            self.current_step += 1
            self.slider.set(self.current_step)
            self.update_step_label()
            self.draw_board()

    def last_step(self):
        self.current_step = len(self.history) - 1
        self.slider.set(self.current_step)
        self.update_step_label()
        self.draw_board()

    def run(self):
        self.root.mainloop()
