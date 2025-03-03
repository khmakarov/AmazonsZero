import tkinter as tk
from tkinter import ttk
import numpy as np
from Amazons import GameCore


class AmazonsVisualizer:

    def __init__(self, episode_history):
        self.root = tk.Tk()
        self.root.title("亚马逊棋对弈回放")
        self.history = episode_history
        self.current_step = 0

        # 棋盘参数调整
        self.canvas_size = 640  # 调整为64的倍数更易整除
        self.cell_size = self.canvas_size // 8
        self.piece_radius = int(self.cell_size * 0.3)  # 棋子半径（单元格大小的30%）
        self.block_radius = int(self.cell_size * 0.25)  # 障碍物稍小

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="#DEB887")  # 木质棋盘颜色
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # 创建控制面板
        self.controls = ttk.Frame(self.root)
        self.controls.grid(row=1, column=0, sticky="ew")

        # 进度条
        self.slider = ttk.Scale(self.controls, from_=0, to=len(self.history) - 1, command=self.on_slider_changed)
        self.slider.pack(fill="x", expand=True)

        # 控制按钮
        self.btn_frame = ttk.Frame(self.controls)
        self.btn_frame.pack(pady=5)

        ttk.Button(self.btn_frame, text="|<", command=self.first_step).pack(side="left")
        ttk.Button(self.btn_frame, text="<", command=self.prev_step).pack(side="left")
        ttk.Button(self.btn_frame, text=">", command=self.next_step).pack(side="left")
        ttk.Button(self.btn_frame, text=">|", command=self.last_step).pack(side="left")

        # 初始化棋盘
        self.draw_board()

    def draw_board(self):
        """绘制棋盘基础网格和坐标"""
        self.canvas.delete("all")  # 清空画布

        # 绘制棋盘网格
        for i in range(8):
            # 水平线
            self.canvas.create_line(0, i * self.cell_size, self.canvas_size, i * self.cell_size)
            # 垂直线
            self.canvas.create_line(i * self.cell_size, 0, i * self.cell_size, self.canvas_size)

        # 绘制坐标标签
        for i in range(8):
            # 列坐标 (A-H)
            self.canvas.create_text(i * self.cell_size + self.cell_size // 2, self.canvas_size - 15, text=chr(65 + i))
            # 行坐标 (1-8)
            self.canvas.create_text(15, i * self.cell_size + self.cell_size // 2, text=str(8 - i))

        # 绘制当前局面
        self.draw_state()

    def draw_state(self):
        self.canvas.delete("all")
        grid = np.array(self.history[self.current_step][0].get_state())

        for y in range(8):
            for x in range(8):
                # 计算中心坐标
                cx = x * self.cell_size + self.cell_size // 2
                cy = y * self.cell_size + self.cell_size // 2

                # === 障碍物绘制（改为蓝色圆形）===
                if grid[y][x][2] == 1:
                    self.canvas.create_oval(
                        cx - self.block_radius,
                        cy - self.block_radius,
                        cx + self.block_radius,
                        cy + self.block_radius,
                        fill="#1E90FF",
                        outline="#000080",
                        width=2  # 深蓝色系
                    )

                # === 棋子绘制（调整大小）===
                # 黑棋（带金色边框）
                if grid[y][x][0] == 1:
                    self.canvas.create_oval(cx - self.piece_radius, cy - self.piece_radius, cx + self.piece_radius, cy + self.piece_radius, fill="black", outline="#FFD700", width=3)
                # 白棋（带深灰边框）
                if grid[y][x][1] == 1:
                    self.canvas.create_oval(cx - self.piece_radius, cy - self.piece_radius, cx + self.piece_radius, cy + self.piece_radius, fill="white", outline="#404040", width=3)

        # === 移动轨迹绘制（改为粉色箭头）===
        if self.current_step > 0:
            action = self.history[self.current_step - 1][1]
            if action:
                self._draw_move_trace(action)

    def _draw_move_trace(self, action):
        """绘制移动轨迹和障碍放置"""
        from_pos, to_pos, block_pos = action

        # 转换坐标（假设位置用0-63整数表示）
        fx, fy = from_pos % 8, from_pos // 8
        tx, ty = to_pos % 8, to_pos // 8
        bx, by = block_pos % 8, block_pos // 8

        # 粉色移动箭头（添加虚线效果）
        self._draw_arrow(fx, fy, tx, ty, color="#FF69B4", width=4, arrowshape=(15, 15, 5), dash=(5, 3))

        # 障碍放置动画（闪烁效果）
        self.canvas.create_oval(
            (bx * self.cell_size) + self.cell_size // 2 - self.block_radius,
            (by * self.cell_size) + self.cell_size // 2 - self.block_radius,
            (bx * self.cell_size) + self.cell_size // 2 + self.block_radius,
            (by * self.cell_size) + self.cell_size // 2 + self.block_radius,
            outline="#FF4500",
            width=3,  # 橙色边框提示新障碍
            tags="block_highlight"
        )
        self.canvas.after(300, lambda: self.canvas.itemconfig("block_highlight", outline=""))

    def _draw_arrow(self, x1, y1, x2, y2, **kwargs):
        """通用箭头绘制方法"""
        start_x = x1 * self.cell_size + self.cell_size // 2
        start_y = y1 * self.cell_size + self.cell_size // 2
        end_x = x2 * self.cell_size + self.cell_size // 2
        end_y = y2 * self.cell_size + self.cell_size // 2

        self.canvas.create_line(start_x, start_y, end_x, end_y, arrow=tk.LAST, **kwargs)

    # 以下为控制逻辑
    def on_slider_changed(self, value):
        new_step = int(float(value))
        if new_step != self.current_step:
            self.current_step = new_step
            self.draw_board()

    def first_step(self):
        self.current_step = 0
        self.slider.set(0)
        self.draw_board()

    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.slider.set(self.current_step)
            self.draw_board()

    def next_step(self):
        if self.current_step < len(self.history) - 1:
            self.current_step += 1
            self.slider.set(self.current_step)
            self.draw_board()

    def last_step(self):
        self.current_step = len(self.history) - 1
        self.slider.set(self.current_step)
        self.draw_board()

    def run(self):
        self.root.mainloop()
