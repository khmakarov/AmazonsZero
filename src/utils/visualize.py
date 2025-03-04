import tkinter as tk
from tkinter import ttk
import numpy as np


class AmazonsVisualizer:

    def __init__(self, episode_history):
        self.root = tk.Tk()
        self.root.title("亚马逊棋对弈回放 - 增强版")

        # 初始化参数
        self.history = episode_history
        self.current_step = 0
        self.cell_size = 80  # 增大单元格尺寸
        self.piece_radius = 25  # 增大棋子半径
        self.block_radius = 20  # 障碍半径

        # 创建主画布
        self.canvas = tk.Canvas(
            self.root,
            width=self.cell_size * 8,
            height=self.cell_size * 8,
            bg="#DEB887"  # 木质棋盘颜色
        )
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # 创建控制面板
        control_frame = ttk.Frame(self.root)
        control_frame.grid(row=1, column=0, sticky="ew")

        # 步数显示
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

        # 初始化绘制
        self.update_step_label()
        self.draw_board()

    def draw_board(self):
        """绘制棋盘基础网格"""
        self.canvas.delete("all")

        # 绘制棋盘网格
        for i in range(9):
            line_pos = i * self.cell_size
            self.canvas.create_line(0, line_pos, self.cell_size * 8, line_pos)  # 水平线
            self.canvas.create_line(line_pos, 0, line_pos, self.cell_size * 8)  # 垂直线

        # 绘制坐标标签
        for i in range(8):
            # 列坐标 (A-H)
            self.canvas.create_text(i * self.cell_size + self.cell_size // 2, self.cell_size * 8 - 15, text=chr(65 + i))
            # 行坐标 (1-8)
            self.canvas.create_text(15, i * self.cell_size + self.cell_size // 2, text=str(8 - i))

        # 绘制当前状态
        self.draw_state()

    def draw_state(self):
        """绘制棋子和障碍"""
        state = self.history[self.current_step][0]
        grid = np.array(state.get_state())

        for y in range(8):
            for x in range(8):
                cell = grid[y][x]
                cx = x * self.cell_size + self.cell_size // 2
                cy = y * self.cell_size + self.cell_size // 2

                # 绘制障碍（蓝色圆形）
                if cell[2] == 1:
                    self.canvas.create_oval(
                        cx - self.block_radius,
                        cy - self.block_radius,
                        cx + self.block_radius,
                        cy + self.block_radius,
                        fill="#4169E1",  # 皇家蓝
                        outline="black"
                    )

                # 绘制黑棋（黑色圆形）
                if cell[0] == 1:
                    self.canvas.create_oval(cx - self.piece_radius, cy - self.piece_radius, cx + self.piece_radius, cy + self.piece_radius, fill="black", outline="gold", width=2)

                # 绘制白棋（白色圆形）
                if cell[1] == 1:
                    self.canvas.create_oval(cx - self.piece_radius, cy - self.piece_radius, cx + self.piece_radius, cy + self.piece_radius, fill="white", outline="gold", width=2)

        if self.current_step > 0:
            prev_action = self.history[self.current_step - 1][1]
            if prev_action is not None:
                self.highlight_action(prev_action)

    def highlight_action(self, action):
        """高亮显示动作"""
        from_pos, to_pos, block_pos = action

        # 转换坐标
        fx = (from_pos % 8) * self.cell_size + self.cell_size // 2
        fy = (from_pos // 8) * self.cell_size + self.cell_size // 2
        tx = (to_pos % 8) * self.cell_size + self.cell_size // 2
        ty = (to_pos // 8) * self.cell_size + self.cell_size // 2
        bx = (block_pos % 8) * self.cell_size + self.cell_size // 2
        by = (block_pos // 8) * self.cell_size + self.cell_size // 2

        # 绘制移动箭头
        self.canvas.create_line(fx, fy, tx, ty, arrow=tk.LAST, fill="#FF4500", width=3, arrowshape=(12, 15, 6))

        # 标记障碍位置
        self.canvas.create_oval(
            bx - self.block_radius,
            by - self.block_radius,
            bx + self.block_radius,
            by + self.block_radius,
            outline="#FFD700",  # 金色边框
            width=3
        )

    def update_step_label(self):
        """更新步数显示"""
        self.step_label.config(text=f"步数：{self.current_step+1}/{len(self.history)}")

    # 控制逻辑
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
