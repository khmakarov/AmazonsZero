import random
import dearpygui.dearpygui as dpg
from src.utils.database import AmazonsDatabase
from core.cpp.build.Amazons import GameCore
from collections import defaultdict
import math
import numpy as np


class AmazonsVisualizerDPG:

    def __init__(self, game_id: int):
        self.db = AmazonsDatabase()
        self.game = GameCore()
        self.history = self.db.load_game(game_id)
        self.current_step = 0
        self.cell_size = 70
        self.board_size = self.cell_size * 8
        self.left_margin = 40
        self.top_margin = 40
        self.piece_radius = 30
        self.block_radius = 25
        self.window_width = self.left_margin + self.board_size + 60
        self.window_height = self.top_margin + self.board_size + 200
        self.control_height = 120
        self.control_top_padding = 20

        self.colors = {
            "board_bg": (222, 184, 135, 255),
            "ui_bg": (240, 240, 240, 255),
            "child_bg": (245, 245, 245, 255),
            "grid_line": (100, 100, 100, 255),
            "slider_bg": (230, 230, 230, 255),
            "button_bg": (220, 220, 220, 255),
            "button_hover": (200, 200, 200, 255),
            "button_active": (180, 180, 180, 255),
            "text": (47, 79, 79, 255),
            "black_piece": (0, 0, 0, 255),
            "white_piece": (255, 255, 255, 255),
            "block": (65, 105, 225, 255),
            "arrow": (255, 69, 0, 255),
            "green": (0, 180, 0, 200),
            "blue": (0, 100, 200, 200),
            "block_highlight": (255, 215, 0, 255)
        }

        self.heatmap_colors = {"from": ("#FFE5CC", "#FF8C00"), "to": ("#E6F2FF", "#0066CC"), "block": ("#E8FFE8", "#009900")}

        dpg.create_context()
        self._setup_font()
        self._setup_theme()
        self._create_main_window()

        dpg.create_viewport(title='Amazons Chess Replay', width=self.window_width, height=self.window_height)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def _setup_font(self):
        with dpg.font_registry():
            self.default_font = dpg.add_font("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf", 16)
            self.bold_font = dpg.add_font("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf", 16)
        dpg.bind_font(self.default_font)

    def _setup_theme(self):
        with dpg.theme() as main_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, self.colors["ui_bg"])
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, self.colors["child_bg"])
                dpg.add_theme_color(dpg.mvThemeCol_Text, self.colors["text"])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, self.colors["slider_bg"])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (220, 220, 220, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (210, 210, 210, 255))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (150, 150, 150, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (200, 200, 200, 100))

        with dpg.theme() as button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, self.colors["button_bg"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, self.colors["button_hover"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, self.colors["button_active"])
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
                dpg.add_theme_color(dpg.mvThemeCol_Border, (150, 150, 150, 255))

        with dpg.theme() as board_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, self.colors["board_bg"])

        with dpg.theme() as slider_theme:
            with dpg.theme_component(dpg.mvSliderInt):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (220, 220, 220, 150))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (200, 200, 200, 150))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (180, 180, 180, 150))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)

                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (80, 180, 250, 220))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (60, 160, 230, 220))
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 12)

        self.slider_theme = slider_theme
        self.main_theme = main_theme
        self.button_theme = button_theme
        self.board_theme = board_theme

    def _create_main_window(self):
        with dpg.window(
            tag="main_window", label="Amazons Replay", no_title_bar=True, no_scrollbar=True, no_move=True, no_resize=True, width=self.window_width, height=self.window_height
        ):
            dpg.bind_item_theme("main_window", self.main_theme)

            # 棋盘区域
            board_window_width = self.board_size + self.left_margin + 20
            board_window_height = self.board_size + self.top_margin + 20
            with dpg.child_window(tag="board_window", width=board_window_width + 20, height=board_window_height + 20, pos=(10, 10), no_scrollbar=True):
                dpg.bind_item_theme("board_window", self.board_theme)
                with dpg.drawlist(tag="main_canvas", width=board_window_width, height=board_window_height):
                    with dpg.draw_node(tag="background_layer"):
                        dpg.draw_rectangle(
                            (self.left_margin, self.top_margin), (self.left_margin + self.board_size + 20, self.top_margin + self.board_size + 20),
                            fill=self.colors["board_bg"],
                            color=(0, 0, 0, 0)
                        )
                        self._draw_static_background()
                    dpg.add_draw_layer(tag="dynamic_layer")

            # 控制面板区域
            control_window_y = 10 + board_window_height + self.control_top_padding
            control_window_width = self.window_width - 20
            control_window_height = self.control_height

            with dpg.child_window(tag="control_window", width=control_window_width, height=control_window_height, pos=[10, control_window_y], no_scrollbar=True):
                # 进度条
                slider_width = control_window_width - 60
                slider_center_x = 60 + slider_width // 2

                # 滑块
                dpg.add_slider_int(
                    tag="step_slider",
                    width=slider_width,
                    pos=[30, 55],  # 下移避免重叠
                    max_value=len(self.history) - 1,
                    callback=lambda s: self.set_step(dpg.get_value(s)),
                    format="",
                    height=20
                )
                dpg.bind_item_theme("step_slider", self.slider_theme)

                # 状态文本
                dpg.add_text(tag="status_text", default_value=f"Step: {self.current_step+1}/{len(self.history)}", pos=[slider_center_x - 60, 15])

                # 控制按钮
                button_width = 70
                button_height = 25
                button_spacing = 6
                total_buttons_width = button_width * 4 + button_spacing * 3
                buttons_start_x = (control_window_width - total_buttons_width) // 2

                with dpg.group(horizontal=True, pos=[buttons_start_x, 85], horizontal_spacing=button_spacing):  # 调整垂直位置
                    buttons = [
                        ("<< Start", lambda: self.set_step(0)), ("< Prev", self.prev_step), ("Next >", self.next_step), ("End >>", lambda: self.set_step(len(self.history) - 1))
                    ]
                    for label, callback in buttons:
                        btn = dpg.add_button(label=label, width=button_width, height=button_height, callback=callback)
                        dpg.bind_item_theme(btn, self.button_theme)

        self.update_display()

    def _draw_static_background(self):
        for i in range(9):
            pos = i * self.cell_size + self.top_margin
            dpg.draw_line((self.left_margin, pos), (self.left_margin + self.board_size, pos), color=self.colors["grid_line"], thickness=1)
            dpg.draw_line(
                (self.left_margin + i * self.cell_size, self.top_margin), (self.left_margin + i * self.cell_size, self.top_margin + self.board_size),
                color=self.colors["grid_line"],
                thickness=1
            )

        for x in range(8):
            label_x = self.left_margin + x * self.cell_size + self.cell_size // 2
            label_y = self.top_margin // 2
            dpg.draw_text((label_x, label_y), text=chr(65 + x), color=self.colors["text"], size=16)

        for y in range(8):
            label_x = self.left_margin // 2
            label_y = self.top_margin + y * self.cell_size + self.cell_size // 2
            dpg.draw_text((label_x, label_y), text=str(y + 1), color=self.colors["text"], size=16)

    def _convert_coords(self, x, y):
        return (self.left_margin + x * self.cell_size, self.top_margin + y * self.cell_size)

    def update_display(self):
        dpg.delete_item("dynamic_layer", children_only=True)

        with dpg.draw_node(parent="dynamic_layer"):
            state = self.history[self.current_step][0]
            action = self.history[self.current_step - 1][1] if self.current_step > 0 else None
            top5_pi = self.history[self.current_step][2]

            self._draw_pieces(state)
            if action:
                self._highlight_action(action)
            if top5_pi:
                self._draw_heatmap(top5_pi)

        dpg.set_value("status_text", f"Step: {self.current_step}/{len(self.history)-1}")
        dpg.set_value("step_slider", self.current_step)

    def _draw_pieces(self, state):
        for y in range(8):
            for x in range(8):
                cell = state[y][x]
                px, py = self._convert_coords(x, y)
                center = (px + self.cell_size // 2, py + self.cell_size // 2)

                if cell[2] == 1:
                    dpg.draw_circle((center[0] + 2, center[1] + 2), self.block_radius, color=(0, 0, 0, 80), fill=(0, 0, 0, 80))
                    dpg.draw_circle(center, self.block_radius, color=self.colors["block"], fill=self.colors["block"])

                if cell[0] == 1:
                    dpg.draw_circle((center[0] + 2, center[1] + 2), self.piece_radius, color=(0, 0, 0, 60), fill=(0, 0, 0, 60))
                    dpg.draw_circle(center, self.piece_radius, color=self.colors["black_piece"], fill=self.colors["black_piece"])

                if cell[1] == 1:
                    dpg.draw_circle((center[0] + 2, center[1] + 2), self.piece_radius, color=(0, 0, 0, 40), fill=(0, 0, 0, 40))
                    dpg.draw_circle(center, self.piece_radius, color=self.colors["text"], fill=self.colors["white_piece"], thickness=2)

    def _highlight_action(self, action):
        from_pos, to_pos, block_pos = action

        def pos_to_coord(pos):
            x = pos % 8
            y = pos // 8
            px, py = self._convert_coords(x, y)
            return (px + self.cell_size // 2, py + self.cell_size // 2)

        def get_edge_point(start, end, radius):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            if dx == 0 and dy == 0:
                return start
            distance = math.hypot(dx, dy)
            ratio = radius / distance
            return (start[0] + dx * ratio, start[1] + dy * ratio)

        def are_collinear(p1, p2, p3, epsilon=1e-9):
            """
            判断三个点是否共线
            :param p1: 第一个点的坐标 (x, y)
            :param p2: 第二个点的坐标 (x, y)
            :param p3: 第三个点的坐标 (x, y)
            :param epsilon: 浮点数计算精度容差
            :return: 布尔值，True表示共线
            """
            # 计算向量面积公式：(x2 - x1)(y3 - y1) - (y2 - y1)(x3 - x1)
            area = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            return abs(area) < epsilon  # 考虑浮点数精度误差

        fx_center, fy_center = pos_to_coord(from_pos)
        tx_center, ty_center = pos_to_coord(to_pos)
        bx_center, by_center = pos_to_coord(block_pos)

        # 检测from和block是否重叠
        offset = 8  # 偏移像素量
        apply_offset = are_collinear((fx_center, fy_center), (tx_center, ty_center), (bx_center, by_center))

        # 绘制block高亮
        dpg.draw_circle((bx_center, by_center), self.block_radius, color=self.colors["block_highlight"], thickness=3)

        # 调整ft箭头
        start = get_edge_point((fx_center, fy_center), (tx_center, ty_center), self.piece_radius)
        end = get_edge_point((tx_center, ty_center), (fx_center, fy_center), self.piece_radius)
        if apply_offset:
            if (fx_center == tx_center):
                start = (start[0] - offset, start[1])
                end = (end[0] - offset, end[1])
            else:
                start = (start[0], start[1] - offset)
                end = (end[0], end[1] - offset)
        self._draw_arrow(start, end, self.colors["green"], thickness=3)

        # 调整tb箭头
        block_end = get_edge_point((bx_center, by_center), (tx_center, ty_center), self.piece_radius)
        block_start = get_edge_point((tx_center, ty_center), (bx_center, by_center), self.piece_radius)
        if apply_offset:  # 向下偏移
            if (tx_center == bx_center):
                block_start = (block_start[0] + offset, block_start[1])
                block_end = (block_end[0] + offset, block_end[1])
            else:
                block_start = (block_start[0], block_start[1] + offset)
                block_end = (block_end[0], block_end[1] + offset)
        self._draw_arrow(block_start, block_end, self.colors["blue"], thickness=3)

    def add_noise_with_constraint(self, top5_data, noise_std=0.005):
        if not top5_data:
            return []

        # 提取第一个元素的值作为上限
        v_max = top5_data[0][1]
        noisy_data = [top5_data[0]]  # 第一个元素保持不变

        for index, prob in top5_data[1:]:
            # 施加高斯噪声（可替换为其他噪声类型）
            noise = random.gauss(0, noise_std)
            noisy_prob = prob + noise

            # 截断到 [0, v_max] 范围内
            noisy_prob_clipped = max(0, min(noisy_prob, v_max))

            noisy_data.append((index, noisy_prob_clipped))

        return noisy_data

    def _draw_heatmap(self, top5_data):
        if not top5_data:
            return
        max_prob = max(prob for _, prob in top5_data)
        if max_prob >= 0.99:
            top5_data = [item for item in top5_data if item[1] >= 0.99]
        else:
            top5_data = sorted(top5_data, key=lambda x: x[1], reverse=True)[:5]
        ft_paths = defaultdict(float)
        tb_paths = defaultdict(float)

        for action_idx, prob in top5_data:
            try:
                f, t, b = self.game.index2action(action_idx)
                ft_paths[(f, t)] += prob
                tb_paths[(t, b)] += prob
            except:
                continue

        self._draw_path_arrows(ft_paths, arrow_type="ft")
        self._draw_path_arrows(tb_paths, arrow_type="tb")
        from_probs = defaultdict(float)
        to_probs = defaultdict(float)
        block_probs = defaultdict(float)

        for action_idx, prob in top5_data:
            try:
                [f, t, b] = self.game.index2action(action_idx)
                if 0 <= f < 64:
                    from_probs[f] += prob
                if 0 <= t < 64:
                    to_probs[t] += prob
                if 0 <= b < 64:
                    block_probs[b] += prob
            except:
                continue

        heatmap_radius = self.piece_radius * 0.9
        positions = set()
        role_map = defaultdict(list)

        for pos in from_probs:
            positions.add(pos)
            role_map[pos].append(("from", from_probs[pos]))
        for pos in to_probs:
            positions.add(pos)
            role_map[pos].append(("to", to_probs[pos]))
        for pos in block_probs:
            positions.add(pos)
            role_map[pos].append(("block", block_probs[pos]))

        for pos in positions:
            roles = role_map[pos]
            x = pos % 8
            y = pos // 8
            px, py = self._convert_coords(x, y)
            center = (px + self.cell_size // 2, py + self.cell_size // 2)

            # 统一处理白色背景
            has_from = any(role == "from" for role, _ in roles)
            if has_from and self.current_step % 2 == 0:
                dpg.draw_circle(center, heatmap_radius, fill=(255, 255, 255, 200), color=(255, 255, 255, 0))

            if len(roles) == 1:
                role, prob = roles[0]
                self._draw_single_heatmap_circle(center, prob, role, heatmap_radius)
            else:
                self._draw_split_heatmap(center, roles, heatmap_radius)

    def _draw_path_arrows(self, path_dict, arrow_type):
        arrow_style = {"ft": {"color": (0, 180, 0, 200), "thickness": 3, "offset_dir": 1}, "tb": {"color": (0, 100, 200, 200), "thickness": 3, "offset_dir": -1}}[arrow_type]

        grouped_paths = defaultdict(list)
        for (start, end), prob in path_dict.items():
            grouped_paths[(start, end)].append(prob)

        for (start, end), probs in grouped_paths.items():
            sorted_probs = sorted(probs, reverse=True)
            for i, prob in enumerate(sorted_probs):
                start_center = self._index_to_center(start)
                end_center = self._index_to_center(end)

                rank_offset = i * 12
                dynamic_alpha = int(200 * prob + 55)
                color = (*arrow_style["color"][:3], dynamic_alpha)

                self._draw_bezier_arrow(
                    start_center, end_center, color=color, thickness=arrow_style["thickness"], head_size=12 + 8 * prob, offset_dir=arrow_style["offset_dir"] * rank_offset
                )

    def _draw_bezier_arrow(self, start, end, color, thickness, head_size, offset_dir=0):

        def get_edge_point(center, target, radius):
            dx = target[0] - center[0]
            dy = target[1] - center[1]
            if dx == 0 and dy == 0:
                return center
            distance = math.hypot(dx, dy)
            ratio = (radius - 3) / distance
            return (center[0] + dx * ratio, center[1] + dy * ratio)

        original_start = start
        original_end = end
        start = get_edge_point(start, end, self.piece_radius)
        end = get_edge_point(end, start, self.piece_radius)

        dx = original_end[0] - original_start[0]
        dy = original_end[1] - original_start[1]
        distance = math.hypot(dx, dy)
        if distance == 0:
            return

        max_offset = 15 + abs(offset_dir)
        perpendicular = (-dy, dx) if offset_dir >= 0 else (dy, -dx)
        per_length = math.hypot(perpendicular[0], perpendicular[1])
        if per_length == 0:
            per_norm = (0, 0)
        else:
            per_norm = (perpendicular[0] / per_length * max_offset, perpendicular[1] / per_length * max_offset)

        start = (start[0] + per_norm[0], start[1] + per_norm[1])
        end = (end[0] + per_norm[0], end[1] + per_norm[1])

        # 计算总距离和方向
        dx_total = end[0] - start[0]
        dy_total = end[1] - start[1]
        total_distance = math.hypot(dx_total, dy_total)
        if total_distance == 0:
            return

        # 固定虚线和间隔长度
        dash_length = 8
        gap_length = 6
        segment_length = dash_length + gap_length

        # 计算相位偏移以确保同一路径上的虚线对齐
        phase_offset = (start[0] + start[1]) % segment_length  # 基于起点坐标生成相位
        current_pos = phase_offset - segment_length  # 确保从有效位置开始
        draw_dash = False

        # 调整初始状态
        while current_pos < -gap_length:
            current_pos += segment_length
            draw_dash = not draw_dash

        # 绘制虚线
        while current_pos < total_distance:
            if draw_dash:
                seg_start = max(current_pos, 0)
                seg_end = min(current_pos + dash_length, total_distance)
                if seg_start < seg_end:
                    start_pt = (start[0] + dx_total * (seg_start / total_distance), start[1] + dy_total * (seg_start / total_distance))
                    end_pt = (start[0] + dx_total * (seg_end / total_distance), start[1] + dy_total * (seg_end / total_distance))
                    dpg.draw_line(start_pt, end_pt, color=color, thickness=thickness)
            current_pos += dash_length if draw_dash else gap_length
            draw_dash = not draw_dash

        # 绘制箭头头部
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        head_size = max(8, head_size * 0.7)
        head_color = (color[0], color[1], color[2], 255)

        for theta in [math.pi / 6, -math.pi / 6]:
            arrow_tip = (end[0] - head_size * math.cos(angle + theta), end[1] - head_size * math.sin(angle + theta))
            dpg.draw_line(end, arrow_tip, color=head_color, thickness=thickness - 1)

        # 加强箭头根部
        strengthen_length = head_size * 0.6
        strengthen_pt = (end[0] - dx_total * (strengthen_length / total_distance), end[1] - dy_total * (strengthen_length / total_distance))
        dpg.draw_line(strengthen_pt, end, color=head_color, thickness=thickness)

    def _draw_arrow(self, start, end, color, thickness=4):
        dpg.draw_line(start, end, color=color, thickness=thickness)

        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        arrow_size = 12

        for theta in [math.pi / 6, -math.pi / 6]:
            px = end[0] - arrow_size * math.cos(angle + theta)
            py = end[1] - arrow_size * math.sin(angle + theta)
            dpg.draw_line(end, (px, py), color=color, thickness=thickness)

    def _index_to_center(self, index):
        x = index % 8
        y = index // 8
        px, py = self._convert_coords(x, y)
        return (px + self.cell_size // 2, py + self.cell_size // 2)

    def _draw_single_heatmap_circle(self, center, prob, role, radius):
        color_map = {"from": (255, 140, 0, int(155 * prob + 70)), "to": (0, 153, 0, int(155 * prob + 70)), "block": (0, 102, 204, int(155 * prob + 70))}
        color = color_map[role]

        dpg.draw_circle(center, radius, fill=color, color=(255, 255, 255, 100))

        if prob > 0.00001:
            text = f"{prob*100:.2f}%"
            text_width = (len(text) - 1) * 7 + 3
            text_pos = (center[0] - text_width / 2, center[1] - 6)
            text_color = (0, 0, 0, 255)
            dpg.draw_text(text_pos, text=text, color=text_color, size=14)

    def _draw_heatmap_sector(self, center, radius, start_angle, end_angle, prob, role):
        color_map = {"from": (255, 140, 0, int(155 * prob + 70)), "to": (0, 153, 0, int(155 * prob + 70)), "block": (0, 102, 204, int(155 * prob + 70))}
        color = color_map[role]

        points = [center]
        steps = 500

        for i in range(steps + 1):
            angle = math.radians(start_angle + (end_angle - start_angle) * i / steps)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))

        points.append(center)
        dpg.draw_polygon(points, fill=color, color=(255, 255, 255, 100))

        if prob > 0.00001:
            mid_angle = math.radians((start_angle + end_angle) / 2)
            text = f"{prob*100:.2f}%"
            text_width = (len(text) - 1) * 7 + 3

            text_x = center[0] + radius * 0.5 * math.cos(mid_angle) + 2
            text_y = center[1] + radius * 0.5 * math.sin(mid_angle)

            text_x -= text_width / 2

            if 0 <= mid_angle < math.pi:
                text_y -= 8
            else:
                text_y -= 2

            text_x = max(center[0] - radius, min(center[0] + radius, text_x))
            text_y = max(center[1] - radius, min(center[1] + radius, text_y))

            text_color = (0, 0, 0, 255)
            dpg.draw_text((text_x, text_y), text=text, color=text_color, size=12)

    def _draw_split_heatmap(self, center, roles, radius):
        roles = sorted(roles, key=lambda x: x[0])

        angle_step = 360 / len(roles)
        start_angle = 0

        for role, prob in roles:
            end_angle = start_angle + angle_step
            self._draw_heatmap_sector(center, radius, start_angle, end_angle, prob, role)
            start_angle = end_angle

    def _index_to_pos(self, index):
        return (index % 8, index // 8)

    def set_step(self, step):
        self.current_step = max(0, min(step, len(self.history) - 1))
        self.update_display()

    def prev_step(self):
        self.set_step(self.current_step - 1)

    def next_step(self):
        self.set_step(self.current_step + 1)

    def run(self):
        dpg.start_dearpygui()
        dpg.destroy_context()

    @classmethod
    def from_recent_game(cls):
        db = AmazonsDatabase()
        games = db.query_games(limit=1)
        if not games:
            raise ValueError("No recent games found")
        return cls(games[0]["game_id"])
