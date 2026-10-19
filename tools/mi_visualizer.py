#!/usr/bin/env python3
"""
Mutual Information & MRMR Feature Selection Visualizer for Minesweeper sliding windows.
Features:
- Heatmap from dark blue/black to light blue for unselected cells (MI relevance)
- Heatmap from dark green to light/bright green for selected cells (MRMR selected set)
- Interactive step-by-step MRMR feature selection controls (Next, Prev, Auto-play, Reset)
- Selection order badges (#1, #2, #3, ...) and live statistics
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from sklearn.metrics import mutual_info_score

# Ensure root directory is on python path for importing stats
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stats.mrmr import select_next_best, calculate_redundancy


def get_blue_gradient(t: float) -> tuple[str, str]:
    """
    Interpolates unselected cell color: dark blue/black -> deep blue -> light sky blue.
    Returns (bg_hex, fg_hex).
    """
    t = max(0.0, min(1.0, float(t)))
    stops = [
        (0.00, (8, 14, 26)),
        (0.35, (14, 56, 122)),
        (0.70, (41, 182, 246)),
        (1.00, (225, 245, 254))
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            local_t = (t - t0) / (t1 - t0)
            r = int(c0[0] + local_t * (c1[0] - c0[0]))
            g = int(c0[1] + local_t * (c1[1] - c0[1]))
            b = int(c0[2] + local_t * (c1[2] - c0[2]))
            break
    else:
        r, g, b = stops[-1][1]

    bg_hex = f"#{r:02x}{g:02x}{b:02x}"
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    fg_hex = "#0A192F" if luminance > 0.55 else "#E2E8F0"
    return bg_hex, fg_hex


def get_green_gradient(t: float) -> tuple[str, str]:
    """
    Interpolates selected cell color: dark pine green -> emerald green -> bright mint/lime green.
    Returns (bg_hex, fg_hex).
    """
    t = max(0.0, min(1.0, float(t)))
    stops = [
        (0.00, (6, 32, 16)),
        (0.35, (16, 92, 44)),
        (0.70, (38, 186, 92)),
        (1.00, (210, 255, 220))
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            local_t = (t - t0) / (t1 - t0)
            r = int(c0[0] + local_t * (c1[0] - c0[0]))
            g = int(c0[1] + local_t * (c1[1] - c0[1]))
            b = int(c0[2] + local_t * (c1[2] - c0[2]))
            break
    else:
        r, g, b = stops[-1][1]

    bg_hex = f"#{r:02x}{g:02x}{b:02x}"
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    fg_hex = "#041C0E" if luminance > 0.55 else "#E6FBF0"
    return bg_hex, fg_hex


class MutualInformationVisualizerApp:
    def __init__(self, root, mi_path="mutual_information.npz", dataset_path="cos700_dataset.npz"):
        self.root = root
        self.root.title("Mutual Information & MRMR Visualizer")
        self.root.geometry("960x700")
        self.root.minsize(860, 620)

        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.mi_path = mi_path
        self.dataset_path = dataset_path

        self.mi_matrix = None
        self.raw_windows = None
        self.hovered_cell = None

        # MRMR Stepping state: list of dicts: {"cell": (r, c), "score": float, "mi": float, "redundancy": float}
        self.selection_history = []
        self.is_playing = False
        self.play_job = None

        self._create_widgets()
        self.load_data(self.mi_path, self.dataset_path)

    def _create_widgets(self):
        # 1. Top Header & Files
        header_frame = ttk.Frame(self.root, padding="10 8 10 4")
        header_frame.pack(fill=tk.X)

        self.lbl_files = ttk.Label(
            header_frame,
            text=f"MI File: {os.path.basename(self.mi_path)} | Dataset: {os.path.basename(self.dataset_path)}",
            font=("Helvetica", 10, "bold")
        )
        self.lbl_files.pack(side=tk.LEFT)

        btn_box = ttk.Frame(header_frame)
        btn_box.pack(side=tk.RIGHT)
        ttk.Button(btn_box, text="Open MI .npz...", command=self.open_mi_dialog).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_box, text="Open Dataset...", command=self.open_dataset_dialog).pack(side=tk.LEFT, padx=3)

        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

        # 2. Main Body (Left: Heatmap & Step Controls, Right: MRMR Stats & History)
        content_frame = ttk.Frame(self.root, padding="10")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left Column: Canvas & Playback Bar
        left_col = ttk.Frame(content_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        heatmap_box = ttk.LabelFrame(left_col, text="Sliding Window Heatmap (Blue: Unselected, Green: Selected)", padding="8")
        heatmap_box.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(heatmap_box, bg="#0D1117", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

        # Step Controls Toolbar
        step_bar = ttk.LabelFrame(left_col, text="MRMR Stepping Controls", padding="8")
        step_bar.pack(fill=tk.X, pady=(8, 0))

        self.btn_reset = ttk.Button(step_bar, text="⏮ Reset", width=8, command=self.reset_selection)
        self.btn_reset.pack(side=tk.LEFT, padx=3)

        self.btn_prev = ttk.Button(step_bar, text="◀ Prev", width=7, command=self.step_prev)
        self.btn_prev.pack(side=tk.LEFT, padx=3)

        self.btn_next = ttk.Button(step_bar, text="Next ▶", width=7, command=self.step_next)
        self.btn_next.pack(side=tk.LEFT, padx=3)

        self.btn_play = ttk.Button(step_bar, text="▶ Auto Play", width=11, command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=3)

        self.btn_top10 = ttk.Button(step_bar, text="+10 Steps", width=9, command=lambda: self.step_forward_n(10))
        self.btn_top10.pack(side=tk.LEFT, padx=3)

        self.lbl_step_counter = ttk.Label(step_bar, text="Selected: 0 / 81", font=("Helvetica", 10, "bold"))
        self.lbl_step_counter.pack(side=tk.RIGHT, padx=6)

        # Right Column: Side Details & Selection Order
        right_col = ttk.Frame(content_frame, width=340)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_col.pack_propagate(False)

        # Current Cell Inspector
        inspector_frame = ttk.LabelFrame(right_col, text="Cell Inspector (Hover)", padding="10")
        inspector_frame.pack(fill=tk.X, pady=(0, 8))

        self.lbl_cell_pos = ttk.Label(inspector_frame, text="Window Position: -", font=("Helvetica", 10))
        self.lbl_cell_pos.pack(anchor=tk.W, pady=1)

        self.lbl_cell_status = ttk.Label(inspector_frame, text="Status: Unselected", font=("Helvetica", 10, "bold"), foreground="#4FC3F7")
        self.lbl_cell_status.pack(anchor=tk.W, pady=1)

        self.lbl_cell_mi = ttk.Label(inspector_frame, text="MI (Relevance): -", font=("Helvetica", 10))
        self.lbl_cell_mi.pack(anchor=tk.W, pady=1)

        self.lbl_cell_red = ttk.Label(inspector_frame, text="Avg Redundancy: -", font=("Helvetica", 10))
        self.lbl_cell_red.pack(anchor=tk.W, pady=1)

        self.lbl_cell_mrmr = ttk.Label(inspector_frame, text="MRMR Score: -", font=("Helvetica", 10, "bold"))
        self.lbl_cell_mrmr.pack(anchor=tk.W, pady=1)

        # Selection History Table
        history_frame = ttk.LabelFrame(right_col, text="Selected Feature Order (MRMR)", padding="8")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        columns = ("rank", "cell", "rel", "mrmr")
        self.tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=8)
        self.tree.heading("rank", text="#")
        self.tree.heading("cell", text="Cell")
        self.tree.heading("rel", text="MI")
        self.tree.heading("mrmr", text="MRMR")

        self.tree.column("rank", width=38, anchor=tk.CENTER)
        self.tree.column("cell", width=68, anchor=tk.CENTER)
        self.tree.column("rel", width=75, anchor=tk.E)
        self.tree.column("mrmr", width=75, anchor=tk.E)

        tree_scroll = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Legend / Gradient Scale
        legend_frame = ttk.LabelFrame(right_col, text="Legend & Color Scales", padding="8")
        legend_frame.pack(fill=tk.X)

        ttk.Label(legend_frame, text="Unselected: Dark Blue → Light Blue", font=("Helvetica", 8)).pack(anchor=tk.W)
        self.blue_bar = tk.Canvas(legend_frame, height=12, bg="#0D1117", highlightthickness=1, highlightbackground="#30363D")
        self.blue_bar.pack(fill=tk.X, pady=(1, 4))

        ttk.Label(legend_frame, text="Selected: Dark Green → Light Green", font=("Helvetica", 8)).pack(anchor=tk.W)
        self.green_bar = tk.Canvas(legend_frame, height=12, bg="#0D1117", highlightthickness=1, highlightbackground="#30363D")
        self.green_bar.pack(fill=tk.X, pady=(1, 2))

    def open_mi_dialog(self):
        filename = filedialog.askopenfilename(
            title="Open Mutual Information File",
            filetypes=[("NumPy Compressed Archive", "*.npz"), ("All Files", "*.*")]
        )
        if filename:
            self.load_data(filename, self.dataset_path)

    def open_dataset_dialog(self):
        filename = filedialog.askopenfilename(
            title="Open Dataset File (for Redundancy)",
            filetypes=[("NumPy Compressed Archive", "*.npz"), ("All Files", "*.*")]
        )
        if filename:
            self.load_data(self.mi_path, filename)

    def load_data(self, mi_filepath, dataset_filepath):
        # 1. Load MI Matrix
        if not os.path.exists(mi_filepath):
            messagebox.showwarning("MI File Not Found", f"Could not find MI file:\n{mi_filepath}")
            return

        try:
            with np.load(mi_filepath) as data:
                if 'mi_matrix' in data:
                    raw_matrix = data['mi_matrix']
                elif 'scores' in data:
                    scores = data['scores']
                    dim = int(np.sqrt(len(scores)))
                    raw_matrix = scores.reshape(dim, dim)
                else:
                    first_key = list(data.keys())[0]
                    raw_matrix = data[first_key]

            self.mi_matrix = np.asarray(raw_matrix, dtype=float)
            self.min_val = float(np.min(self.mi_matrix))
            self.max_val = float(np.max(self.mi_matrix))
            self.mi_path = mi_filepath
        except Exception as e:
            messagebox.showerror("Error Loading MI", f"Failed to load {mi_filepath}:\n{str(e)}")
            return

        # 2. Load Raw Dataset for Redundancy Calculation
        if os.path.exists(dataset_filepath):
            try:
                with np.load(dataset_filepath) as data:
                    windows = data['windows']
                    # Flatten windows to 2D
                    num_samples, h, w = windows.shape
                    flat = windows.reshape(num_samples, h * w)

                    # Subsample up to 50k rows for instantaneous MRMR calculations
                    if num_samples > 50000:
                        rng = np.random.RandomState(42)
                        idx = rng.choice(num_samples, size=50000, replace=False)
                        self.raw_windows = flat[idx]
                    else:
                        self.raw_windows = flat
                self.dataset_path = dataset_filepath
            except Exception as e:
                print(f"Warning: Could not load dataset for redundancy: {e}")
                self.raw_windows = None
        else:
            self.raw_windows = None

        self.lbl_files.config(
            text=f"MI: {os.path.basename(self.mi_path)} | Data: {os.path.basename(self.dataset_path)}"
        )

        self.reset_selection()
        self._draw_colorbars()
        self._draw_heatmap()

    def get_selectable_count(self):
        if self.mi_matrix is None:
            return 0
        rows, cols = self.mi_matrix.shape
        # Inner cells excluding the center target cell
        return max(0, (rows - 2) * (cols - 2) - 1)

    def reset_selection(self):
        if self.is_playing:
            self.toggle_play()
        self.selection_history.clear()
        self.tree.delete(*self.tree.get_children())
        total_selectable = self.get_selectable_count()
        self.lbl_step_counter.config(text=f"Selected: 0 / {total_selectable}")
        self._draw_heatmap()

    def step_next(self):
        if self.mi_matrix is None or self.raw_windows is None:
            if self.raw_windows is None:
                messagebox.showwarning("Dataset Missing", "Please open the dataset file (.npz) to compute redundancy.")
            return False

        total_selectable = self.get_selectable_count()
        if len(self.selection_history) >= total_selectable:
            if self.is_playing:
                self.toggle_play()
            return False

        selected_coords = [item["cell"] for item in self.selection_history]
        next_cell, best_score = select_next_best(self.raw_windows, self.mi_matrix, selected_coords, exclude_border=True)

        if next_cell is None:
            return False

        if isinstance(next_cell, (int, np.integer)):
            r, c = divmod(int(next_cell), self.mi_matrix.shape[1])
        else:
            r, c = next_cell
        mi_score = float(self.mi_matrix[r, c])

        if len(selected_coords) == 0:
            avg_red = 0.0
        else:
            cand_idx = r * self.mi_matrix.shape[1] + c
            reds = [
                calculate_redundancy(self.raw_windows, cand_idx, sr * self.mi_matrix.shape[1] + sc)
                for (sr, sc) in selected_coords
            ]
            avg_red = float(np.mean(reds))

        step_info = {
            "cell": (r, c),
            "score": best_score,
            "mi": mi_score,
            "redundancy": avg_red
        }
        self.selection_history.append(step_info)

        # Update table
        rank = len(self.selection_history)
        self.tree.insert("", "end", values=(f"#{rank}", f"({r},{c})", f"{mi_score:.4f}", f"{best_score:.4f}"))
        self.tree.yview_moveto(1.0)

        self.lbl_step_counter.config(text=f"Selected: {len(self.selection_history)} / {total_selectable}")
        self._draw_heatmap()
        return True

    def step_prev(self):
        if self.is_playing:
            self.toggle_play()
        if not self.selection_history:
            return

        self.selection_history.pop()
        children = self.tree.get_children()
        if children:
            self.tree.delete(children[-1])

        total_selectable = self.get_selectable_count()
        self.lbl_step_counter.config(text=f"Selected: {len(self.selection_history)} / {total_selectable}")
        self._draw_heatmap()

    def step_forward_n(self, n=10):
        for _ in range(n):
            if not self.step_next():
                break

    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.btn_play.config(text="▶ Auto Play")
            if self.play_job:
                self.root.after_cancel(self.play_job)
                self.play_job = None
        else:
            self.is_playing = True
            self.btn_play.config(text="⏸ Pause")
            self._auto_step()

    def _auto_step(self):
        if not self.is_playing:
            return
        has_next = self.step_next()
        if has_next and self.is_playing:
            self.play_job = self.root.after(350, self._auto_step)
        else:
            self.is_playing = False
            self.btn_play.config(text="▶ Auto Play")

    def _draw_colorbars(self):
        for canvas, grad_fn in [(self.blue_bar, get_blue_gradient), (self.green_bar, get_green_gradient)]:
            canvas.delete("all")
            w = canvas.winfo_width()
            if w < 10:
                w = 260
            h = 12
            for x in range(w):
                t = x / max(1, w - 1)
                bg, _ = grad_fn(t)
                canvas.create_line(x, 0, x, h, fill=bg)

    def _draw_heatmap(self):
        self.canvas.delete("all")
        if self.mi_matrix is None:
            return

        rows, cols = self.mi_matrix.shape
        center_r, center_c = rows // 2, cols // 2

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 50 or canvas_h < 50:
            canvas_w, canvas_h = 440, 440

        max_cell_w = (canvas_w - 40) // cols
        max_cell_h = (canvas_h - 40) // rows
        cell_size = max(24, min(max_cell_w, max_cell_h, 54))

        start_x = (canvas_w - (cols * cell_size)) // 2
        start_y = (canvas_h - (rows * cell_size)) // 2

        self.grid_meta = {
            "start_x": start_x,
            "start_y": start_y,
            "cell_size": cell_size,
            "rows": rows,
            "cols": cols
        }

        # Build lookup for selected cells: cell -> (rank, step_info)
        selected_lookup = {
            item["cell"]: (idx + 1, item)
            for idx, item in enumerate(self.selection_history)
        }

        font_size = max(8, int(cell_size * 0.23))
        font_style = ("Helvetica", font_size, "bold")
        badge_font = ("Helvetica", max(7, int(cell_size * 0.19)), "bold")

        for r in range(rows):
            for c in range(cols):
                val = float(self.mi_matrix[r, c])
                x1 = start_x + c * cell_size
                y1 = start_y + r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                is_center = (r == center_r and c == center_c)
                is_border = (r == 0 or r == rows - 1 or c == 0 or c == cols - 1)
                is_selected = (r, c) in selected_lookup

                t = (val - self.min_val) / max(1e-9, (self.max_val - self.min_val))

                if is_center:
                    # Distinct target center cell
                    bg_color = "#231F0A"
                    fg_color = "#FFD700"
                    border_color = "#FFD700"
                elif is_border:
                    # Muted styling for excluded outer border
                    bg_color = "#151B26"
                    fg_color = "#55657E"
                    border_color = "#21262D"
                elif is_selected:
                    bg_color, fg_color = get_green_gradient(t)
                    border_color = "#2ECC71"
                else:
                    bg_color, fg_color = get_blue_gradient(t)
                    border_color = "#161B22"

                # Draw cell
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=bg_color,
                    outline=border_color,
                    width=2 if (is_selected or is_center) else 1
                )

                # Format MI text
                if is_center:
                    text_val = "TARGET"
                    center_font = ("Helvetica", max(7, int(cell_size * 0.18)), "bold")
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text=text_val,
                        fill=fg_color,
                        font=center_font
                    )
                else:
                    text_val = f"{val:.3f}" if val >= 0.001 else f"{val:.1e}"
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2 + (4 if is_selected else 0),
                        text=text_val,
                        fill=fg_color,
                        font=font_style
                    )

                # Draw selection rank badge
                if is_selected:
                    rank, _ = selected_lookup[(r, c)]
                    badge_text = f"#{rank}"
                    self.canvas.create_text(
                        x1 + 14, y1 + 9,
                        text=badge_text,
                        fill="#00FF66" if fg_color == "#E6FBF0" else "#023D19",
                        font=badge_font
                    )

                # Highlight hovered cell
                if self.hovered_cell == (r, c):
                    self.canvas.create_rectangle(
                        x1 + 1, y1 + 1, x2 - 1, y2 - 1,
                        outline="#00E5FF",
                        width=2
                    )

    def _on_canvas_motion(self, event):
        if not hasattr(self, 'grid_meta') or self.mi_matrix is None:
            return

        sx = self.grid_meta["start_x"]
        sy = self.grid_meta["start_y"]
        cs = self.grid_meta["cell_size"]
        rows = self.grid_meta["rows"]
        cols = self.grid_meta["cols"]

        col = (event.x - sx) // cs
        row = (event.y - sy) // cs

        if 0 <= row < rows and 0 <= col < cols:
            if self.hovered_cell != (row, col):
                self.hovered_cell = (row, col)
                val = float(self.mi_matrix[row, col])
                cr, cc = rows // 2, cols // 2
                rel_r, rel_c = row - cr, col - cc

                is_center = (row == cr and col == cc)
                is_border = (row == 0 or row == rows - 1 or col == 0 or col == cols - 1)
                selected_dict = {item["cell"]: (idx + 1, item) for idx, item in enumerate(self.selection_history)}

                self.lbl_cell_pos.config(text=f"Window Position: ({row}, {col}) [Rel: {rel_r:+d}, {rel_c:+d}]")
                self.lbl_cell_mi.config(text=f"MI (Relevance): {val:.5f}")

                if is_center:
                    self.lbl_cell_status.config(text="Status: Target Cell (Center - Excluded)", foreground="#FFD700")
                    self.lbl_cell_red.config(text="Avg Redundancy: N/A (Target)")
                    self.lbl_cell_mrmr.config(text="MRMR Score: N/A (Target)")
                elif is_border:
                    self.lbl_cell_status.config(text="Status: Excluded (Outer Border)", foreground="#8B949E")
                    self.lbl_cell_red.config(text="Avg Redundancy: N/A (Excluded)")
                    self.lbl_cell_mrmr.config(text="MRMR Score: N/A (Excluded)")
                elif (row, col) in selected_dict:
                    rank, item = selected_dict[(row, col)]
                    self.lbl_cell_status.config(text=f"Status: Selected (Rank #{rank})", foreground="#2ECC71")
                    self.lbl_cell_red.config(text=f"Avg Redundancy: {item['redundancy']:.5f}")
                    self.lbl_cell_mrmr.config(text=f"MRMR Score: {item['score']:.5f}")
                else:
                    self.lbl_cell_status.config(text="Status: Candidate (Unselected)", foreground="#4FC3F7")
                    if self.selection_history and self.raw_windows is not None:
                        cand_idx = row * cols + col
                        selected_coords = [it["cell"] for it in self.selection_history]
                        reds = [
                            calculate_redundancy(self.raw_windows, cand_idx, sr * cols + sc)
                            for (sr, sc) in selected_coords
                        ]
                        cand_red = float(np.mean(reds))
                        cand_mrmr = val - cand_red
                        self.lbl_cell_red.config(text=f"Avg Redundancy: {cand_red:.5f}")
                        self.lbl_cell_mrmr.config(text=f"MRMR Score: {cand_mrmr:.5f}")
                    else:
                        self.lbl_cell_red.config(text="Avg Redundancy: 0.00000")
                        self.lbl_cell_mrmr.config(text=f"MRMR Score: {val:.5f}")

                self._draw_heatmap()
        else:
            self._on_canvas_leave(event)

    def _on_canvas_leave(self, event):
        if self.hovered_cell is not None:
            self.hovered_cell = None
            self.lbl_cell_pos.config(text="Window Position: -")
            self.lbl_cell_status.config(text="Status: -", foreground="#4FC3F7")
            self.lbl_cell_mi.config(text="MI (Relevance): -")
            self.lbl_cell_red.config(text="Avg Redundancy: -")
            self.lbl_cell_mrmr.config(text="MRMR Score: -")
            self._draw_heatmap()




def main():
    mi_file = sys.argv[1] if len(sys.argv) > 1 else "mutual_information.npz"
    dataset_file = sys.argv[2] if len(sys.argv) > 2 else "cos700_dataset.npz"

    root = tk.Tk()
    app = MutualInformationVisualizerApp(root, mi_path=mi_file, dataset_path=dataset_file)

    def on_resize(event):
        if app.mi_matrix is not None:
            app._draw_colorbars()
            app._draw_heatmap()

    app.canvas.bind("<Configure>", on_resize)
    app.blue_bar.bind("<Configure>", lambda e: app._draw_colorbars())
    root.mainloop()


if __name__ == "__main__":
    main()
