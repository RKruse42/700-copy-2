#!/usr/bin/env python3
"""
Theoretical Maximum Mutual Information Visualizer.
Allows interactive selection of arbitrary subsets of window features (by clicking cells),
calculating joint conditional entropy H(Mine | Features) and theoretical maximum
joint mutual information I(Features; Mine).
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

# Ensure root directory is on python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.theoretical_maximum import get_theoretical_maximum, get_conditional_entropy, binary_entropy, solve_fano


def get_blue_gradient(t: float) -> tuple[str, str]:
    """Interpolates unselected cell color: dark blue/black -> deep blue -> light sky blue."""
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
    """Interpolates selected cell color: dark pine green -> emerald green -> bright mint/lime green."""
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


class TheoreticalMaximumVisualizerApp:
    def __init__(self, root, dataset_path="cos700_dataset.npz", mi_path="mutual_information.npz"):
        self.root = root
        self.root.title("Theoretical Maximum Information Gain Visualizer")
        self.root.geometry("980x740")
        self.root.minsize(880, 640)

        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.dataset_path = dataset_path
        self.mi_path = mi_path

        self.windows = None
        self.is_mine = None
        self.mi_matrix = None
        self.prior_entropy = 0.0

        # Set of selected (row, col) feature tuples
        self.selected_features = []
        self.hovered_cell = None

        self._create_widgets()
        self.load_data(self.dataset_path, self.mi_path)

    def _create_widgets(self):
        # 1. Top Header & Files
        header_frame = ttk.Frame(self.root, padding="10 8 10 4")
        header_frame.pack(fill=tk.X)

        self.lbl_files = ttk.Label(
            header_frame,
            text=f"Dataset: {os.path.basename(self.dataset_path)} | MI File: {os.path.basename(self.mi_path)}",
            font=("Helvetica", 10, "bold")
        )
        self.lbl_files.pack(side=tk.LEFT)

        btn_box = ttk.Frame(header_frame)
        btn_box.pack(side=tk.RIGHT)
        ttk.Button(btn_box, text="Open Dataset...", command=self.open_dataset_dialog).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_box, text="Open MI .npz...", command=self.open_mi_dialog).pack(side=tk.LEFT, padx=3)

        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

        # 2. Main Content
        content_frame = ttk.Frame(self.root, padding="10")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left Column: Canvas & Preset Buttons
        left_col = ttk.Frame(content_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        heatmap_box = ttk.LabelFrame(left_col, text="Sliding Window (Click cells to add / remove features)", padding="8")
        heatmap_box.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(heatmap_box, bg="#0D1117", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

        # Preset Toolbar
        presets_bar = ttk.LabelFrame(left_col, text="Quick Presets & Actions", padding="8")
        presets_bar.pack(fill=tk.X, pady=(8, 0))

        ttk.Button(presets_bar, text="Clear All", command=self.clear_features).pack(side=tk.LEFT, padx=3)
        ttk.Button(presets_bar, text="+ 4 Neighbors", command=self.select_cardinal).pack(side=tk.LEFT, padx=3)
        ttk.Button(presets_bar, text="+ 8 Neighbors (3x3)", command=self.select_3x3).pack(side=tk.LEFT, padx=3)
        ttk.Button(presets_bar, text="+ 5x5 Ring", command=self.select_5x5).pack(side=tk.LEFT, padx=3)
        ttk.Button(presets_bar, text="+ Select All (48)", command=self.select_all_inner).pack(side=tk.LEFT, padx=3)

        self.lbl_selected_counter = ttk.Label(presets_bar, text="Selected: 0 / 48", font=("Helvetica", 10, "bold"))
        self.lbl_selected_counter.pack(side=tk.RIGHT, padx=6)

        # Right Column: Theoretical Maximum Stats & Features List
        right_col = ttk.Frame(content_frame, width=360)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_col.pack_propagate(False)

        # Information Theory Results Box
        stats_frame = ttk.LabelFrame(right_col, text="Theoretical Maximum Statistics", padding="12")
        stats_frame.pack(fill=tk.X, pady=(0, 8))

        # Prior Entropy
        row_prior = ttk.Frame(stats_frame)
        row_prior.pack(fill=tk.X, pady=2)
        ttk.Label(row_prior, text="Prior Entropy H(Mine):", font=("Helvetica", 10)).pack(side=tk.LEFT)
        self.lbl_prior = ttk.Label(row_prior, text="- bits", font=("Helvetica", 10, "bold"))
        self.lbl_prior.pack(side=tk.RIGHT)

        # Conditional Entropy
        row_cond = ttk.Frame(stats_frame)
        row_cond.pack(fill=tk.X, pady=2)
        ttk.Label(row_cond, text="Conditional H(Mine | Features):", font=("Helvetica", 10)).pack(side=tk.LEFT)
        self.lbl_cond = ttk.Label(row_cond, text="- bits", font=("Helvetica", 10, "bold"), foreground="#FF7043")
        self.lbl_cond.pack(side=tk.RIGHT)

        # Joint Mutual Information
        row_mi = ttk.Frame(stats_frame)
        row_mi.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(row_mi, text="Joint MI I(Features; Mine):", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.lbl_mi = ttk.Label(row_mi, text="- bits", font=("Helvetica", 11, "bold"), foreground="#2ECC71")
        self.lbl_mi.pack(side=tk.RIGHT)

        # Theoretical Min Error Rate (Fano)
        row_fano = ttk.Frame(stats_frame)
        row_fano.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(row_fano, text="Theoretical Min Error Rate (Pe):", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.lbl_fano = ttk.Label(row_fano, text="- %", font=("Helvetica", 10, "bold"), foreground="#E040FB")
        self.lbl_fano.pack(side=tk.RIGHT)

        # Information Gain Bar
        ttk.Label(stats_frame, text="Uncertainty Resolved (% of H(Mine)):", font=("Helvetica", 9)).pack(anchor=tk.W, pady=(6, 1))
        self.progress_bar = ttk.Progressbar(stats_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=2)
        self.lbl_pct = ttk.Label(stats_frame, text="0.0%", font=("Helvetica", 10, "bold"))
        self.lbl_pct.pack(anchor=tk.E)


        # Selected Features Table
        feat_frame = ttk.LabelFrame(right_col, text="Selected Features Set", padding="8")
        feat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        cols = ("idx", "cell", "rel_offset", "single_mi")
        self.tree = ttk.Treeview(feat_frame, columns=cols, show="headings", height=8)
        self.tree.heading("idx", text="#")
        self.tree.heading("cell", text="Cell")
        self.tree.heading("rel_offset", text="Offset")
        self.tree.heading("single_mi", text="Indiv. MI")

        self.tree.column("idx", width=34, anchor=tk.CENTER)
        self.tree.column("cell", width=64, anchor=tk.CENTER)
        self.tree.column("rel_offset", width=68, anchor=tk.CENTER)
        self.tree.column("single_mi", width=80, anchor=tk.E)

        tree_scroll = ttk.Scrollbar(feat_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Cell Inspector (Hover)
        inspector_frame = ttk.LabelFrame(right_col, text="Hover Inspector", padding="8")
        inspector_frame.pack(fill=tk.X)

        self.lbl_hover_pos = ttk.Label(inspector_frame, text="Position: -", font=("Helvetica", 9))
        self.lbl_hover_pos.pack(anchor=tk.W)

        self.lbl_hover_status = ttk.Label(inspector_frame, text="Status: -", font=("Helvetica", 9, "bold"))
        self.lbl_hover_status.pack(anchor=tk.W)

        self.lbl_hover_mi = ttk.Label(inspector_frame, text="Individual MI: -", font=("Helvetica", 9))
        self.lbl_hover_mi.pack(anchor=tk.W)

    def open_dataset_dialog(self):
        filename = filedialog.askopenfilename(
            title="Open Dataset File",
            filetypes=[("NumPy Compressed Archive", "*.npz"), ("All Files", "*.*")]
        )
        if filename:
            self.load_data(filename, self.mi_path)

    def open_mi_dialog(self):
        filename = filedialog.askopenfilename(
            title="Open Mutual Information File",
            filetypes=[("NumPy Compressed Archive", "*.npz"), ("All Files", "*.*")]
        )
        if filename:
            self.load_data(self.dataset_path, filename)

    def load_data(self, dataset_filepath, mi_filepath):
        # 1. Load Dataset
        if not os.path.exists(dataset_filepath):
            messagebox.showwarning("Dataset Not Found", f"Could not find dataset file:\n{dataset_filepath}")
            return

        try:
            with np.load(dataset_filepath) as data:
                raw_win = data['windows']
                raw_mine = data['is_mine']

                num_samples = len(raw_mine)
                # Subsample up to 50k rows for instantaneous interactive calculations
                if num_samples > 50000:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(num_samples, size=50000, replace=False)
                    self.windows = raw_win[idx]
                    self.is_mine = raw_mine[idx]
                else:
                    self.windows = raw_win
                    self.is_mine = raw_mine

            self.dataset_path = dataset_filepath

            # Calculate base Shannon prior entropy H(Mine)
            prob_mine = float(np.mean(self.is_mine))
            if 0.0 < prob_mine < 1.0:
                self.prior_entropy = binary_entropy(prob_mine)
            else:
                self.prior_entropy = 0.0

            self.lbl_prior.config(text=f"{self.prior_entropy:.4f} bits")
        except Exception as e:
            messagebox.showerror("Error Loading Dataset", f"Failed to load dataset:\n{str(e)}")
            return

        # 2. Load MI Matrix for individual cell coloring
        if os.path.exists(mi_filepath):
            try:
                with np.load(mi_filepath) as data:
                    key = 'mi_matrix' if 'mi_matrix' in data else list(data.keys())[0]
                    self.mi_matrix = np.asarray(data[key], dtype=float)
                self.mi_path = mi_filepath
                self.min_val = float(np.min(self.mi_matrix))
                self.max_val = float(np.max(self.mi_matrix))
            except Exception as e:
                print(f"Warning: Could not load MI matrix: {e}")
                self.mi_matrix = np.zeros(self.windows.shape[1:])
                self.min_val, self.max_val = 0.0, 1.0
        else:
            self.mi_matrix = np.zeros(self.windows.shape[1:])
            self.min_val, self.max_val = 0.0, 1.0

        self.lbl_files.config(
            text=f"Dataset: {os.path.basename(self.dataset_path)} | MI: {os.path.basename(self.mi_path)}"
        )

        self.recalculate()

    def toggle_cell(self, r: int, c: int):
        rows, cols = self.windows.shape[1], self.windows.shape[2]
        center_r, center_c = rows // 2, cols // 2

        # Ignore outer border and center cell
        if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
            return
        if r == center_r and c == center_c:
            return

        cell = (r, c)
        if cell in self.selected_features:
            self.selected_features.remove(cell)
        else:
            self.selected_features.append(cell)

        self.recalculate()

    def clear_features(self):
        self.selected_features.clear()
        self.recalculate()

    def select_cardinal(self):
        self.selected_features = [(3, 4), (5, 4), (4, 3), (4, 5)]
        self.recalculate()

    def select_3x3(self):
        self.selected_features = [
            (r, c)
            for r in range(3, 6)
            for c in range(3, 6)
            if not (r == 4 and c == 4)
        ]
        self.recalculate()

    def select_5x5(self):
        self.selected_features = [
            (r, c)
            for r in range(2, 7)
            for c in range(2, 7)
            if not (r == 4 and c == 4)
        ]
        self.recalculate()

    def select_all_inner(self):
        rows, cols = self.windows.shape[1], self.windows.shape[2]
        self.selected_features = [
            (r, c)
            for r in range(1, rows - 1)
            for c in range(1, cols - 1)
            if not (r == rows // 2 and c == cols // 2)
        ]
        self.recalculate()

    def recalculate(self):
        if self.windows is None or self.is_mine is None:
            return

        num_selected = len(self.selected_features)
        self.lbl_selected_counter.config(text=f"Selected: {num_selected} / 48")

        # 1. Calculate Theoretical Maximum MI, Conditional Entropy, and Fano Min Error Rate
        if num_selected == 0:
            h_cond = self.prior_entropy
            mi = 0.0
            min_error_rate = solve_fano(self.prior_entropy)
        else:
            mi, h_cond, min_error_rate = get_theoretical_maximum(
                self.windows,
                self.is_mine,
                self.selected_features
            )

        # 2. Update Stats Labels
        self.lbl_cond.config(text=f"{h_cond:.4f} bits")
        self.lbl_mi.config(text=f"{mi:.4f} bits")
        self.lbl_fano.config(text=f"{min_error_rate * 100:.2f}%")

        # Percentage of uncertainty resolved
        pct = (mi / self.prior_entropy * 100.0) if self.prior_entropy > 0 else 0.0
        self.progress_bar['value'] = min(100.0, max(0.0, pct))
        self.lbl_pct.config(text=f"{pct:.2f}%")


        # 3. Update Selected Features Tree
        self.tree.delete(*self.tree.get_children())
        rows, cols = self.windows.shape[1], self.windows.shape[2]
        cr, cc = rows // 2, cols // 2

        for idx, (r, c) in enumerate(self.selected_features, 1):
            rel_offset = f"({r - cr:+d}, {c - cc:+d})"
            indiv_mi = float(self.mi_matrix[r, c]) if self.mi_matrix is not None else 0.0
            self.tree.insert("", "end", values=(f"#{idx}", f"({r},{c})", rel_offset, f"{indiv_mi:.4f}"))

        self._draw_heatmap()

    def _draw_heatmap(self):
        self.canvas.delete("all")
        if self.windows is None:
            return

        rows, cols = self.windows.shape[1], self.windows.shape[2]
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

        selected_dict = {cell: idx + 1 for idx, cell in enumerate(self.selected_features)}

        font_size = max(8, int(cell_size * 0.23))
        font_style = ("Helvetica", font_size, "bold")
        badge_font = ("Helvetica", max(7, int(cell_size * 0.19)), "bold")

        for r in range(rows):
            for c in range(cols):
                val = float(self.mi_matrix[r, c]) if self.mi_matrix is not None else 0.0
                x1 = start_x + c * cell_size
                y1 = start_y + r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                is_center = (r == center_r and c == center_c)
                is_border = (r == 0 or r == rows - 1 or c == 0 or c == cols - 1)
                is_selected = (r, c) in selected_dict

                t = (val - self.min_val) / max(1e-9, (self.max_val - self.min_val))

                if is_center:
                    bg_color = "#231F0A"
                    fg_color = "#FFD700"
                    border_color = "#FFD700"
                elif is_border:
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

                # Format text
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

                # Draw selection badge
                if is_selected:
                    rank = selected_dict[(r, c)]
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

    def _on_canvas_click(self, event):
        if not hasattr(self, 'grid_meta') or self.windows is None:
            return

        sx = self.grid_meta["start_x"]
        sy = self.grid_meta["start_y"]
        cs = self.grid_meta["cell_size"]
        rows = self.grid_meta["rows"]
        cols = self.grid_meta["cols"]

        col = (event.x - sx) // cs
        row = (event.y - sy) // cs

        if 0 <= row < rows and 0 <= col < cols:
            self.toggle_cell(row, col)

    def _on_canvas_motion(self, event):
        if not hasattr(self, 'grid_meta') or self.windows is None:
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
                val = float(self.mi_matrix[row, col]) if self.mi_matrix is not None else 0.0
                cr, cc = rows // 2, cols // 2
                rel_r, rel_c = row - cr, col - cc

                is_center = (row == cr and col == cc)
                is_border = (row == 0 or row == rows - 1 or col == 0 or col == cols - 1)
                is_selected = (row, col) in self.selected_features

                self.lbl_hover_pos.config(text=f"Position: ({row}, {col}) [Offset: {rel_r:+d}, {rel_c:+d}]")
                self.lbl_hover_mi.config(text=f"Individual MI: {val:.5f}")

                if is_center:
                    self.lbl_hover_status.config(text="Status: Target Cell (Center - Excluded)", foreground="#FFD700")
                elif is_border:
                    self.lbl_hover_status.config(text="Status: Excluded (Outer Border)", foreground="#8B949E")
                elif is_selected:
                    rank = self.selected_features.index((row, col)) + 1
                    self.lbl_hover_status.config(text=f"Status: Selected Feature (#{rank}) - Click to remove", foreground="#2ECC71")
                else:
                    self.lbl_hover_status.config(text="Status: Available Feature - Click to select", foreground="#4FC3F7")

                self._draw_heatmap()
        else:
            self._on_canvas_leave(event)

    def _on_canvas_leave(self, event):
        if self.hovered_cell is not None:
            self.hovered_cell = None
            self.lbl_hover_pos.config(text="Position: -")
            self.lbl_hover_status.config(text="Status: -", foreground="#4FC3F7")
            self.lbl_hover_mi.config(text="Individual MI: -")
            self._draw_heatmap()


def main():
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else "cos700_dataset.npz"
    mi_file = sys.argv[2] if len(sys.argv) > 2 else "mutual_information.npz"

    root = tk.Tk()
    app = TheoreticalMaximumVisualizerApp(root, dataset_path=dataset_file, mi_path=mi_file)

    def on_resize(event):
        if app.windows is not None:
            app._draw_heatmap()

    app.canvas.bind("<Configure>", on_resize)
    root.mainloop()


if __name__ == "__main__":
    main()
