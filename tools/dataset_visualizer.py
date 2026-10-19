#!/usr/bin/env python3
"""
Dataset Visualizer for Minesweeper CSP dataset (.npz).
Allows browsing samples with a Tkinter GUI, visualizing sliding windows,
and viewing CSP probabilities, coordinates, seed info, and mine ground truth.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np


def cell_color(value: int) -> str:
    """Returns the text hex color for a given Minesweeper cell value."""
    match int(value):
        case 1:
            return "#0000FF"  # Blue
        case 2:
            return "#008000"  # Green
        case 3:
            return "#FF0000"  # Red
        case 4:
            return "#000080"  # Navy
        case 5:
            return "#800000"  # Maroon
        case 6:
            return "#008080"  # Teal
        case 7:
            return "#000000"  # Black
        case 8:
            return "#808080"  # Gray
        case 12:
            return "#D32F2F"  # Mine Red
        case _:
            return "#202020"


def get_risk_color_hex(value: float) -> str:
    """Computes RGB hex color for CSP risk (green -> yellow -> red)."""
    value = max(0.0, min(1.0, float(value)))
    if value <= 0.5:
        t = value / 0.5
        r = int(255 * t)
        g = int(255 - (90 * t))
        b = 0
    else:
        t = (value - 0.5) / 0.5
        r = 255
        g = int(165 * (1 - t))
        b = 0
    return f"#{r:02x}{g:02x}{b:02x}"


class DatasetVisualizerApp:
    def __init__(self, root, npz_path="cos700_dataset.npz"):
        self.root = root
        self.root.title("COS700 Dataset Visualizer")
        self.root.geometry("820x620")
        self.root.minsize(760, 560)

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.npz_path = npz_path
        self.data = None
        self.current_index = 0
        self.total_samples = 0
        self.cell_size = 42

        # Build UI layout
        self._create_widgets()

        # Load dataset
        self.load_dataset(self.npz_path)

        # Keyboard bindings
        self.root.bind("<Left>", lambda e: self.prev_sample())
        self.root.bind("<Right>", lambda e: self.next_sample())
        self.root.bind("<Home>", lambda e: self.jump_to_index(0))
        self.root.bind("<End>", lambda e: self.jump_to_index(self.total_samples - 1))

    def _create_widgets(self):
        # 1. Top File & Info Header
        header_frame = ttk.Frame(self.root, padding="10 8 10 4")
        header_frame.pack(fill=tk.X)

        self.file_label = ttk.Label(
            header_frame,
            text=f"Dataset: {os.path.basename(self.npz_path)}",
            font=("Helvetica", 11, "bold")
        )
        self.file_label.pack(side=tk.LEFT)

        open_btn = ttk.Button(header_frame, text="Open .npz File...", command=self.open_file_dialog)
        open_btn.pack(side=tk.RIGHT)

        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

        # 2. Main Content Area (Grid on Left, Details on Right)
        content_frame = ttk.Frame(self.root, padding="10")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left Column: Grid Canvas
        left_frame = ttk.LabelFrame(content_frame, text="Sliding Window", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.canvas = tk.Canvas(left_frame, bg="#1E1E1E", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right Column: Metadata & Details
        right_frame = ttk.Frame(content_frame, width=320)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        right_frame.pack_propagate(False)

        # Info Box
        info_frame = ttk.LabelFrame(right_frame, text="Sample Details", padding="12")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        # Labels for details
        self.lbl_index = ttk.Label(info_frame, text="Sample: 0 / 0", font=("Helvetica", 10, "bold"))
        self.lbl_index.pack(anchor=tk.W, pady=2)

        self.lbl_seed = ttk.Label(info_frame, text="Seed: -", font=("Helvetica", 10))
        self.lbl_seed.pack(anchor=tk.W, pady=2)

        self.lbl_coord = ttk.Label(info_frame, text="Coordinates (Row, Col): -", font=("Helvetica", 10))
        self.lbl_coord.pack(anchor=tk.W, pady=2)

        # CSP Probability Display
        prob_frame = ttk.Frame(info_frame)
        prob_frame.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(prob_frame, text="CSP Probability:", font=("Helvetica", 10)).pack(side=tk.LEFT)
        self.lbl_prob_val = ttk.Label(prob_frame, text="0.0%", font=("Helvetica", 10, "bold"))
        self.lbl_prob_val.pack(side=tk.RIGHT)

        # Risk indicator bar
        self.risk_bar = tk.Canvas(info_frame, height=14, bg="#333333", highlightthickness=1, highlightbackground="#555555")
        self.risk_bar.pack(fill=tk.X, pady=(2, 6))

        # Ground Truth Mine Status
        mine_frame = ttk.Frame(info_frame)
        mine_frame.pack(fill=tk.X, pady=4)
        ttk.Label(mine_frame, text="Is Mine (Ground Truth):", font=("Helvetica", 10)).pack(side=tk.LEFT)
        self.lbl_mine_badge = tk.Label(
            mine_frame,
            text="SAFE",
            bg="#2E7D32",
            fg="white",
            font=("Helvetica", 9, "bold"),
            padx=8,
            pady=2,
            relief=tk.FLAT
        )
        self.lbl_mine_badge.pack(side=tk.RIGHT)

        # Jump / Search Frame
        search_frame = ttk.LabelFrame(right_frame, text="Search & Navigation", padding="10")
        search_frame.pack(fill=tk.X, pady=(0, 10))

        # Search by Seed
        seed_row = ttk.Frame(search_frame)
        seed_row.pack(fill=tk.X, pady=4)
        ttk.Label(seed_row, text="Seed:").pack(side=tk.LEFT)
        self.entry_seed = ttk.Entry(seed_row, width=10)
        self.entry_seed.pack(side=tk.LEFT, padx=6)
        self.entry_seed.bind("<Return>", lambda e: self.search_seed())
        btn_find_seed = ttk.Button(seed_row, text="Find", width=6, command=self.search_seed)
        btn_find_seed.pack(side=tk.LEFT)

        # Jump by Index
        index_row = ttk.Frame(search_frame)
        index_row.pack(fill=tk.X, pady=4)
        ttk.Label(index_row, text="Index:").pack(side=tk.LEFT)
        self.entry_index = ttk.Entry(index_row, width=10)
        self.entry_index.pack(side=tk.LEFT, padx=6)
        self.entry_index.bind("<Return>", lambda e: self.jump_to_entered_index())
        btn_go_index = ttk.Button(index_row, text="Go", width=6, command=self.jump_to_entered_index)
        btn_go_index.pack(side=tk.LEFT)

        # Legend / Guide
        legend_frame = ttk.LabelFrame(right_frame, text="Legend", padding="8")
        legend_frame.pack(fill=tk.BOTH, expand=True)

        legend_text = (
            "• Numbers (1-8): Revealed adjacent mines\n"
            "• Blank (0): Revealed safe cell\n"
            "• '?': Unrevealed cell\n"
            "• 'X': Out of bounds cell\n"
            "• 'M': Mine (if revealed)\n"
            "• Center Yellow Ring: Target cell"
        )
        lbl_legend = ttk.Label(legend_frame, text=legend_text, font=("Helvetica", 9), justify=tk.LEFT)
        lbl_legend.pack(anchor=tk.W)

        # 3. Bottom Controls (Navigation Bar)
        nav_frame = ttk.Frame(self.root, padding="10 6 10 10")
        nav_frame.pack(fill=tk.X)

        btn_first = ttk.Button(nav_frame, text="⏮ First", command=lambda: self.jump_to_index(0))
        btn_first.pack(side=tk.LEFT, padx=3)

        btn_prev = ttk.Button(nav_frame, text="◀ Previous", command=self.prev_sample)
        btn_prev.pack(side=tk.LEFT, padx=3)

        btn_next = ttk.Button(nav_frame, text="Next ▶", command=self.next_sample)
        btn_next.pack(side=tk.LEFT, padx=3)

        btn_last = ttk.Button(nav_frame, text="Last ⏭", command=lambda: self.jump_to_index(self.total_samples - 1))
        btn_last.pack(side=tk.LEFT, padx=3)

        self.lbl_status = ttk.Label(nav_frame, text="Use ← and → keys to browse", font=("Helvetica", 9, "italic"))
        self.lbl_status.pack(side=tk.RIGHT, padx=5)

    def open_file_dialog(self):
        filename = filedialog.askopenfilename(
            title="Open Dataset File",
            filetypes=[("NumPy Compressed Archive", "*.npz"), ("All Files", "*.*")]
        )
        if filename:
            self.load_dataset(filename)

    def load_dataset(self, filepath):
        if not os.path.exists(filepath):
            messagebox.showwarning(
                "Dataset Not Found",
                f"Could not find dataset file:\n{filepath}\n\nPlease generate it or open a valid .npz file."
            )
            return

        try:
            self.data = np.load(filepath)
            self.npz_path = filepath
            self.file_label.config(text=f"Dataset: {os.path.basename(filepath)}")
            self.total_samples = len(self.data['seeds'])
            self.current_index = 0

            # Pre-cache seeds for quick searching
            self.seeds_array = self.data['seeds']

            self.update_view()
        except Exception as e:
            messagebox.showerror("Error Loading Dataset", f"Failed to load {filepath}:\n{str(e)}")

    def jump_to_index(self, index: int):
        if self.total_samples == 0:
            return
        self.current_index = max(0, min(self.total_samples - 1, index))
        self.update_view()

    def prev_sample(self):
        if self.total_samples == 0:
            return
        if self.current_index > 0:
            self.current_index -= 1
            self.update_view()

    def next_sample(self):
        if self.total_samples == 0:
            return
        if self.current_index < self.total_samples - 1:
            self.current_index += 1
            self.update_view()

    def jump_to_entered_index(self):
        text = self.entry_index.get().strip()
        if not text:
            return
        try:
            val = int(text)
            if 0 <= val < self.total_samples:
                self.jump_to_index(val)
            else:
                messagebox.showwarning("Out of Range", f"Please enter an index between 0 and {self.total_samples - 1}.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Index must be an integer.")

    def search_seed(self):
        text = self.entry_seed.get().strip()
        if not text:
            return
        try:
            target_seed = int(text)
            matches = np.where(self.seeds_array == target_seed)[0]
            if len(matches) == 0:
                messagebox.showinfo("Seed Not Found", f"No samples found with seed {target_seed}.")
                return

            # If current sample is already a match, go to next match if available
            current_in_matches = np.where(matches > self.current_index)[0]
            if len(current_in_matches) > 0:
                next_idx = matches[current_in_matches[0]]
            else:
                next_idx = matches[0]

            self.jump_to_index(next_idx)
            self.lbl_status.config(text=f"Showing match for seed {target_seed} ({len(matches)} total samples)")
        except ValueError:
            messagebox.showerror("Invalid Input", "Seed must be an integer.")

    def update_view(self):
        if self.total_samples == 0 or self.data is None:
            return

        idx = self.current_index
        seed = int(self.data['seeds'][idx])
        window = self.data['windows'][idx]
        csp_prob = float(self.data['csp_probs'][idx])
        is_mine = bool(self.data['is_mine'][idx])
        coords = tuple(self.data['coords'][idx])

        # 1. Update side info labels
        self.lbl_index.config(text=f"Sample: {idx + 1:,} / {self.total_samples:,} (Index {idx})")
        self.lbl_seed.config(text=f"Seed: {seed}")
        self.lbl_coord.config(text=f"Coordinates (Row, Col): ({coords[0]}, {coords[1]})")

        # Probability and risk bar
        prob_pct = csp_prob * 100
        risk_hex = get_risk_color_hex(csp_prob)
        self.lbl_prob_val.config(text=f"{csp_prob:.4f} ({prob_pct:.1f}%)", foreground=risk_hex)

        self.risk_bar.delete("all")
        bar_w = self.risk_bar.winfo_width()
        if bar_w < 10:
            bar_w = 260
        fill_w = max(2, int(bar_w * csp_prob))
        self.risk_bar.create_rectangle(0, 0, fill_w, 14, fill=risk_hex, outline="")

        # Mine badge
        if is_mine:
            self.lbl_mine_badge.config(text="MINE", bg="#D32F2F", fg="white")
        else:
            self.lbl_mine_badge.config(text="SAFE", bg="#2E7D32", fg="white")

        # 2. Draw the grid
        self._draw_grid(window)

    def _draw_grid(self, window: np.ndarray):
        self.canvas.delete("all")

        rows, cols = window.shape
        center_r, center_c = rows // 2, cols // 2

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 50 or canvas_h < 50:
            canvas_w, canvas_h = 420, 420

        # Calculate fitting cell size
        max_cell_w = (canvas_w - 40) // cols
        max_cell_h = (canvas_h - 40) // rows
        cell_size = max(20, min(max_cell_w, max_cell_h, 44))

        start_x = (canvas_w - (cols * cell_size)) // 2
        start_y = (canvas_h - (rows * cell_size)) // 2

        font_size = max(10, int(cell_size * 0.48))
        font_style = ("Helvetica", font_size, "bold")

        for r in range(rows):
            for c in range(cols):
                val = window[r, c]
                x1 = start_x + c * cell_size
                y1 = start_y + r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                # Cell styling based on value
                if val == 10:  # Out of bounds
                    fill = "#2B2B2B"
                    border = "#404040"
                    text = "X"
                    text_color = "#555555"
                elif val == 9:   # Hidden / Unrevealed
                    fill = "#C0C0C0"
                    border = "#787878"
                    text = "?"
                    text_color = "#333333"
                elif val == 0:   # Revealed empty cell
                    fill = "#E0E0E0"
                    border = "#9E9E9E"
                    text = ""
                    text_color = "#000000"
                elif val == 12:  # Mine
                    fill = "#D32F2F"
                    border = "#787878"
                    text = "M"
                    text_color = "#FFFFFF"
                else:            # Number cell 1-8
                    fill = "#C0C0C0"
                    border = "#787878"
                    text = str(int(val))
                    text_color = cell_color(val)

                # Draw base cell
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=border, width=1)

                # Draw cell text
                if text:
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text=text,
                        fill=text_color,
                        font=font_style
                    )

                # Highlight center / target cell
                if r == center_r and c == center_c:
                    self.canvas.create_rectangle(
                        x1 + 2, y1 + 2, x2 - 2, y2 - 2,
                        outline="#FFB300",
                        width=3
                    )


def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "cos700_dataset.npz"
    root = tk.Tk()
    app = DatasetVisualizerApp(root, npz_path=dataset_path)

    # Re-draw grid on canvas resize
    def on_resize(event):
        if app.data is not None:
            app.update_view()

    app.canvas.bind("<Configure>", on_resize)
    root.mainloop()


if __name__ == "__main__":
    main()
