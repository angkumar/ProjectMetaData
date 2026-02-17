import tkinter as tk
from tkinter import filedialog, ttk
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageTk, ImageDraw, ImageFilter
from Cancer_Detector import CancerCNN
import numpy as np
import os
import math
import time

def load_model(model_path):
    model = CancerCNN(num_classes=2)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def predict(model, image_path):
    preprocess = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]
    return probs[1].item(), probs[0].item()   # cancer_prob, clear_prob


# ─── GUI ─────────────────────────────────────────────────────────────────────

BG        = "#0a0d14"
PANEL     = "#10151f"
BORDER    = "#1e2535"
ACCENT    = "#00d4ff"
ACCENT2   = "#0077ff"
DANGER    = "#ff3b5c"
SUCCESS   = "#00e5a0"
TEXT      = "#e8edf5"
SUBTEXT   = "#6b7a99"
FONT_MONO = "Courier New"
FONT_SANS = "Helvetica Neue"

class ScanlineCanvas(tk.Canvas):
    """Animated scanline / grid background."""
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._offset = 0
        self._draw_grid()
        self._animate()

    def _draw_grid(self):
        self.delete("grid")
        w = int(self["width"])
        h = int(self["height"])
        for x in range(0, w, 40):
            self.create_line(x, 0, x, h, fill="#111824", width=1, tags="grid")
        for y in range(0, h, 40):
            self.create_line(0, y, w, y, fill="#111824", width=1, tags="grid")

    def _animate(self):
        self._offset = (self._offset + 1) % 40
        self.delete("scan")
        w = int(self["width"])
        h = int(self["height"])
        for y in range(-40 + self._offset, h, 40):
            alpha = max(0, 30 - abs(y - h // 2) // 8)
            if alpha > 0:
                self.create_line(0, y, w, y, fill="#0d1826",
                                 width=2, tags="scan")
        self.after(60, self._animate)


class PulsingRing(tk.Canvas):
    """Animated pulsing ring shown while scanning."""
    def __init__(self, master, size=120, **kw):
        super().__init__(master, width=size, height=size,
                         bg=BG, highlightthickness=0, **kw)
        self.size = size
        self._t = 0
        self._running = False
        self._job = None

    def start(self):
        self._running = True
        self._tick()

    def stop(self):
        self._running = False
        if self._job:
            self.after_cancel(self._job)
        self.delete("all")

    def _tick(self):
        if not self._running:
            return
        self.delete("all")
        cx = cy = self.size // 2
        r_base = self.size // 2 - 10

        # outer pulse rings
        for i in range(3):
            phase = (self._t * 0.05 + i * 0.33) % 1.0
            r = r_base * (0.5 + phase * 0.5)
            alpha_hex = format(int((1 - phase) * 80), "02x")
            color = f"#00d4{alpha_hex}" if i % 2 == 0 else f"#0055{alpha_hex}"
            try:
                self.create_oval(cx - r, cy - r, cx + r, cy + r,
                                 outline=color, width=2)
            except Exception:
                pass

        # rotating arc
        start_angle = (self._t * 4) % 360
        self.create_arc(cx - r_base, cy - r_base, cx + r_base, cy + r_base,
                        start=start_angle, extent=240,
                        outline=ACCENT, width=3, style="arc")

        # center dot
        self.create_oval(cx - 6, cy - 6, cx + 6, cy + 6,
                         fill=ACCENT, outline="")

        # crosshairs
        for angle_deg in [0, 90, 180, 270]:
            angle = math.radians(angle_deg + self._t * 2)
            x2 = cx + (r_base + 8) * math.cos(angle)
            y2 = cy + (r_base + 8) * math.sin(angle)
            x1 = cx + (r_base - 4) * math.cos(angle)
            y1 = cy + (r_base - 4) * math.sin(angle)
            self.create_line(x1, y1, x2, y2, fill=ACCENT, width=2)

        self._t += 1
        self._job = self.after(30, self._tick)


class ConfidenceBar(tk.Canvas):
    def __init__(self, master, label, color, **kw):
        super().__init__(master, height=36, bg=PANEL,
                         highlightthickness=0, **kw)
        self.label = label
        self.color = color
        self._value = 0
        self._target = 0
        self._animating = False
        self.bind("<Configure>", self._redraw)

    def set_value(self, v):
        self._target = v
        if not self._animating:
            self._animating = True
            self._step()

    def _step(self):
        diff = self._target - self._value
        if abs(diff) < 0.002:
            self._value = self._target
            self._animating = False
            self._redraw()
            return
        self._value += diff * 0.12
        self._redraw()
        self.after(16, self._step)

    def _redraw(self, _=None):
        self.delete("all")
        w = self.winfo_width()
        if w < 10:
            return
        h = 36
        pad = 4
        bar_h = 10
        bar_y = h // 2 + 4
        # label
        self.create_text(0, h // 2 - 8, anchor="w",
                         text=self.label, fill=SUBTEXT,
                         font=(FONT_MONO, 9))
        # track
        self.create_rectangle(0, bar_y, w, bar_y + bar_h,
                               fill="#1a2030", outline="")
        # fill
        fill_w = int(w * self._value)
        if fill_w > 0:
            # gradient effect via multiple rects
            for i in range(fill_w):
                ratio = i / max(fill_w, 1)
                r1 = int(int(self.color[1:3], 16) * (0.6 + 0.4 * ratio))
                g1 = int(int(self.color[3:5], 16) * (0.6 + 0.4 * ratio))
                b1 = int(int(self.color[5:7], 16) * (0.6 + 0.4 * ratio))
                c = f"#{min(r1,255):02x}{min(g1,255):02x}{min(b1,255):02x}"
                self.create_rectangle(i, bar_y, i + 1, bar_y + bar_h,
                                      fill=c, outline="")
        # percentage
        self.create_text(w, h // 2 - 8, anchor="e",
                         text=f"{self._value * 100:.1f}%",
                         fill=TEXT, font=(FONT_MONO, 9, "bold"))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cancer Detector  —   Kumar Industries")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.model_path = "Cancter_Detector.pt"
        self.model = None
        self.image_path = None
        self._photo = None

        self._build_ui()
        self._load_model_async()

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        W, H = 860, 640

        # animated background
        self.bg_canvas = ScanlineCanvas(
            self, width=W, height=H, bg=BG, highlightthickness=0)
        self.bg_canvas.place(x=0, y=0)

        # ── header ──
        hdr = tk.Frame(self, bg=BG)
        hdr.place(x=30, y=22)

        tk.Label(hdr, text="Cancer", font=(FONT_MONO, 22, "bold"),
                 fg=ACCENT, bg=BG).pack(side="left")
        tk.Label(hdr, text="Detector", font=(FONT_MONO, 22, "bold"),
                 fg=TEXT, bg=BG).pack(side="left")
        tk.Label(hdr, text="  /  Kumar Industries",
                 font=(FONT_MONO, 9), fg=SUBTEXT, bg=BG).pack(
                 side="left", pady=6)

        # status LED
        self.led_canvas = tk.Canvas(self, width=10, height=10,
                                    bg=BG, highlightthickness=0)
        self.led_canvas.place(x=W - 30, y=28)
        self._led_id = self.led_canvas.create_oval(0, 0, 10, 10, fill=SUBTEXT)
        self._led_blink_state = False
        self._blink_led()

        # divider
        tk.Canvas(self, width=W - 60, height=1, bg=BORDER,
                  highlightthickness=0).place(x=30, y=58)

        # ── left panel: upload + image preview ──
        left = tk.Frame(self, bg=PANEL, bd=0,
                        highlightbackground=BORDER, highlightthickness=1)
        left.place(x=30, y=74, width=380, height=520)

        tk.Label(left, text="SAMPLE INPUT", font=(FONT_MONO, 8),
                 fg=SUBTEXT, bg=PANEL).place(x=14, y=10)

        # image display area
        self.img_canvas = tk.Canvas(left, width=340, height=340,
                                    bg="#080c14", highlightthickness=1,
                                    highlightbackground=BORDER)
        self.img_canvas.place(x=20, y=34)
        self._draw_placeholder()

        # pulsing ring overlay (hidden until scan)
        self.ring = PulsingRing(left, size=120)
        self.ring.place(x=130, y=144)
        self.ring.place_forget()

        # upload button
        self.upload_btn = tk.Button(
            left, text="⬆  UPLOAD TISSUE IMAGE",
            font=(FONT_MONO, 10, "bold"),
            fg=BG, bg=ACCENT, activebackground="#00b8e0",
            activeforeground=BG, relief="flat", cursor="hand2",
            command=self._upload_image, pady=10)
        self.upload_btn.place(x=20, y=388, width=340, height=44)

        # scan button
        self.scan_btn = tk.Button(
            left, text="⬡  RUN ANALYSIS",
            font=(FONT_MONO, 10, "bold"),
            fg=ACCENT, bg="#0d1520", activebackground="#152030",
            activeforeground=ACCENT, relief="flat", cursor="hand2",
            bd=0, highlightthickness=1, highlightbackground=ACCENT,
            command=self._run_scan, pady=10, state="disabled")
        self.scan_btn.place(x=20, y=442, width=340, height=44)

        # file label
        self.file_label = tk.Label(left, text="No file selected",
                                   font=(FONT_MONO, 8), fg=SUBTEXT, bg=PANEL,
                                   wraplength=340)
        self.file_label.place(x=20, y=495)

        # ── right panel: results ──
        right = tk.Frame(self, bg=PANEL, bd=0,
                         highlightbackground=BORDER, highlightthickness=1)
        right.place(x=430, y=74, width=400, height=520)

        tk.Label(right, text="ANALYSIS REPORT", font=(FONT_MONO, 8),
                 fg=SUBTEXT, bg=PANEL).place(x=14, y=10)

        # verdict box
        self.verdict_frame = tk.Frame(right, bg="#080c14",
                                      highlightbackground=BORDER,
                                      highlightthickness=1)
        self.verdict_frame.place(x=50, y=34, width=360, height=130)

        self.verdict_icon = tk.Label(
            self.verdict_frame, text="◈", font=(FONT_MONO, 36),
            fg=BORDER, bg="#080c14")
        self.verdict_icon.place(x=20, y=24)

        self.verdict_label = tk.Label(
            self.verdict_frame, text="AWAITING\nSAMPLE",
            font=(FONT_MONO, 18, "bold"), fg=BORDER, bg="#080c14",
            justify="left")
        self.verdict_label.place(x=90, y=20)

        self.verdict_sub = tk.Label(
            self.verdict_frame, text="Upload an image and run analysis",
            font=(FONT_MONO, 8), fg=SUBTEXT, bg="#080c14")
        self.verdict_sub.place(x=20, y=100)

        # confidence bars
        tk.Label(right, text="CONFIDENCE SCORES",
                 font=(FONT_MONO, 8), fg=SUBTEXT, bg=PANEL).place(x=20, y=180)

        self.bar_cancer = ConfidenceBar(right, "MALIGNANT TISSUE", DANGER)
        self.bar_cancer.place(x=20, y=200, width=360, height=36)

        self.bar_clear = ConfidenceBar(right, "HEALTHY TISSUE", SUCCESS)
        self.bar_clear.place(x=20, y=246, width=360, height=36)

        # separator
        tk.Canvas(right, width=360, height=1, bg=BORDER,
                  highlightthickness=0).place(x=20, y=296)

        # metadata readout
        tk.Label(right, text="SCAN METADATA",
                 font=(FONT_MONO, 8), fg=SUBTEXT, bg=PANEL).place(x=20, y=310)

        self.meta_frame = tk.Frame(right, bg=PANEL)
        self.meta_frame.place(x=20, y=330, width=360, height=120)

        self._meta_rows = {}
        meta_fields = [
            ("MODEL",    "Cancter_Detector.pt"),
            ("STATUS",   "Initializing..."),
            ("DEVICE",   "CPU"),
            ("IMAGE",    "—"),
            ("RESULT",   "—"),
        ]
        for i, (k, v) in enumerate(meta_fields):
            tk.Label(self.meta_frame, text=k, font=(FONT_MONO, 8),
                     fg=SUBTEXT, bg=PANEL, width=10, anchor="w").grid(
                     row=i, column=0, sticky="w", pady=1)
            lbl = tk.Label(self.meta_frame, text=v, font=(FONT_MONO, 8),
                           fg=TEXT, bg=PANEL, anchor="w")
            lbl.grid(row=i, column=1, sticky="w", pady=1)
            self._meta_rows[k] = lbl

        # disclaimer
        tk.Label(right,
                 text="⚠  FOR RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS",
                 font=(FONT_MONO, 7), fg="#3a4560", bg=PANEL,
                 wraplength=360, justify="center").place(
                 x=20, y=480, width=360)

        # model status bar at bottom
        self.status_bar = tk.Label(
            self, text="● Initializing model...",
            font=(FONT_MONO, 8), fg=SUBTEXT, bg="#07090f", anchor="w",
            padx=12)
        self.status_bar.place(x=0, y=610, width=860, height=30)

        self.geometry("860x640")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _blink_led(self):
        self._led_blink_state = not self._led_blink_state
        color = ACCENT if self._led_blink_state else SUBTEXT
        self.led_canvas.itemconfig(self._led_id, fill=color)
        self.after(800, self._blink_led)

    def _draw_placeholder(self):
        self.img_canvas.delete("all")
        w, h = 340, 340
        # corner brackets
        size = 20
        lw = 2
        corners = [(10, 10), (w - 10, 10), (10, h - 10), (w - 10, h - 10)]
        dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        for (cx, cy), (dx, dy) in zip(corners, dirs):
            self.img_canvas.create_line(cx, cy, cx + dx * size, cy,
                                        fill=BORDER, width=lw)
            self.img_canvas.create_line(cx, cy, cx, cy + dy * size,
                                        fill=BORDER, width=lw)
        # center text
        self.img_canvas.create_text(
            w // 2, h // 2 - 12, text="◈",
            font=(FONT_MONO, 32), fill=BORDER)
        self.img_canvas.create_text(
            w // 2, h // 2 + 28, text="NO IMAGE LOADED",
            font=(FONT_MONO, 10), fill=BORDER)

    def _set_status(self, msg, color=SUBTEXT):
        self.status_bar.config(text=f"●  {msg}", fg=color)

    def _update_meta(self, key, value, color=TEXT):
        if key in self._meta_rows:
            self._meta_rows[key].config(text=value, fg=color)

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model_async(self):
        def _load():
            if not os.path.exists(self.model_path):
                self.after(0, lambda: self._set_status(
                    f"Model not found: {self.model_path}", DANGER))
                self.after(0, lambda: self._update_meta(
                    "STATUS", "NOT FOUND", DANGER))
                return
            try:
                self.model = load_model(self.model_path)
                self.after(0, lambda: self._set_status(
                    "Model loaded — ready for analysis", SUCCESS))
                self.after(0, lambda: self._update_meta(
                    "STATUS", "READY", SUCCESS))
                self.after(0, lambda: self._update_meta("DEVICE", "CPU"))
            except Exception as e:
                self.after(0, lambda: self._set_status(
                    f"Model error: {e}", DANGER))
                self.after(0, lambda: self._update_meta(
                    "STATUS", f"ERROR", DANGER))

        threading.Thread(target=_load, daemon=True).start()

    # ── Upload ───────────────────────────────────────────────────────────────

    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Tissue Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")])
        if not path:
            return

        self.image_path = path
        fname = os.path.basename(path)
        self.file_label.config(text=fname, fg=TEXT)
        self._update_meta("IMAGE", fname[:30] + ("..." if len(fname) > 30 else ""))

        # display image
        img = Image.open(path).convert("RGB")
        img_disp = img.resize((340, 340), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img_disp)
        self.img_canvas.delete("all")
        self.img_canvas.create_image(0, 0, anchor="nw", image=self._photo)

        # reset result panel
        self.verdict_icon.config(text="◈", fg=BORDER)
        self.verdict_label.config(text="READY\nTO SCAN", fg=ACCENT)
        self.verdict_sub.config(text="Press RUN ANALYSIS to begin", fg=SUBTEXT)
        self.bar_cancer.set_value(0)
        self.bar_clear.set_value(0)
        self._update_meta("RESULT", "—", TEXT)

        if self.model:
            self.scan_btn.config(state="normal")
        self._set_status("Image loaded — ready to scan", ACCENT)

    # ── Scan ─────────────────────────────────────────────────────────────────

    def _run_scan(self):
        if not self.image_path or not self.model:
            return

        self.scan_btn.config(state="disabled")
        self.upload_btn.config(state="disabled")
        self._set_status("Analyzing tissue sample...", ACCENT)

        # show ring overlay
        self.ring.place(x=130, y=144)
        self.ring.start()
        self.verdict_label.config(text="SCAN-\nNING...", fg=ACCENT)
        self.verdict_icon.config(text="⟳", fg=ACCENT)

        def _infer():
            time.sleep(0.6)   # tiny delay so animation is visible
            try:
                cancer_prob, clear_prob = predict(self.model, self.image_path)
                self.after(0, lambda: self._show_result(cancer_prob, clear_prob))
            except Exception as e:
                self.after(0, lambda: self._show_error(str(e)))

        threading.Thread(target=_infer, daemon=True).start()

    def _show_result(self, cancer_prob, clear_prob):
        self.ring.stop()
        self.ring.place_forget()

        self.bar_cancer.set_value(cancer_prob)
        self.bar_clear.set_value(clear_prob)

        is_cancer = cancer_prob >= 0.5
        confidence = max(cancer_prob, clear_prob) * 100

        if is_cancer:
            self.verdict_icon.config(text="⚠", fg=DANGER)
            self.verdict_label.config(text="MALIG-\nNANT", fg=DANGER)
            self.verdict_sub.config(
                text=f"Malignant tissue detected  ·  {confidence:.1f}% confidence",
                fg=DANGER)
            self._update_meta("RESULT", "MALIGNANT", DANGER)
            self._set_status(
                f"⚠ Malignant tissue detected ({confidence:.1f}% confidence)", DANGER)
            # flash border red
            self._flash_border(DANGER)
        else:
            self.verdict_icon.config(text="✓", fg=SUCCESS)
            self.verdict_label.config(text="CLEAR\nTISSUE", fg=SUCCESS)
            self.verdict_sub.config(
                text=f"No malignancy detected  ·  {confidence:.1f}% confidence",
                fg=SUCCESS)
            self._update_meta("RESULT", "CLEAR", SUCCESS)
            self._set_status(
                f"✓ No malignancy detected ({confidence:.1f}% confidence)", SUCCESS)
            self._flash_border(SUCCESS)

        self.scan_btn.config(state="normal")
        self.upload_btn.config(state="normal")

    def _show_error(self, msg):
        self.ring.stop()
        self.ring.place_forget()
        self.verdict_label.config(text="ERROR", fg=DANGER)
        self.verdict_sub.config(text=msg[:60], fg=DANGER)
        self._set_status(f"Error: {msg}", DANGER)
        self.scan_btn.config(state="normal")
        self.upload_btn.config(state="normal")

    def _flash_border(self, color, count=6):
        """Flash the image canvas border."""
        if count <= 0:
            self.img_canvas.config(highlightbackground=BORDER)
            return
        c = color if count % 2 == 0 else BORDER
        self.img_canvas.config(highlightbackground=c)
        self.after(120, lambda: self._flash_border(color, count - 1))


if __name__ == "__main__":
    app = App()
    app.mainloop()