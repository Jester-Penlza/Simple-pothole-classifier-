"""Small desktop upload interface for the scikit-learn pothole classifier."""

from __future__ import annotations

import tkinter as tk
import tempfile
from pathlib import Path
from tkinter import filedialog, messagebox
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, ImageTk
from tkinterdnd2 import DND_FILES, DND_TEXT, TkinterDnD

from predict import PROJECT_DIR, load_bundle, predict_image


class PotholeClassifierApp:
    MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024

    def __init__(self, root: TkinterDnD.Tk) -> None:
        self.root = root
        self.root.title("Simple Pothole Image Classifier")
        self.root.geometry("760x620")
        self.root.configure(bg="#f4f6fb")
        self.preview_photo: ImageTk.PhotoImage | None = None

        try:
            self.bundle = load_bundle(PROJECT_DIR / "model.joblib")
        except FileNotFoundError as error:
            messagebox.showerror("Model missing", str(error))
            raise SystemExit(1) from error

        tk.Label(
            root,
            text="Simple Pothole Image Classifier",
            font=("Segoe UI", 22, "bold"),
            fg="#233b84",
            bg="#f4f6fb",
        ).pack(pady=(24, 4))
        tk.Label(
            root,
            text="Drop or choose a road image. The model predicts Pothole or No Pothole.",
            font=("Segoe UI", 11),
            fg="#333333",
            bg="#f4f6fb",
        ).pack(pady=(0, 18))

        self.preview = tk.Label(
            root,
            text="Drag and drop a road image here\n\nor use the button below",
            font=("Segoe UI", 12),
            width=70,
            height=22,
            bg="white",
            fg="#555555",
            relief="solid",
            bd=1,
        )
        self.preview.pack(padx=24)
        self.preview.drop_target_register(DND_FILES, DND_TEXT)
        self.preview.dnd_bind("<<DropEnter>>", self.on_drop_enter)
        self.preview.dnd_bind("<<DropLeave>>", self.on_drop_leave)
        self.preview.dnd_bind("<<Drop>>", self.on_drop)

        self.result = tk.Label(
            root,
            text="Prediction will appear here",
            font=("Segoe UI", 18, "bold"),
            fg="#333333",
            bg="#f4f6fb",
        )
        self.result.pack(pady=18)

        tk.Button(
            root,
            text="Choose Image and Predict",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg="#233b84",
            activebackground="#18275a",
            activeforeground="white",
            padx=24,
            pady=10,
            command=self.choose_image,
        ).pack()

    def choose_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose a road image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return

        self.show_prediction(Path(selected))

    def on_drop_enter(self, event: tk.Event) -> str:
        self.preview.configure(bg="#e6edff", relief="solid", bd=3)
        return event.action

    def on_drop_leave(self, event: tk.Event) -> str:
        self.preview.configure(bg="white", relief="solid", bd=1)
        return event.action

    def on_drop(self, event: tk.Event) -> str:
        self.preview.configure(bg="white", relief="solid", bd=1)
        dropped_items = self.root.tk.splitlist(event.data)

        for item in dropped_items:
            source = item.strip()
            if source.lower().startswith("file://"):
                parsed = urlparse(source)
                source = unquote(parsed.path)
                if source.startswith("/") and len(source) > 2 and source[2] == ":":
                    source = source[1:]

            if source.lower().startswith(("http://", "https://")):
                self.show_url_prediction(source)
                return event.action

            image_path = Path(source)
            if image_path.is_file():
                self.show_prediction(image_path)
                return event.action

        messagebox.showerror(
            "Cannot use dropped item",
            "Drop an image file from File Explorer or an image URL from a browser.",
        )
        return event.action

    def show_url_prediction(self, url: str) -> None:
        temporary_path: Path | None = None
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=15) as response:
                image_bytes = response.read(self.MAX_DOWNLOAD_BYTES + 1)
            if len(image_bytes) > self.MAX_DOWNLOAD_BYTES:
                raise ValueError("The dropped image is larger than 20 MB.")

            suffix = Path(urlparse(url).path).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                suffix = ".img"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
                temporary.write(image_bytes)
                temporary_path = Path(temporary.name)
            self.show_prediction(temporary_path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot download image", str(error))
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def show_prediction(self, image_path: Path) -> None:
        try:
            label, confidence = predict_image(self.bundle, image_path)
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((620, 350), Image.Resampling.LANCZOS)
                self.preview_photo = ImageTk.PhotoImage(image)
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot read image", str(error))
            return

        self.preview.configure(image=self.preview_photo, text="", width=620, height=350)
        color = "#b42318" if label == "Pothole" else "#177245"
        self.result.configure(
            text=f"Prediction: {label}  |  Confidence: {confidence:.1%}", fg=color
        )


def main() -> None:
    root = TkinterDnD.Tk()
    PotholeClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

