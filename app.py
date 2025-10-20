from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from src.sam_crack_mask_extractor import SAMCrackMaskExtractor


# Backend processing function


def process_directory(path):
    path = Path(path)
    sam_extractor = SAMCrackMaskExtractor(
        input_dir=path.as_posix(), save_dir=(path.parent / (path.name + "_processed")).as_posix()
    )
    sam_extractor.extract()

# Function triggered by button


def submit_path():
    directory = entry.get()
    result = process_directory(directory)
    messagebox.showinfo("Result", result)


def browse_directory():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        entry.delete(0, tk.END)
        entry.insert(0, folder_selected)


# Tkinter UI setup
root = tk.Tk()
root.title("Directory Processor")

tk.Label(root, text="Enter or select a directory path:").pack(pady=10)
entry = tk.Entry(root, width=50)
entry.pack(pady=5)

browse_btn = tk.Button(root, text="Browse", command=browse_directory)
browse_btn.pack(pady=5)

submit_btn = tk.Button(root, text="Process", command=submit_path)
submit_btn.pack(pady=10)

root.mainloop()
