# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 16:11:18 2026

@author: messinat
"""

import tkinter as tk
from tkinter import filedialog
import os

class FileDialogApp:
    def __init__(self, root):
        self.root = root
        self.root.title("File Dialog with Radio Buttons")

        # 1. Variable to store the radio button choice
        self.file_type_choice = tk.StringVar(value="txt") # Set a default value

        # 2. Create Radio Buttons
        tk.Label(root, text="Select File Type:").pack(pady=10)

        tk.Radiobutton(root, text="Text Files (*.txt)", variable=self.file_type_choice, value="txt").pack(anchor=tk.W)
        tk.Radiobutton(root, text="Python Files (*.py)", variable=self.file_type_choice, value="py").pack(anchor=tk.W)
        tk.Radiobutton(root, text="All Files (*.*)", variable=self.file_type_choice, value="all").pack(anchor=tk.W)
        
        # 3. Button to open the file dialog
        tk.Button(root, text="Browse for File", command=self.open_file_dialog).pack(pady=20)

        # 4. Label to display the selected file path
        self.file_path_label = tk.Label(root, text="Selected File: None", wraplength=400)
        self.file_path_label.pack(pady=10)

    def open_file_dialog(self):
        # Get the current value of the selected radio button
        selected_type = self.file_type_choice.get()

        # Define file filters based on the selection
        if selected_type == "txt":
            filetypes = (("Text files", "*.txt"), ("All files", "*.*"))
        elif selected_type == "py":
            filetypes = (("Python files", "*.py"), ("All files", "*.*"))
        else:
            filetypes = (("All files", "*.*"),)

        # Open the file dialog with the determined filters
        file_path = filedialog.askopenfilename(
            title="Select a file",
            initialdir=os.getcwd(), # Start in the current working directory
            filetypes=filetypes
        )

        # Update the label with the selected file path
        if file_path:
            self.file_path_label.config(text=f"Selected File: {file_path}")
        else:
            self.file_path_label.config(text="Selected File: None")

if __name__ == "__main__":
    root = tk.Tk()
    app = FileDialogApp(root)
    root.mainloop()
