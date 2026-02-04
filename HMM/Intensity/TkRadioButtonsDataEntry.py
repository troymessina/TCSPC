# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 14:08:15 2026

@author: messinat
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import os


class FileSelectionDialog:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("File Selection Dialog")
        self.root.geometry("500x450")
        self.root.resizable(False, False)
        
        # Variables to store user selections
        self.selected_file = None
        self.file_format = tk.StringVar(value=".txt")
        self.binning = tk.StringVar()
        self.confirmed = False
        
        self.create_widgets()
        
    def create_widgets(self):
        # File selection section
        file_frame = tk.LabelFrame(self.root, text="File Selection", padx=10, pady=10)
        file_frame.pack(padx=20, pady=10, fill="x")
        
        self.file_label = tk.Label(file_frame, text="No file selected", 
                                   fg="gray", wraplength=400, justify="left")
        self.file_label.pack(pady=5)
        
        select_button = tk.Button(file_frame, text="Browse...", 
                                 command=self.select_file, width=15)
        select_button.pack(pady=5)
        
        # File format section
        format_frame = tk.LabelFrame(self.root, text="File Format", padx=10, pady=10)
        format_frame.pack(padx=20, pady=10, fill="x")
        
        formats = [
            (".txt", "Text File (.txt)"),
            (".csv", "CSV File (.csv)"),
            (".xlsx", "Excel File (.xlsx)")
        ]
        
        for value, text in formats:
            rb = tk.Radiobutton(format_frame, text=text, variable=self.file_format, 
                               value=value)
            rb.pack(anchor="w", pady=2)
        
        # Float input section
        float_frame = tk.LabelFrame(self.root, text="Enter the time step in seconds", 
                                   padx=10, pady=10)
        float_frame.pack(padx=20, pady=10, fill="x")
        
        float_entry = tk.Entry(float_frame, textvariable=self.binning, width=30)
        float_entry.pack(pady=5)
        
        tk.Label(float_frame, text="Enter a time step or frame rate in seconds (e.g., 0.01)", 
                fg="gray", font=("Arial", 8)).pack()
        
        # Confirm button
        confirm_button = tk.Button(self.root, text="Confirm", command=self.confirm, 
                                   width=15, bg="#4CAF50", fg="white", 
                                   font=("Arial", 10, "bold"))
        confirm_button.pack(pady=10)
        
    def select_file(self):
        filename = filedialog.askopenfilename(
            title="Select a file",
            filetypes=[
                ("All files", "*.*"),
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx")
            ]
        )
        
        if filename:
            self.selected_file = filename
            # Display the filename (truncate if too long)
            display_name = os.path.basename(filename)
            if len(filename) > 50:
                display_name = "..." + filename[-47:]
            self.file_label.config(text=f"Selected: {display_name}", fg="black")
    
    def confirm(self):
        # Validate inputs
        if self.selected_file is None:
            messagebox.showerror("Error", "Please select a file!")
            return
        
        if not self.binning.get():
            messagebox.showerror("Error", "Please enter a time step increment (frame rate)!")
            return
        
        try:
            float(self.binning.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid fame rate!")
            return
        
        self.confirmed = True
        self.root.quit()
    
    def run(self):
        self.root.mainloop()
        self.root.destroy()
        
        if self.confirmed:
            return {
                'file_path': self.selected_file,
                'file_format': self.file_format.get(),
                'binning': float(self.binning.get())
            }
        else:
            return None


def main():
    # Create and run the dialog
    dialog = FileSelectionDialog()
    result = dialog.run()
    
    # Display the results
    if result:
        print("\n" + "="*50)
        print("USER SELECTIONS:")
        print("="*50)
        print(f"Selected File: {result['file_path']}")
        print(f"File Format: {result['file_format']}")
        print(f"Frame rate: {result['binning']} seconds")
        print("="*50)
        
        # Example: Read the file based on format
        try:
            with open(result['file_path'], 'rb') as f:
                file_content = f.read()
                print(f"\nFile loaded successfully!")
                print(f"File size: {len(file_content)} bytes")
        except Exception as e:
            print(f"\nError reading file: {e}")
    else:
        print("\nDialog was closed without confirmation.")


if __name__ == "__main__":
    main()