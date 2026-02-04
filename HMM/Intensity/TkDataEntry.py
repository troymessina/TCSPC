# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 16:13:25 2026

@author: messinat
"""

import tkinter as tk
from tkinter import simpledialog

root = tk.Tk()
root.withdraw() # Hide the main window

# Ask for an integer
age = simpledialog.askinteger("Age Input", "How old are you?", parent=root, minvalue=0, maxvalue=120)
if age is not None:
    print(f"Age entered: {age}")

# Ask for a float
salary = simpledialog.askfloat("Salary Input", "What is your salary?", parent=root, minvalue=0.0)
if salary is not None:
    print(f"Salary entered: {salary}")

root.destroy()