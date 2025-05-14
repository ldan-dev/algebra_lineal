# app_entry.py
# Script principal para PyInstaller
import sys
import os

# Asegurar que la ruta actual esté en sys.path
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar el archivo principal UI.py
from UI import LinearAlgebraUI
import tkinter as tk

def main():
    root = tk.Tk()
    app = LinearAlgebraUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()