"""
Simple UI test script
"""

import tkinter as tk
from ui_fixed import AlgebraLinealGUI

def main():
    """Run the UI test"""
    root = tk.Tk()
    app = AlgebraLinealGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
