import tkinter as tk
from tkinter import messagebox
import numpy as np
from AlgebraLineal import AlgebraLineal

class AlgebraLinealApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Algebra Lineal App")
        self.algebra = AlgebraLineal()

        self.create_widgets()

    def create_widgets(self):
        # Vector addition
        self.vector_frame = tk.Frame(self.root)
        self.vector_frame.pack(pady=10)

        tk.Label(self.vector_frame, text="Vector 1 (comma-separated):").grid(row=0, column=0)
        self.vector1_entry = tk.Entry(self.vector_frame)
        self.vector1_entry.grid(row=0, column=1)

        tk.Label(self.vector_frame, text="Vector 2 (comma-separated):").grid(row=1, column=0)
        self.vector2_entry = tk.Entry(self.vector_frame)
        self.vector2_entry.grid(row=1, column=1)

        self.add_button = tk.Button(self.vector_frame, text="Add Vectors", command=self.add_vectors)
        self.add_button.grid(row=2, columnspan=2)

        # Matrix inversion
        self.matrix_frame = tk.Frame(self.root)
        self.matrix_frame.pack(pady=10)

        tk.Label(self.matrix_frame, text="Matrix (comma-separated rows):").grid(row=0, column=0)
        self.matrix_entry = tk.Entry(self.matrix_frame)
        self.matrix_entry.grid(row=0, column=1)

        self.invert_button = tk.Button(self.matrix_frame, text="Invert Matrix", command=self.invert_matrix)
        self.invert_button.grid(row=1, columnspan=2)

    def add_vectors(self):
        try:
            v1 = np.array([float(x) for x in self.vector1_entry.get().split(',')])
            v2 = np.array([float(x) for x in self.vector2_entry.get().split(',')])
            result = self.algebra.suma_vectores(v1, v2)
            messagebox.showinfo("Result", f"Result: {result}")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def invert_matrix(self):
        try:
            rows = self.matrix_entry.get().split(';')
            matrix = np.array([[float(num) for num in row.split(',')] for row in rows])
            result = self.algebra.inversa_matriz(matrix)
            if result is not None:
                messagebox.showinfo("Result", f"Inverted Matrix:\n{result}")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AlgebraLinealApp(root)
    root.mainloop()