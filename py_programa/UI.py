import tkinter as tk
from tkinter import ttk, messagebox
from maths import MathOperations  # Importamos la lógica de conexión

class AlgebraLinealUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Algebra Lineal Tool")
        self.root.geometry("800x600")
        self.math_ops = MathOperations()  # Instancia de la lógica

        # Estilo
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", padding=10, font=("Helvetica", 12))
        self.style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 12))

        # Crear pestañas
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Pestaña de operaciones básicas
        self.tab_basicas = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_basicas, text="Operaciones Básicas")
        self._crear_tab_basicas()

        # Pestaña de matrices
        self.tab_matrices = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_matrices, text="Matrices")
        self._crear_tab_matrices()

        # Pestaña de gráficos
        self.tab_graficos = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_graficos, text="Gráficos")
        self._crear_tab_graficos()

    def _crear_tab_basicas(self):
        """Interfaz para operaciones básicas como suma, resta, producto punto, etc."""
        frame = ttk.Frame(self.tab_basicas)
        frame.pack(padx=20, pady=20)

        # Entrada para vectores
        ttk.Label(frame, text="Vector 1 (ej: 1,2,3):").grid(row=0, column=0, padx=5, pady=5)
        self.vector1_entry = ttk.Entry(frame, width=30)
        self.vector1_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Vector 2 (ej: 4,5,6):").grid(row=1, column=0, padx=5, pady=5)
        self.vector2_entry = ttk.Entry(frame, width=30)
        self.vector2_entry.grid(row=1, column=1, padx=5, pady=5)

        # Botones de operaciones
        ttk.Button(frame, text="Sumar", command=self.sumar_vectores).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(frame, text="Producto Punto", command=self.producto_punto).grid(row=2, column=1, padx=5, pady=5)

        # Resultado
        self.resultado_basicas = ttk.Label(frame, text="Resultado: ")
        self.resultado_basicas.grid(row=3, column=0, columnspan=2, pady=10)

    def _crear_tab_matrices(self):
        """Interfaz para operaciones con matrices."""
        frame = ttk.Frame(self.tab_matrices)
        frame.pack(padx=20, pady=20)

        ttk.Label(frame, text="Matriz 1 (ej: 1,2;3,4):").grid(row=0, column=0, padx=5, pady=5)
        self.matriz1_entry = ttk.Entry(frame, width=30)
        self.matriz1_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Matriz 2 (ej: 5,6;7,8):").grid(row=1, column=0, padx=5, pady=5)
        self.matriz2_entry = ttk.Entry(frame, width=30)
        self.matriz2_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(frame, text="Multiplicar Matrices", command=self.multiplicar_matrices).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(frame, text="Calcular Determinante", command=self.calcular_determinante).grid(row=2, column=1, padx=5, pady=5)

        self.resultado_matrices = ttk.Label(frame, text="Resultado: ")
        self.resultado_matrices.grid(row=3, column=0, columnspan=2, pady=10)

    def _crear_tab_graficos(self):
        """Interfaz para graficar vectores."""
        frame = ttk.Frame(self.tab_graficos)
        frame.pack(padx=20, pady=20)

        ttk.Label(frame, text="Vector (ej: 1,2,3):").grid(row=0, column=0, padx=5, pady=5)
        self.vector_grafico_entry = ttk.Entry(frame, width=30)
        self.vector_grafico_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(frame, text="Graficar en 2D", command=self.graficar_2d).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(frame, text="Graficar en 3D", command=self.graficar_3d).grid(row=1, column=1, padx=5, pady=5)

    def sumar_vectores(self):
        """Suma dos vectores."""
        try:
            v1 = list(map(float, self.vector1_entry.get().split(',')))
            v2 = list(map(float, self.vector2_entry.get().split(',')))
            resultado = self.math_ops.sumar_vectores(v1, v2)
            self.resultado_basicas.config(text=f"Resultado: {resultado}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al sumar vectores: {e}")

    def producto_punto(self):
        """Calcula el producto punto."""
        try:
            v1 = list(map(float, self.vector1_entry.get().split(',')))
            v2 = list(map(float, self.vector2_entry.get().split(',')))
            resultado = self.math_ops.producto_punto(v1, v2)
            self.resultado_basicas.config(text=f"Resultado: {resultado}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular producto punto: {e}")

    def multiplicar_matrices(self):
        """Multiplica dos matrices."""
        try:
            m1 = [list(map(float, row.split(','))) for row in self.matriz1_entry.get().split(';')]
            m2 = [list(map(float, row.split(','))) for row in self.matriz2_entry.get().split(';')]
            resultado = self.math_ops.multiplicar_matrices(m1, m2)
            self.resultado_matrices.config(text=f"Resultado: {resultado}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al multiplicar matrices: {e}")

    def graficar_2d(self):
        """Grafica un vector en 2D."""
        try:
            vector = list(map(float, self.vector_grafico_entry.get().split(',')))
            self.math_ops.graficar_2d([vector], ['blue'])
        except Exception as e:
            messagebox.showerror("Error", f"Error al graficar: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AlgebraLinealUI(root)
    root.mainloop()