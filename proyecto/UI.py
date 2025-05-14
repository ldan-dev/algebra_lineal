import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from linear_algebra import LinearAlgebra

class LinearAlgebraUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Álgebra Lineal - Universidad de Guanajuato: Campus Irapuato-Salamanca")
        self.root.geometry("900x1000")
        
        # Configurar estilo
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", padding=6, font=("Helvetica", 10))
        self.style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 10))
        self.style.configure("Header.TLabel", font=("Helvetica", 12, "bold"))
        self.style.configure("Author.TLabel", font=("Helvetica", 8, "italic"), foreground="#555555")

        # Crear un notebook para organizar en pestañas
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Crear las pestañas
        self.tab_vector_ops = ttk.Frame(self.notebook)
        self.tab_matrix_ops = ttk.Frame(self.notebook)
        self.tab_eigenvalues = ttk.Frame(self.notebook)
        self.tab_linear_systems = ttk.Frame(self.notebook)
        self.tab_linear_transformations = ttk.Frame(self.notebook)
        self.tab_visualization = ttk.Frame(self.notebook)
        
        # Añadir pestañas al notebook
        self.notebook.add(self.tab_vector_ops, text="Operaciones con Vectores")
        self.notebook.add(self.tab_matrix_ops, text="Operaciones con Matrices")
        self.notebook.add(self.tab_eigenvalues, text="Eigenvalores/Eigenvectores")
        self.notebook.add(self.tab_linear_systems, text="Sistemas Lineales")
        self.notebook.add(self.tab_linear_transformations, text="Transformaciones Lineales")
        self.notebook.add(self.tab_visualization, text="Visualización")
        
        # Inicializar las pestañas
        self._init_vector_ops_tab()
        self._init_matrix_ops_tab()
        self._init_eigenvalues_tab()
        self._init_linear_systems_tab()
        self._init_linear_transformations_tab()
        self._init_visualization_tab()
        
        # Figura para mostrar gráficos
        self.fig = None
        self.canvas = None
        
        # Etiqueta de autor
        self.author_label = ttk.Label(root, text="Desarrollado por: LEONARDO DANIEL AVIÑA NERI", style="Author.TLabel")
        self.author_label.pack(side=tk.BOTTOM, pady=5)
    
    # ======================== INICIALIZACIÓN DE PESTAÑAS ========================
    
    def _init_vector_ops_tab(self):
        frame = ttk.Frame(self.tab_vector_ops)
        frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(frame, text="Operaciones con Vectores", style="Header.TLabel").grid(row=0, column=0, columnspan=4, pady=10)
        
        # Control para especificar el número de vectores
        num_vectors_frame = ttk.Frame(frame)
        num_vectors_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="w")
        
        ttk.Label(num_vectors_frame, text="Número de vectores:").pack(side=tk.LEFT, padx=5)
        self.num_vectors_var = tk.StringVar(value="3")
        self.num_vectors_entry = ttk.Entry(num_vectors_frame, width=5, textvariable=self.num_vectors_var)
        self.num_vectors_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(num_vectors_frame, text="Actualizar", command=self.update_vector_entries).pack(side=tk.LEFT, padx=5)
        
        # Contenedor con scroll para los vectores
        self.vectors_canvas = tk.Canvas(frame, highlightthickness=0)
        self.vectors_canvas.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        self.vectors_scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.vectors_canvas.yview)
        self.vectors_scrollbar.grid(row=2, column=2, sticky="ns")
        self.vectors_canvas.configure(yscrollcommand=self.vectors_scrollbar.set)
        
        self.vectors_frame = ttk.Frame(self.vectors_canvas)
        self.vectors_canvas_window = self.vectors_canvas.create_window((0, 0), window=self.vectors_frame, anchor="nw")
        
        # Diccionario para almacenar las entradas de vectores
        self.vector_entries = {}
        
        # Escalar
        ttk.Label(frame, text="Escalar:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.scalar_entry = ttk.Entry(frame, width=10)
        self.scalar_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.scalar_entry.insert(0, "2")
        
        # Operaciones básicas
        ttk.Label(frame, text="Operaciones Básicas:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        vector_ops_frame = ttk.Frame(frame)
        vector_ops_frame.grid(row=4, column=1, columnspan=3, padx=5, pady=5, sticky="w")
        
        ttk.Button(vector_ops_frame, text="Suma", command=self.add_vectors).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(vector_ops_frame, text="Resta", command=self.subtract_vectors).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(vector_ops_frame, text="Producto Escalar", command=self.scalar_multiply_vector).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(vector_ops_frame, text="Producto Punto", command=self.dot_product).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(vector_ops_frame, text="Producto Cruz", command=self.cross_product).grid(row=0, column=4, padx=5, pady=5)
        
        # Operaciones adicionales
        ttk.Label(frame, text="Operaciones Adicionales:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        vector_add_ops_frame = ttk.Frame(frame)
        vector_add_ops_frame.grid(row=5, column=1, columnspan=3, padx=5, pady=5, sticky="w")
        
        ttk.Button(vector_add_ops_frame, text="Magnitud", command=self.magnitude).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(vector_add_ops_frame, text="Normalizar", command=self.normalize).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(vector_add_ops_frame, text="Ángulo", command=self.angle_between).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(vector_add_ops_frame, text="Proyección", command=self.project_vector).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(vector_add_ops_frame, text="Ind. Lineal", command=self.check_linear_independence).grid(row=0, column=4, padx=5, pady=5)
        
        # Operaciones avanzadas (nuevas)
        ttk.Label(frame, text="Operaciones Avanzadas:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        vector_advanced_ops_frame = ttk.Frame(frame)
        vector_advanced_ops_frame.grid(row=6, column=1, columnspan=3, padx=5, pady=5, sticky="w")
        
        ttk.Button(vector_advanced_ops_frame, text="Suma N Vectores", command=self.suma_n_vectores).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(vector_advanced_ops_frame, text="Resta N Vectores", command=self.resta_n_vectores).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(vector_advanced_ops_frame, text="Triple Producto", command=self.triple_producto_cruz).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(vector_advanced_ops_frame, text="Ortogonalizar", command=self.ortogonalizar).grid(row=0, column=3, padx=5, pady=5)
        
        # Visualizar vectores
        ttk.Button(frame, text="Visualizar Vectores", command=self.visualize_vectors).grid(row=7, column=0, columnspan=2, padx=5, pady=15)
        
        # Resultado
        ttk.Label(frame, text="Resultado:").grid(row=8, column=0, sticky="nw", padx=5, pady=5)
        self.vector_result = scrolledtext.ScrolledText(frame, width=50, height=10)
        self.vector_result.grid(row=8, column=1, columnspan=3, padx=5, pady=5)
        
        # Configuración para el scroll
        frame.grid_rowconfigure(2, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        
        self.vectors_frame.bind("<Configure>", self.on_frame_configure)
        self.vectors_canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Inicializar con 3 vectores por defecto
        self.update_vector_entries()
    
    def on_frame_configure(self, event):
        """Actualiza el scrollregion del canvas cuando el frame cambia de tamaño"""
        self.vectors_canvas.configure(scrollregion=self.vectors_canvas.bbox("all"))
    
    def on_canvas_configure(self, event):
        """Ajusta el ancho del frame interior cuando el canvas cambia de tamaño"""
        canvas_width = event.width
        self.vectors_canvas.itemconfig(self.vectors_canvas_window, width=canvas_width)
    
    def update_vector_entries(self):
        """Actualiza las entradas de vectores según el número especificado"""
        try:
            num_vectors = int(self.num_vectors_var.get())
            if num_vectors <= 0:
                messagebox.showerror("Error", "El número de vectores debe ser mayor que 0")
                return
                
            # Limpiar el frame de vectores
            for widget in self.vectors_frame.winfo_children():
                widget.destroy()
            
            # Vaciar el diccionario de entradas
            self.vector_entries.clear()
            
            # Crear nuevas entradas para los vectores
            for i in range(num_vectors):
                vector_frame = ttk.Frame(self.vectors_frame)
                vector_frame.pack(fill=tk.X, padx=5, pady=2)
                
                ttk.Label(vector_frame, text=f"Vector {i+1} (separar componentes con comas):").pack(side=tk.LEFT, padx=5)
                entry = ttk.Entry(vector_frame, width=30)
                entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
                
                # Insertar valores iniciales
                if i == 0:
                    entry.insert(0, "1, 2, 3")
                elif i == 1:
                    entry.insert(0, "4, 5, 6")
                elif i == 2:
                    entry.insert(0, "7, 8, 9")
                else:
                    entry.insert(0, f"{i+1}, {i+2}, {i+3}")
                
                self.vector_entries[i] = entry
            
            # Actualizar la vista del scroll
            self.vectors_frame.update_idletasks()
            self.vectors_canvas.configure(scrollregion=self.vectors_canvas.bbox("all"))
            
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingrese un número entero válido para la cantidad de vectores")
            
    def get_all_vectors(self):
        """Obtiene todos los vectores de las entradas"""
        vectors = []
        for i in range(len(self.vector_entries)):
            vector = self.parse_vector(self.vector_entries[i].get())
            if vector is not None:
                vectors.append(vector)
        return vectors
    
    def _init_matrix_ops_tab(self):
        frame = ttk.Frame(self.tab_matrix_ops)
        frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(frame, text="Operaciones con Matrices", style="Header.TLabel").grid(row=0, column=0, columnspan=4, pady=10)
        
        # Entradas para matrices
        ttk.Label(frame, text="Matriz 1 (filas separadas por ';', elementos por ','):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.matrix1_entry = ttk.Entry(frame, width=40)
        self.matrix1_entry.grid(row=1, column=1, padx=5, pady=5)
        self.matrix1_entry.insert(0, "1, 2; 3, 4")
        
        ttk.Label(frame, text="Matriz 2 (filas separadas por ';', elementos por ','):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.matrix2_entry = ttk.Entry(frame, width=40)
        self.matrix2_entry.grid(row=2, column=1, padx=5, pady=5)
        self.matrix2_entry.insert(0, "5, 6; 7, 8")
        
        ttk.Label(frame, text="Escalar:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.matrix_scalar_entry = ttk.Entry(frame, width=10)
        self.matrix_scalar_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.matrix_scalar_entry.insert(0, "2")
        
        # Operaciones básicas
        ttk.Label(frame, text="Operaciones Básicas:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        matrix_ops_frame = ttk.Frame(frame)
        matrix_ops_frame.grid(row=4, column=1, columnspan=3, padx=5, pady=5, sticky="w")
        
        ttk.Button(matrix_ops_frame, text="Suma", command=self.add_matrices).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(matrix_ops_frame, text="Resta", command=self.subtract_matrices).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(matrix_ops_frame, text="Producto Escalar", command=self.scalar_multiply_matrix).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(matrix_ops_frame, text="Multiplicación", command=self.matrix_multiply).grid(row=0, column=3, padx=5, pady=5)
        
        # Operaciones adicionales
        ttk.Label(frame, text="Operaciones Adicionales:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        matrix_add_ops_frame = ttk.Frame(frame)
        matrix_add_ops_frame.grid(row=5, column=1, columnspan=3, padx=5, pady=5, sticky="w")
        
        ttk.Button(matrix_add_ops_frame, text="Transpuesta", command=self.transpose).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(matrix_add_ops_frame, text="Inversa", command=self.inverse).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(matrix_add_ops_frame, text="Determinante", command=self.determinant).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(matrix_add_ops_frame, text="Traza", command=self.trace).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(matrix_add_ops_frame, text="Rango", command=self.rank).grid(row=0, column=4, padx=5, pady=5)
        
        # Resultado
        ttk.Label(frame, text="Resultado:").grid(row=6, column=0, sticky="nw", padx=5, pady=5)
        self.matrix_result = scrolledtext.ScrolledText(frame, width=50, height=10)
        self.matrix_result.grid(row=6, column=1, columnspan=3, padx=5, pady=5)
    
    def _init_eigenvalues_tab(self):
        frame = ttk.Frame(self.tab_eigenvalues)
        frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(frame, text="Eigenvalores y Eigenvectores", style="Header.TLabel").grid(row=0, column=0, columnspan=2, pady=10)
        
        # Entrada para matriz
        ttk.Label(frame, text="Matriz (filas separadas por ';', elementos por ','):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.eigen_matrix_entry = ttk.Entry(frame, width=40)
        self.eigen_matrix_entry.grid(row=1, column=1, padx=5, pady=5)
        self.eigen_matrix_entry.insert(0, "2, 1; 1, 2")
        
        ttk.Label(frame, text="Eigenvalor (para calcular eigenespacio):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.eigenvalue_entry = ttk.Entry(frame, width=10)
        self.eigenvalue_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.eigenvalue_entry.insert(0, "3")
        
        # Operaciones
        ttk.Label(frame, text="Calcular:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        eigen_ops_frame = ttk.Frame(frame)
        eigen_ops_frame.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Button(eigen_ops_frame, text="Eigenvalores", command=self.calc_eigenvalues).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(eigen_ops_frame, text="Eigenvectores", command=self.calc_eigenvectors).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(eigen_ops_frame, text="Eigenespacio", command=self.calc_eigenspace).grid(row=0, column=2, padx=5, pady=5)
        
        # Resultado
        ttk.Label(frame, text="Resultado:").grid(row=4, column=0, sticky="nw", padx=5, pady=5)
        self.eigen_result = scrolledtext.ScrolledText(frame, width=50, height=15)
        self.eigen_result.grid(row=4, column=1, padx=5, pady=5)
    
    def _init_linear_systems_tab(self):
        frame = ttk.Frame(self.tab_linear_systems)
        frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(frame, text="Sistemas de Ecuaciones Lineales", style="Header.TLabel").grid(row=0, column=0, columnspan=2, pady=10)
        
        # Entradas
        ttk.Label(frame, text="Matriz de coeficientes A (filas separadas por ';', elementos por ','):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.coef_matrix_entry = ttk.Entry(frame, width=40)
        self.coef_matrix_entry.grid(row=1, column=1, padx=5, pady=5)
        self.coef_matrix_entry.insert(0, "2, 1; 1, 1")
        
        ttk.Label(frame, text="Vector b (elementos separados por comas):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.b_vector_entry = ttk.Entry(frame, width=40)
        self.b_vector_entry.grid(row=2, column=1, padx=5, pady=5)
        self.b_vector_entry.insert(0, "3, 2")
        
        # Operaciones
        ttk.Label(frame, text="Método de solución:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        linear_ops_frame = ttk.Frame(frame)
        linear_ops_frame.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Button(linear_ops_frame, text="Método Directo", command=self.solve_direct).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(linear_ops_frame, text="Método LU", command=self.solve_lu).grid(row=0, column=1, padx=5, pady=5)
        
        # Resultado
        ttk.Label(frame, text="Resultado:").grid(row=4, column=0, sticky="nw", padx=5, pady=5)
        self.system_result = scrolledtext.ScrolledText(frame, width=50, height=15)
        self.system_result.grid(row=4, column=1, padx=5, pady=5)
        
    def _init_linear_transformations_tab(self):
        frame = ttk.Frame(self.tab_linear_transformations)
        frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(frame, text="Transformaciones Lineales", style="Header.TLabel").grid(row=0, column=0, columnspan=2, pady=10)
        
        # Entradas
        ttk.Label(frame, text="Matriz de transformación (filas separadas por ';', elementos por ','):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.transform_matrix_entry = ttk.Entry(frame, width=40)
        self.transform_matrix_entry.grid(row=1, column=1, padx=5, pady=5)
        self.transform_matrix_entry.insert(0, "0, -1; 1, 0")
        
        ttk.Label(frame, text="Vector a transformar (elementos separados por comas):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.transform_vector_entry = ttk.Entry(frame, width=40)
        self.transform_vector_entry.grid(row=2, column=1, padx=5, pady=5)
        self.transform_vector_entry.insert(0, "1, 0")
        
        # Combinación lineal
        ttk.Label(frame, text="Vectores para combinación lineal (cada vector separado por ';', elementos por ','):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.comb_vectors_entry = ttk.Entry(frame, width=40)
        self.comb_vectors_entry.grid(row=3, column=1, padx=5, pady=5)
        self.comb_vectors_entry.insert(0, "1, 0; 0, 1")
        
        ttk.Label(frame, text="Coeficientes (separados por comas):").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.comb_coefs_entry = ttk.Entry(frame, width=40)
        self.comb_coefs_entry.grid(row=4, column=1, padx=5, pady=5)
        self.comb_coefs_entry.insert(0, "2, 3")
        
        # Bases y transformación
        ttk.Label(frame, text="Base del dominio (cada vector separado por ';', elementos por ','):").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.domain_basis_entry = ttk.Entry(frame, width=40)
        self.domain_basis_entry.grid(row=5, column=1, padx=5, pady=5)
        self.domain_basis_entry.insert(0, "1, 0; 0, 1")
        
        ttk.Label(frame, text="Imágenes de la base (cada vector separado por ';', elementos por ','):").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.codomain_vectors_entry = ttk.Entry(frame, width=40)
        self.codomain_vectors_entry.grid(row=6, column=1, padx=5, pady=5)
        self.codomain_vectors_entry.insert(0, "0, 1; -1, 0")
        
        # Operaciones
        transform_ops_frame = ttk.Frame(frame)
        transform_ops_frame.grid(row=7, column=0, columnspan=2, padx=5, pady=15)
        
        ttk.Button(transform_ops_frame, text="Aplicar Transformación", command=self.apply_transformation).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(transform_ops_frame, text="Combinación Lineal", command=self.linear_combination).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(transform_ops_frame, text="Encontrar Matriz de Transformación", command=self.find_transformation_matrix).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(transform_ops_frame, text="Visualizar Transformación 2D", command=self.visualize_transformation).grid(row=0, column=3, padx=5, pady=5)
        
        # Resultado
        ttk.Label(frame, text="Resultado:").grid(row=8, column=0, sticky="nw", padx=5, pady=5)
        self.transform_result = scrolledtext.ScrolledText(frame, width=50, height=10)
        self.transform_result.grid(row=8, column=1, padx=5, pady=5)
        
    def _init_visualization_tab(self):
        frame = ttk.Frame(self.tab_visualization)
        frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(frame, text="Visualización", style="Header.TLabel").grid(row=0, column=0, columnspan=2, pady=10)
        
        # Entrada para vectores 2D
        ttk.Label(frame, text="Vectores 2D (cada vector separado por ';', elementos por ','):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.viz_vectors_2d_entry = ttk.Entry(frame, width=40)
        self.viz_vectors_2d_entry.grid(row=1, column=1, padx=5, pady=5)
        self.viz_vectors_2d_entry.insert(0, "1, 0; 0, 1; 1, 1")
        
        # Entrada para vectores 3D
        ttk.Label(frame, text="Vectores 3D (cada vector separado por ';', elementos por ','):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.viz_vectors_3d_entry = ttk.Entry(frame, width=40)
        self.viz_vectors_3d_entry.grid(row=2, column=1, padx=5, pady=5)
        self.viz_vectors_3d_entry.insert(0, "1, 0, 0; 0, 1, 0; 0, 0, 1")
        
        # Entrada para transformación lineal
        ttk.Label(frame, text="Matriz de transformación 2D (filas separadas por ';', elementos por ','):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.viz_transform_entry = ttk.Entry(frame, width=40)
        self.viz_transform_entry.grid(row=3, column=1, padx=5, pady=5)
        self.viz_transform_entry.insert(0, "0, -1; 1, 0")
        
        # Botones para visualizar
        viz_ops_frame = ttk.Frame(frame)
        viz_ops_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=15)
        
        ttk.Button(viz_ops_frame, text="Visualizar Vectores 2D", command=self.plot_vectors_2d).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(viz_ops_frame, text="Visualizar Vectores 3D", command=self.plot_vectors_3d).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(viz_ops_frame, text="Visualizar Transformación 2D", command=self.plot_transform_2d).grid(row=0, column=2, padx=5, pady=5)
        
        # Frame para mostrar gráficos
        self.plot_frame = ttk.Frame(frame)
        self.plot_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
    
    # ======================== UTILIDADES DE PARSING ========================
    
    def parse_vector(self, vector_str):
        """Convierte una cadena en un vector numpy"""
        try:
            return np.array([float(x.strip()) for x in vector_str.split(',')])
        except:
            messagebox.showerror("Error", "Error al parsear el vector. Formato esperado: '1, 2, 3'")
            return None
    
    def parse_vectors(self, vectors_str):
        """Convierte una cadena en una lista de vectores numpy"""
        try:
            return [np.array([float(x.strip()) for x in vector.split(',')]) 
                    for vector in vectors_str.split(';')]
        except:
            messagebox.showerror("Error", "Error al parsear los vectores. Formato esperado: '1, 2; 3, 4'")
            return None
    
    def parse_matrix(self, matrix_str):
        """Convierte una cadena en una matriz numpy"""
        try:
            rows = matrix_str.split(';')
            return np.array([[float(x.strip()) for x in row.split(',')] for row in rows])
        except:
            messagebox.showerror("Error", "Error al parsear la matriz. Formato esperado: '1, 2; 3, 4'")
            return None
    
    def format_vector(self, vector):
        """Formatea un vector para mostrar"""
        return "[" + ", ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in vector) + "]"
    
    def format_matrix(self, matrix):
        """Formatea una matriz para mostrar"""
        return "\n".join("[" + ", ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in row) + "]" for row in matrix)
    
    # ======================== OPERACIONES CON VECTORES ========================
    
    def add_vectors(self):
        vectors = self.get_all_vectors()
        if len(vectors) < 2:
            messagebox.showerror("Error", "Se necesitan al menos dos vectores para sumar")
            return
        
        try:
            result = vectors[0]
            for i in range(1, len(vectors)):
                result = LinearAlgebra.add_vectors(result, vectors[i])
            
            # Construir cadena para mostrar la operación
            vectors_str = " + ".join([self.format_vector(v) for v in vectors])
            
            self.vector_result.delete(1.0, tk.END)
            self.vector_result.insert(tk.END, f"Suma de vectores:\n{vectors_str} = {self.format_vector(result)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al sumar vectores: {str(e)}")
    
    def subtract_vectors(self):
        vectors = self.get_all_vectors()
        if len(vectors) < 2:
            messagebox.showerror("Error", "Se necesitan al menos dos vectores para restar")
            return
        
        try:
            result = vectors[0]
            for i in range(1, len(vectors)):
                result = LinearAlgebra.subtract_vectors(result, vectors[i])
            
            # Construir cadena para mostrar la operación
            vectors_str = " - ".join([self.format_vector(v) for v in vectors])
            
            self.vector_result.delete(1.0, tk.END)
            self.vector_result.insert(tk.END, f"Resta de vectores:\n{vectors_str} = {self.format_vector(result)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al restar vectores: {str(e)}")
    
    def scalar_multiply_vector(self):
        try:
            scalar = float(self.scalar_entry.get())
            vectors = self.get_all_vectors()
            if not vectors:
                return
            
            results = []
            result_text = ""
            
            for i, vector in enumerate(vectors):
                result = LinearAlgebra.scalar_multiply(scalar, vector)
                results.append(result)
                result_text += f"Vector {i+1}: {scalar} * {self.format_vector(vector)} = {self.format_vector(result)}\n\n"
            
            self.vector_result.delete(1.0, tk.END)
            self.vector_result.insert(tk.END, f"Multiplicación escalar:\n{result_text}")
        except ValueError:
            messagebox.showerror("Error", "El escalar debe ser un número válido")
        except Exception as e:
            messagebox.showerror("Error", f"Error en la multiplicación escalar: {str(e)}")
    
    def dot_product(self):
        vectors = self.get_all_vectors()
        if len(vectors) < 2:
            messagebox.showerror("Error", "Se necesitan al menos dos vectores para calcular el producto punto")
            return
        
        try:
            result = LinearAlgebra.dot_product(vectors[0], vectors[1])
            self.vector_result.delete(1.0, tk.END)
            self.vector_result.insert(tk.END, f"Producto punto:\n{result:.4f}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular el producto punto: {str(e)}")
    
    def cross_product(self):
        vectors = self.get_all_vectors()
        if len(vectors) < 2:
            messagebox.showerror("Error", "Se necesitan al menos dos vectores para calcular el producto cruz")
            return
        
        try:
            result = LinearAlgebra.cross_product(vectors[0], vectors[1])
            self.vector_result.delete(1.0, tk.END)
            self.vector_result.insert(tk.END, f"Producto cruz:\n{result}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular el producto cruz: {str(e)}")
    
    def magnitude(self):
        vectors = self.get_all_vectors()
        if not vectors:
            return
        
        try:
            result_text = "Magnitudes de los vectores:\n\n"
            
            for i, vector in enumerate(vectors):
                result = LinearAlgebra.magnitude(vector)
                result_text += f"Vector {i+1}: ||{self.format_vector(vector)}|| = {result:.4f}\n"
            
            self.vector_result.delete(1.0, tk.END)
            self.vector_result.insert(tk.END, result_text)
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular la magnitud: {str(e)}")
    
    def normalize(self):
        vectors = self.get_all_vectors()
        if not vectors:
            return
        
        try:
            result_text = "Vectores normalizados:\n\n"
            
            for i, vector in enumerate(vectors):
                result = LinearAlgebra.normalize(vector)
                magnitude = LinearAlgebra.magnitude(result)
                result_text += f"Vector {i+1}: {self.format_vector(vector)} → {self.format_vector(result)}\n"
                result_text += f"Magnitud verificada: {magnitude:.4f}\n\n"
            
            self.vector_result.delete(1.0, tk.END)
            self.vector_result.insert(tk.END, result_text)
        except Exception as e:
            messagebox.showerror("Error", f"Error al normalizar el vector: {str(e)}")
    
    def angle_between(self):
        vectors = self.get_all_vectors()
        if len(vectors) < 2:
            messagebox.showerror("Error", "Se necesitan al menos dos vectores para calcular el ángulo")
            return
        
        try:
            result_deg = LinearAlgebra.angle_between(vectors[0], vectors[1], in_degrees=True)
            result_rad = LinearAlgebra.angle_between(vectors[0], vectors[1], in_degrees=False)
            self.vector_result.delete(1.0, tk.END)
            self.vector_result.insert(tk.END, f"Ángulo entre vectores:\n{result_deg:.4f}° ({result_rad:.4f} radianes)")
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular el ángulo: {str(e)}")
    
    def project_vector(self):
        vectors = self.get_all_vectors()
        if len(vectors) < 2:
            messagebox.showerror("Error", "Se necesitan al menos dos vectores para calcular la proyección")
            return
        
        try:
            result = LinearAlgebra.project_vector(vectors[0], vectors[1])
            self.vector_result.delete(1.0, tk.END)
            self.vector_result.insert(tk.END, f"Proyección de {vectors[0]} sobre {vectors[1]}:\n{self.format_vector(result)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular la proyección: {str(e)}")
    
    def check_linear_independence(self):
        vectors = self.get_all_vectors()
        if not vectors:
            return
        
        try:
            result = LinearAlgebra.check_linear_independence(vectors)
            self.vector_result.delete(1.0, tk.END)
            if result:
                self.vector_result.insert(tk.END, "Los vectores son linealmente independientes.")
            else:
                self.vector_result.insert(tk.END, "Los vectores son linealmente dependientes.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al verificar independencia lineal: {str(e)}")
    
    def visualize_vectors(self):
        vectors = self.get_all_vectors()
        if not vectors:
            return
        
        # Verificar dimensión de los vectores
        vector_dim = len(vectors[0])
        for v in vectors:
            if len(v) != vector_dim:
                messagebox.showerror("Error", "Todos los vectores deben tener la misma dimensión")
                return
        
        if vector_dim == 2:
            # Vectores 2D
            try:
                # Colores para los vectores
                colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
                color_list = [colors[i % len(colors)] for i in range(len(vectors))]
                
                fig, ax = plt.subplots(figsize=(8, 8))
                
                for i, (v, c) in enumerate(zip(vectors, color_list)):
                    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=c, label=f"v{i+1}")
                    ax.text(v[0]*1.1, v[1]*1.1, f"v{i+1}=({v[0]}, {v[1]})", color=c)
                
                ax.set_xlim([-10, 10])
                ax.set_ylim([-10, 10])
                ax.grid(True)
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                ax.set_title("Vectores en 2D")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.legend()
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al visualizar vectores 2D: {str(e)}")
        elif vector_dim == 3:
            # Vectores 3D
            try:
                # Colores para los vectores
                colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
                color_list = [colors[i % len(colors)] for i in range(len(vectors))]
                
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                for i, (v, c) in enumerate(zip(vectors, color_list)):
                    ax.quiver(0, 0, 0, v[0], v[1], v[2], color=c, label=f"v{i+1}")
                    ax.text(v[0], v[1], v[2], f"v{i+1}=({v[0]}, {v[1]}, {v[2]})", color=c)
                
                ax.set_xlim([-10, 10])
                ax.set_ylim([-10, 10])
                ax.set_zlim([-10, 10])
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title("Vectores en 3D")
                ax.legend()
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al visualizar vectores 3D: {str(e)}")
        else:
            messagebox.showerror("Error", "Solo se pueden visualizar vectores 2D o 3D")
            
    def suma_n_vectores(self):
        """Suma múltiples vectores"""
        vectors = self.get_all_vectors()
        if len(vectors) < 2:
            messagebox.showerror("Error", "Se necesitan al menos dos vectores para sumar")
            return
        
        try:
            result = LinearAlgebra.suma_n_vectores(*vectors)
            
            # Construir cadena para mostrar la operación
            vectors_str = " + ".join([self.format_vector(v) for v in vectors])
            
            self.vector_result.delete(1.0, tk.END)
            self.vector_result.insert(tk.END, f"Suma de vectores:\n{vectors_str} = {self.format_vector(result)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al sumar vectores: {str(e)}")
    
    def resta_n_vectores(self):
        """Resta múltiples vectores"""
        vectors = self.get_all_vectors()
        if len(vectors) < 2:
            messagebox.showerror("Error", "Se necesitan al menos dos vectores para restar")
            return
        
        try:
            result = LinearAlgebra.resta_n_vectores(*vectors)
            
            # Construir cadena para mostrar la operación
            vectors_str = " - ".join([self.format_vector(v) for v in vectors])
            
            self.vector_result.delete(1.0, tk.END)
            self.vector_result.insert(tk.END, f"Resta de vectores:\n{vectors_str} = {self.format_vector(result)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al restar vectores: {str(e)}")
    
    # ======================== OPERACIONES AVANZADAS CON VECTORES ========================
    
    def triple_producto_cruz(self):
        """Calcula el triple producto cruz: v1·(v2×v3)"""
        vectors = self.get_all_vectors()
        if len(vectors) < 3:
            messagebox.showerror("Error", "Se necesitan tres vectores para calcular el triple producto cruz")
            return
        
        try:
            result = LinearAlgebra.triple_producto_cruz(vectors[0], vectors[1], vectors[2])
            self.vector_result.delete(1.0, tk.END)
            self.vector_result.insert(tk.END, f"Triple producto cruz:\n{result:.4f}")
            self.vector_result.insert(tk.END, f"\n\nInterpretación geométrica: Volumen del paralelepípedo")
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular el triple producto cruz: {str(e)}")
    
    def ortogonalizar(self):
        """Ortogonaliza un vector respecto a otro"""
        vectors = self.get_all_vectors()
        if len(vectors) < 2:
            messagebox.showerror("Error", "Se necesitan al menos dos vectores para ortogonalizar")
            return
        
        try:
            result = LinearAlgebra.ortogonalizar(vectors[0], vectors[1])
            self.vector_result.delete(1.0, tk.END)
            self.vector_result.insert(tk.END, f"Vector {vectors[0]} ortogonalizado respecto a {vectors[1]}:\n{self.format_vector(result)}")
            
            # Verificar ortogonalidad
            dot_product = LinearAlgebra.dot_product(result, vectors[1])
            self.vector_result.insert(tk.END, f"\n\nVerificación de ortogonalidad:\nProducto punto = {dot_product:.10f}")
            if abs(dot_product) < 1e-10:
                self.vector_result.insert(tk.END, "\nLos vectores son ortogonales ✓")
            else:
                self.vector_result.insert(tk.END, "\nAtención: Los vectores no son perfectamente ortogonales debido a errores de redondeo numérico.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al ortogonalizar vector: {str(e)}")
    
    # ======================== OPERACIONES CON MATRICES ========================
    
    def add_matrices(self):
        m1 = self.parse_matrix(self.matrix1_entry.get())
        m2 = self.parse_matrix(self.matrix2_entry.get())
        if m1 is None or m2 is None:
            return
        
        try:
            result = LinearAlgebra.add_matrices(m1, m2)
            self.matrix_result.delete(1.0, tk.END)
            self.matrix_result.insert(tk.END, f"Suma de matrices:\n\n{self.format_matrix(result)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al sumar matrices: {str(e)}")
    
    def subtract_matrices(self):
        m1 = self.parse_matrix(self.matrix1_entry.get())
        m2 = self.parse_matrix(self.matrix2_entry.get())
        if m1 is None or m2 is None:
            return
        
        try:
            result = LinearAlgebra.subtract_matrices(m1, m2)
            self.matrix_result.delete(1.0, tk.END)
            self.matrix_result.insert(tk.END, f"Resta de matrices:\n\n{self.format_matrix(result)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al restar matrices: {str(e)}")
    
    def scalar_multiply_matrix(self):
        try:
            scalar = float(self.matrix_scalar_entry.get())
            matrix = self.parse_matrix(self.matrix1_entry.get())
            if matrix is None:
                return
            
            result = LinearAlgebra.scalar_matrix_multiply(scalar, matrix)
            self.matrix_result.delete(1.0, tk.END)
            self.matrix_result.insert(tk.END, f"Multiplicación escalar:\n{scalar} * matriz =\n\n{self.format_matrix(result)}")
        except ValueError:
            messagebox.showerror("Error", "El escalar debe ser un número válido")
        except Exception as e:
            messagebox.showerror("Error", f"Error en la multiplicación escalar: {str(e)}")
    
    def matrix_multiply(self):
        m1 = self.parse_matrix(self.matrix1_entry.get())
        m2 = self.parse_matrix(self.matrix2_entry.get())
        if m1 is None or m2 is None:
            return
        
        try:
            result = LinearAlgebra.matrix_multiply(m1, m2)
            self.matrix_result.delete(1.0, tk.END)
            self.matrix_result.insert(tk.END, f"Multiplicación de matrices:\n\n{self.format_matrix(result)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al multiplicar matrices: {str(e)}")
    
    def transpose(self):
        matrix = self.parse_matrix(self.matrix1_entry.get())
        if matrix is None:
            return
        
        try:
            result = LinearAlgebra.transpose(matrix)
            self.matrix_result.delete(1.0, tk.END)
            self.matrix_result.insert(tk.END, f"Matriz transpuesta:\n\n{self.format_matrix(result)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al transponer la matriz: {str(e)}")
    
    def inverse(self):
        matrix = self.parse_matrix(self.matrix1_entry.get())
        if matrix is None:
            return
        
        try:
            result = LinearAlgebra.inverse(matrix)
            self.matrix_result.delete(1.0, tk.END)
            self.matrix_result.insert(tk.END, f"Matriz inversa:\n\n{self.format_matrix(result)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular la inversa: {str(e)}")
    
    def determinant(self):
        matrix = self.parse_matrix(self.matrix1_entry.get())
        if matrix is None:
            return
        
        try:
            result = LinearAlgebra.determinant(matrix)
            self.matrix_result.delete(1.0, tk.END)
            self.matrix_result.insert(tk.END, f"Determinante: {result:.4f}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular el determinante: {str(e)}")
    
    def trace(self):
        matrix = self.parse_matrix(self.matrix1_entry.get())
        if matrix is None:
            return
        
        try:
            result = LinearAlgebra.trace(matrix)
            self.matrix_result.delete(1.0, tk.END)
            self.matrix_result.insert(tk.END, f"Traza: {result:.4f}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular la traza: {str(e)}")
    
    def rank(self):
        matrix = self.parse_matrix(self.matrix1_entry.get())
        if matrix is None:
            return
        
        try:
            result = LinearAlgebra.rank(matrix)
            self.matrix_result.delete(1.0, tk.END)
            self.matrix_result.insert(tk.END, f"Rango: {result}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular el rango: {str(e)}")
    
    # ======================== EIGENVALUES & EIGENVECTORS ========================
    
    def calc_eigenvalues(self):
        matrix = self.parse_matrix(self.eigen_matrix_entry.get())
        if matrix is None:
            return
        
        try:
            result = LinearAlgebra.eigenvalues(matrix)
            self.eigen_result.delete(1.0, tk.END)
            self.eigen_result.insert(tk.END, "Eigenvalores:\n\n")
            for i, val in enumerate(result):
                self.eigen_result.insert(tk.END, f"λ{i+1} = {val:.4f}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular eigenvalores: {str(e)}")
    
    def calc_eigenvectors(self):
        matrix = self.parse_matrix(self.eigen_matrix_entry.get())
        if matrix is None:
            return
        
        try:
            values, vectors = LinearAlgebra.eigenvectors(matrix)
            self.eigen_result.delete(1.0, tk.END)
            self.eigen_result.insert(tk.END, "Eigenvalores y eigenvectores:\n\n")
            
            for i, (val, vec) in enumerate(zip(values, vectors.T)):
                self.eigen_result.insert(tk.END, f"λ{i+1} = {val:.4f}\n")
                self.eigen_result.insert(tk.END, f"v{i+1} = {self.format_vector(vec)}\n\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular eigenvectores: {str(e)}")
    
    def calc_eigenspace(self):
        matrix = self.parse_matrix(self.eigen_matrix_entry.get())
        if matrix is None:
            return
        
        try:
            eigenvalue = float(self.eigenvalue_entry.get())
            result = LinearAlgebra.eigenspace(matrix, eigenvalue)
            
            self.eigen_result.delete(1.0, tk.END)
            self.eigen_result.insert(tk.END, f"Eigenespacio para λ = {eigenvalue}:\n\n")
            
            if result.size == 0:
                self.eigen_result.insert(tk.END, "No hay eigenvectores para este eigenvalor.")
            else:
                self.eigen_result.insert(tk.END, "Base del eigenespacio:\n")
                for i, vec in enumerate(result):
                    self.eigen_result.insert(tk.END, f"v{i+1} = {self.format_vector(vec)}\n")
                
                self.eigen_result.insert(tk.END, f"\nDimensión del eigenespacio: {len(result)}")
        except ValueError:
            messagebox.showerror("Error", "El eigenvalor debe ser un número válido")
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular el eigenespacio: {str(e)}")
    
    # ======================== SISTEMAS LINEALES ========================
    
    def solve_direct(self):
        A = self.parse_matrix(self.coef_matrix_entry.get())
        b = self.parse_vector(self.b_vector_entry.get())
        if A is None or b is None:
            return
        
        try:
            result = LinearAlgebra.solve_linear_system(A, b)
            self.system_result.delete(1.0, tk.END)
            self.system_result.insert(tk.END, "Solución del sistema Ax = b:\n\n")
            self.system_result.insert(tk.END, f"A =\n{self.format_matrix(A)}\n\n")
            self.system_result.insert(tk.END, f"b = {self.format_vector(b)}\n\n")
            self.system_result.insert(tk.END, f"x = {self.format_vector(result)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al resolver el sistema: {str(e)}")
    
    def solve_lu(self):
        A = self.parse_matrix(self.coef_matrix_entry.get())
        b = self.parse_vector(self.b_vector_entry.get())
        if A is None or b is None:
            return
        
        try:
            result = LinearAlgebra.solve_linear_system_lu(A, b)
            self.system_result.delete(1.0, tk.END)
            self.system_result.insert(tk.END, "Solución del sistema Ax = b (mediante LU):\n\n")
            self.system_result.insert(tk.END, f"x = {self.format_vector(result)}")
            
            # Verificar la solución
            verification = np.allclose(A @ result, b)
            if verification:
                self.system_result.insert(tk.END, "\n\nLa solución es correcta: A·x ≈ b")
            else:
                self.system_result.insert(tk.END, "\n\nAdvertencia: La solución podría ser aproximada.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al resolver el sistema mediante LU: {str(e)}")
    
    # ======================== TRANSFORMACIONES LINEALES ========================
    
    def apply_transformation(self):
        matrix = self.parse_matrix(self.transform_matrix_entry.get())
        vector = self.parse_vector(self.transform_vector_entry.get())
        if matrix is None or vector is None:
            return
        
        try:
            result = LinearAlgebra.apply_linear_transformation(matrix, vector)
            self.transform_result.delete(1.0, tk.END)
            self.transform_result.insert(tk.END, "Aplicación de transformación lineal:\n\n")
            self.transform_result.insert(tk.END, f"T({self.format_vector(vector)}) = {self.format_vector(result)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al aplicar la transformación: {str(e)}")
    
    def linear_combination(self):
        vectors = self.parse_vectors(self.comb_vectors_entry.get())
        if vectors is None:
            return
        
        try:
            coefficients = [float(x.strip()) for x in self.comb_coefs_entry.get().split(',')]
            if len(vectors) != len(coefficients):
                messagebox.showerror("Error", "El número de vectores y coeficientes no coincide")
                return
            
            result = LinearAlgebra.linear_combination(vectors, coefficients)
            self.transform_result.delete(1.0, tk.END)
            self.transform_result.insert(tk.END, "Combinación lineal de vectores:\n\n")
            
            combo_str = " + ".join([f"{c}·{self.format_vector(v)}" for c, v in zip(coefficients, vectors)])
            self.transform_result.insert(tk.END, f"{combo_str} = {self.format_vector(result)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular la combinación lineal: {str(e)}")
    
    def find_transformation_matrix(self):
        domain_basis = self.parse_vectors(self.domain_basis_entry.get())
        codomain_vectors = self.parse_vectors(self.codomain_vectors_entry.get())
        if domain_basis is None or codomain_vectors is None:
            return
        
        try:
            result = LinearAlgebra.transformation_matrix(domain_basis, codomain_vectors)
            self.transform_result.delete(1.0, tk.END)
            self.transform_result.insert(tk.END, "Matriz de transformación lineal:\n\n")
            self.transform_result.insert(tk.END, f"{self.format_matrix(result)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al encontrar la matriz de transformación: {str(e)}")
    
    def visualize_transformation(self):
        matrix = self.parse_matrix(self.transform_matrix_entry.get())
        if matrix is None:
            return
        
        try:
            if matrix.shape != (2, 2):
                messagebox.showerror("Error", "La matriz debe ser 2x2 para visualizar la transformación")
                return
            
            LinearAlgebra.plot_linear_transformation_2d(matrix)
        except Exception as e:
            messagebox.showerror("Error", f"Error al visualizar la transformación: {str(e)}")
    
    # ======================== VISUALIZACIÓN ========================
    
    def plot_vectors_2d(self):
        vectors = self.parse_vectors(self.viz_vectors_2d_entry.get())
        if vectors is None:
            return
        
        try:
            for vector in vectors:
                if len(vector) != 2:
                    messagebox.showerror("Error", "Todos los vectores deben ser 2D")
                    return
            
            # Limpiar frame de gráfico anterior
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
                
            # Colores para los vectores
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            color_list = [colors[i % len(colors)] for i in range(len(vectors))]
            
            # Generar nuevo gráfico
            fig, ax = plt.subplots(figsize=(6, 6))
            
            for i, (v, c) in enumerate(zip(vectors, color_list)):
                ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=c, label=f"v{i+1}")
                ax.text(v[0]*1.1, v[1]*1.1, f"v{i+1}=({v[0]}, {v[1]})", color=c)
            
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.grid(True)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            ax.set_title("Vectores en 2D")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            
            # Mostrar en el canvas
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al visualizar vectores 2D: {str(e)}")
    
    def plot_vectors_3d(self):
        vectors = self.parse_vectors(self.viz_vectors_3d_entry.get())
        if vectors is None:
            return
        
        try:
            for vector in vectors:
                if len(vector) != 3:
                    messagebox.showerror("Error", "Todos los vectores deben ser 3D")
                    return
            
            # Limpiar frame de gráfico anterior
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
                
            # Colores para los vectores
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            color_list = [colors[i % len(colors)] for i in range(len(vectors))]
            
            # Generar nuevo gráfico
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            for i, (v, c) in enumerate(zip(vectors, color_list)):
                ax.quiver(0, 0, 0, v[0], v[1], v[2], color=c, label=f"v{i+1}")
                ax.text(v[0], v[1], v[2], f"v{i+1}=({v[0]}, {v[1]}, {v[2]})", color=c)
            
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_zlim([-10, 10])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title("Vectores en 3D")
            ax.legend()
            
            # Mostrar en el canvas
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al visualizar vectores 3D: {str(e)}")
    
    def plot_transform_2d(self):
        matrix = self.parse_matrix(self.viz_transform_entry.get())
        if matrix is None:
            return
        
        try:
            if matrix.shape != (2, 2):
                messagebox.showerror("Error", "La matriz debe ser 2x2 para visualizar la transformación")
                return
            
            # Limpiar frame de gráfico anterior
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            
            # Generar nuevo gráfico
            fig, ax = LinearAlgebra.plot_linear_transformation_2d(matrix, figsize=(12, 6))
            
            # Mostrar en el canvas
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al visualizar la transformación: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LinearAlgebraUI(root)
    root.mainloop()