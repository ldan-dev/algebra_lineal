"""
LEONARDO DANIEL AVIÑA NERI
Fecha: 28/04/2025 (dd/mm/aaaa)
CARRERA: LIDIA
Universidad de Guanajuato - Campus Irapuato-Salamanca
Correo: ld.avinaneri@ugto.mx
UDA: Álgebra Lineal
DESCRIPCION: Interfaz gráfica para utilizar los métodos de la clase AlgebraLineal
             sin necesidad de escribir código.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from io import StringIO
import inspect
from functools import partial
from PIL import Image, ImageTk

# Importar la clase AlgebraLineal
from AlgebraLineal import AlgebraLineal

class RedirectOutput:
    """
    Clase para redireccionar la salida de consola a un widget de tkinter
    """
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = StringIO()

    def write(self, string):
        self.buffer.write(string)
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")

    def flush(self):
        self.buffer.flush()

class VectorEntryFrame(ttk.Frame):
    """
    Frame para la entrada de vectores de dimensión variable
    """
    def __init__(self, parent, label_text, initial_dim=1):
        super().__init__(parent, padding=10)
        self.parent = parent
        self.label_text = label_text
        self.entries = []
        
        # Frame superior con etiqueta y selector de dimensión
        top_frame = ttk.Frame(self)
        top_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(top_frame, text=label_text).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(top_frame, text="Dimensión:").pack(side=tk.LEFT, padx=(0, 5))
        self.dim_var = tk.IntVar(value=initial_dim)
        dim_spinner = ttk.Spinbox(top_frame, from_=1, to=10, width=5, textvariable=self.dim_var)
        dim_spinner.pack(side=tk.LEFT)
        ttk.Button(top_frame, text="Aplicar", command=self.update_dimension).pack(side=tk.LEFT, padx=(5, 0))
        
        # Frame para las entradas
        self.entries_frame = ttk.Frame(self)
        self.entries_frame.pack(fill=tk.X)
        
        # Crear las entradas iniciales
        self.update_dimension()
    
    def update_dimension(self):
        """Actualiza el número de entradas según la dimensión seleccionada"""
        # Limpiar entradas existentes
        for widget in self.entries_frame.winfo_children():
            widget.destroy()
        
        self.entries = []
        dim = self.dim_var.get()
        
        # Crear nuevas entradas
        for i in range(dim):
            entry_frame = ttk.Frame(self.entries_frame)
            entry_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(entry_frame, text=f"Componente {i+1}:").pack(side=tk.LEFT, padx=(0, 5))
            entry = ttk.Entry(entry_frame, width=10)
            entry.pack(side=tk.LEFT, padx=(0, 5))
            entry.insert(0, "0")
            self.entries.append(entry)
    
    def get_vector(self):
        """Devuelve el vector como una lista de números"""
        try:
            vector = [float(entry.get()) for entry in self.entries]
            return vector
        except ValueError:
            messagebox.showerror("Error", f"Todas las componentes del {self.label_text} deben ser números.")
            return None

class MatrixEntryFrame(ttk.Frame):
    """
    Frame para la entrada de matrices de dimensión variable
    """
    def __init__(self, parent, label_text, initial_rows=2, initial_cols=2):
        super().__init__(parent, padding=10)
        self.parent = parent
        self.label_text = label_text
        self.entries = []
        
        # Frame superior con etiqueta y selector de dimensiones
        top_frame = ttk.Frame(self)
        top_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(top_frame, text=label_text).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(top_frame, text="Filas:").pack(side=tk.LEFT, padx=(0, 5))
        self.rows_var = tk.IntVar(value=initial_rows)
        rows_spinner = ttk.Spinbox(top_frame, from_=1, to=10, width=5, textvariable=self.rows_var)
        rows_spinner.pack(side=tk.LEFT)
        
        ttk.Label(top_frame, text="Columnas:").pack(side=tk.LEFT, padx=(10, 5))
        self.cols_var = tk.IntVar(value=initial_cols)
        cols_spinner = ttk.Spinbox(top_frame, from_=1, to=10, width=5, textvariable=self.cols_var)
        cols_spinner.pack(side=tk.LEFT)
        
        ttk.Button(top_frame, text="Aplicar", command=self.update_dimensions).pack(side=tk.LEFT, padx=(5, 0))
        
        # Frame para las entradas
        self.entries_frame = ttk.Frame(self)
        self.entries_frame.pack(fill=tk.X)
        
        # Crear las entradas iniciales
        self.update_dimensions()
    
    def update_dimensions(self):
        """Actualiza el número de entradas según las dimensiones seleccionadas"""
        # Limpiar entradas existentes
        for widget in self.entries_frame.winfo_children():
            widget.destroy()
        
        self.entries = []
        rows = self.rows_var.get()
        cols = self.cols_var.get()
        
        # Crear nuevas entradas
        for i in range(rows):
            row_entries = []
            row_frame = ttk.Frame(self.entries_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            for j in range(cols):
                entry = ttk.Entry(row_frame, width=6)
                entry.pack(side=tk.LEFT, padx=2)
                entry.insert(0, "0")
                row_entries.append(entry)
            
            self.entries.append(row_entries)
    
    def get_matrix(self):
        """Devuelve la matriz como una lista de listas"""
        try:
            matrix = [[float(entry.get()) for entry in row] for row in self.entries]
            return matrix
        except ValueError:
            messagebox.showerror("Error", f"Todos los elementos de la {self.label_text} deben ser números.")
            return None

class ScalarEntryFrame(ttk.Frame):
    """
    Frame para la entrada de un escalar
    """
    def __init__(self, parent, label_text="Escalar"):
        super().__init__(parent, padding=10)
        self.label_text = label_text
        
        ttk.Label(self, text=label_text).pack(side=tk.LEFT, padx=(0, 5))
        self.entry = ttk.Entry(self, width=10)
        self.entry.pack(side=tk.LEFT)
        self.entry.insert(0, "1")
    
    def get_value(self):
        """Devuelve el valor escalar"""
        try:
            return float(self.entry.get())
        except ValueError:
            messagebox.showerror("Error", f"El {self.label_text} debe ser un número.")
            return None

class AlgebraLinealGUI:
    """
    Interfaz gráfica para usar los métodos de la clase AlgebraLineal
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Álgebra Lineal - Leonardo Daniel Aviña Neri")
        self.root.geometry("1200x700")
        self.root.minsize(1000, 600)
        
        # Configurar el estilo
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f5f5f5")
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("TLabel", font=("Arial", 10))
        self.style.configure("Title.TLabel", font=("Arial", 14, "bold"))
        self.style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        
        # Crear el contenedor principal
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Cabecera con nombre y logo
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Agregar logo a la derecha
        try:
            logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
            logo_image = Image.open(logo_path)
            logo_image = logo_image.resize((100, 100))  # Ajustar tamaño
            self.logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = ttk.Label(header_frame, image=self.logo_photo)
            logo_label.pack(side=tk.RIGHT, padx=10)
        except Exception as e:
            print(f"Error al cargar el logo: {e}")
        
        # Título
        title_label = ttk.Label(header_frame, text="Álgebra Lineal - Interfaz Gráfica", style="Title.TLabel")
        title_label.pack(side=tk.LEFT, padx=10)
        author_label = ttk.Label(header_frame, text="Leonardo Daniel Aviña Neri")
        author_label.pack(side=tk.LEFT, padx=10)
        
        # Paneles principales
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo (lista de métodos)
        left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(left_frame, weight=1)
        
        # Título de operaciones
        ttk.Label(left_frame, text="Operaciones Disponibles", style="Header.TLabel").pack(fill=tk.X, pady=(0, 5))
        
        # Crear notebook para categorías
        self.category_notebook = ttk.Notebook(left_frame)
        self.category_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Categorías de métodos
        categories = {
            "Vectores": [
                "producto_escalar", "producto_vectorial", "suma_vectores", "resta_vectores",
                "escalar_por_vector", "norma", "normalizar", "angulo_entre_vectores", "proyeccion"
            ],
            "Matrices": [
                "crear_matriz", "crear_matriz_identidad", "suma_matrices", "resta_matrices",
                "escalar_por_matriz", "mult_matrices", "transpuesta", "submatriz", "determinante", "inversa"
            ],
            "Sistemas": [
                "gauss_jordan", "gauss", "resolver_sistema"
            ],
            "Análisis Lineal": [
                "calcular_rango", "es_linealmente_independiente", "combinacion_lineal", "es_combinacion_lineal"
            ],            "Visualización": [
                "graficar_funcion", "graficar_vectores"
            ]
        }
        
        # Crear tabs para cada categoría
        self.category_frames = {}
        for category, methods in categories.items():
            frame = ttk.Frame(self.category_notebook)
            self.category_frames[category] = frame
            self.category_notebook.add(frame, text=category)
            
            # Crear botones para cada método
            for method in methods:
                method_button = ttk.Button(
                    frame, 
                    text=method.replace("_", " ").title(),
                    command=lambda m=method: self.show_method_interface(m)
                )
                method_button.pack(fill=tk.X, pady=2)
        
        # Panel derecho (interfaces de método y resultados)
        right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(right_frame, weight=3)
        
        # Panel superior para la interfaz del método
        self.method_frame = ttk.Frame(right_frame, padding=10)
        self.method_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Dividir área de resultados
        results_paned = ttk.PanedWindow(right_frame, orient=tk.HORIZONTAL)
        results_paned.pack(fill=tk.BOTH, expand=True)
        
        # Área de resultados
        result_frame = ttk.Frame(results_paned, padding=10)
        results_paned.add(result_frame, weight=1)
        
        ttk.Label(result_frame, text="Resultado", style="Header.TLabel").pack(fill=tk.X)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=10)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.configure(state="disabled")
        
        # Área de salida de consola
        console_frame = ttk.Frame(results_paned, padding=10)
        results_paned.add(console_frame, weight=1)
        
        ttk.Label(console_frame, text="Mensajes de salida", style="Header.TLabel").pack(fill=tk.X)
        
        self.console_text = scrolledtext.ScrolledText(console_frame, height=10)
        self.console_text.pack(fill=tk.BOTH, expand=True)
        self.console_text.configure(state="disabled")
        
        # Redirigir la salida estándar al widget de consola
        self.stdout_redirector = RedirectOutput(self.console_text)
        self.original_stdout = sys.stdout
        sys.stdout = self.stdout_redirector
        
        # Variable para almacenar el frame actual del método
        self.current_method_frame = None
        
        # Mostrar mensaje de bienvenida
        self.display_result("Bienvenido a la Interfaz de Álgebra Lineal\n\n"
                           "Seleccione una operación del panel izquierdo para comenzar.")
    
    def display_result(self, text):
        """Muestra un resultado en el área de resultados"""
        self.result_text.configure(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state="disabled")
    
    def clear_console(self):
        """Limpia el área de consola"""
        self.console_text.configure(state="normal")
        self.console_text.delete(1.0, tk.END)
        self.console_text.configure(state="disabled")
    
    def show_method_interface(self, method_name):
        """Muestra la interfaz para el método seleccionado"""
        # Limpiar el frame de método actual
        if self.current_method_frame:
            self.current_method_frame.destroy()
        
        # Crear un nuevo frame
        self.current_method_frame = ttk.Frame(self.method_frame, padding=10)
        self.current_method_frame.pack(fill=tk.BOTH, expand=True)
        
        # Obtener la información del método seleccionado
        method = getattr(AlgebraLineal, method_name)
        signature = inspect.signature(method)
        doc = method.__doc__ or "No hay documentación disponible para este método."
        
        # Título del método
        method_title = ttk.Label(
            self.current_method_frame, 
            text=method_name.replace("_", " ").title(),
            style="Header.TLabel"
        )
        method_title.pack(fill=tk.X, pady=(0, 5))
        
        # Descripción del método
        description_frame = ttk.Frame(self.current_method_frame)
        description_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Extraer la primera parte del docstring como descripción breve
        description = doc.split("\n\n")[0] if "\n\n" in doc else doc
        description_text = scrolledtext.ScrolledText(description_frame, height=3, wrap=tk.WORD)
        description_text.pack(fill=tk.X)
        description_text.insert(tk.END, description)
        description_text.configure(state="disabled")
        
        # Frame para los parámetros
        params_frame = ttk.Frame(self.current_method_frame)
        params_frame.pack(fill=tk.BOTH, expand=True)
        
        # Crear los widgets para cada parámetro
        param_widgets = {}
        
        # Gestionar cada método específico
        if method_name == "producto_escalar":
            param_widgets["v1"] = VectorEntryFrame(params_frame, "Vector 1", initial_dim=3)
            param_widgets["v1"].pack(fill=tk.X)
            param_widgets["v2"] = VectorEntryFrame(params_frame, "Vector 2", initial_dim=3)
            param_widgets["v2"].pack(fill=tk.X)
        
        elif method_name == "producto_vectorial":
            param_widgets["v1"] = VectorEntryFrame(params_frame, "Vector 1", initial_dim=3)
            param_widgets["v1"].pack(fill=tk.X)
            param_widgets["v2"] = VectorEntryFrame(params_frame, "Vector 2", initial_dim=3)
            param_widgets["v2"].pack(fill=tk.X)
        
        elif method_name == "suma_vectores" or method_name == "resta_vectores":
            param_widgets["v1"] = VectorEntryFrame(params_frame, "Vector 1", initial_dim=3)
            param_widgets["v1"].pack(fill=tk.X)
            param_widgets["v2"] = VectorEntryFrame(params_frame, "Vector 2", initial_dim=3)
            param_widgets["v2"].pack(fill=tk.X)
        
        elif method_name == "escalar_por_vector":
            param_widgets["escalar"] = ScalarEntryFrame(params_frame, "Escalar")
            param_widgets["escalar"].pack(fill=tk.X)
            param_widgets["vector"] = VectorEntryFrame(params_frame, "Vector", initial_dim=3)
            param_widgets["vector"].pack(fill=tk.X)
        
        elif method_name == "norma" or method_name == "normalizar":
            param_widgets["vector"] = VectorEntryFrame(params_frame, "Vector", initial_dim=3)
            param_widgets["vector"].pack(fill=tk.X)
        
        elif method_name == "angulo_entre_vectores" or method_name == "proyeccion":
            param_widgets["v1"] = VectorEntryFrame(params_frame, "Vector 1", initial_dim=3)
            param_widgets["v1"].pack(fill=tk.X)
            param_widgets["v2"] = VectorEntryFrame(params_frame, "Vector 2", initial_dim=3)
            param_widgets["v2"].pack(fill=tk.X)
        
        elif method_name == "crear_matriz":
            param_widgets["filas"] = ScalarEntryFrame(params_frame, "Filas")
            param_widgets["filas"].pack(fill=tk.X, pady=5)
            param_widgets["columnas"] = ScalarEntryFrame(params_frame, "Columnas")
            param_widgets["columnas"].pack(fill=tk.X, pady=5)
            param_widgets["valor_inicial"] = ScalarEntryFrame(params_frame, "Valor Inicial")
            param_widgets["valor_inicial"].pack(fill=tk.X, pady=5)
        
        elif method_name == "crear_matriz_identidad":
            param_widgets["n"] = ScalarEntryFrame(params_frame, "Tamaño (n)")
            param_widgets["n"].pack(fill=tk.X, pady=5)
        
        elif method_name == "suma_matrices" or method_name == "resta_matrices" or method_name == "mult_matrices":
            param_widgets["m1"] = MatrixEntryFrame(params_frame, "Matriz 1")
            param_widgets["m1"].pack(fill=tk.X, pady=5)
            param_widgets["m2"] = MatrixEntryFrame(params_frame, "Matriz 2")
            param_widgets["m2"].pack(fill=tk.X, pady=5)
        
        elif method_name == "escalar_por_matriz":
            param_widgets["escalar"] = ScalarEntryFrame(params_frame, "Escalar")
            param_widgets["escalar"].pack(fill=tk.X, pady=5)
            param_widgets["matriz"] = MatrixEntryFrame(params_frame, "Matriz")
            param_widgets["matriz"].pack(fill=tk.X, pady=5)
        
        elif method_name == "transpuesta":
            param_widgets["matriz"] = MatrixEntryFrame(params_frame, "Matriz")
            param_widgets["matriz"].pack(fill=tk.X, pady=5)
        
        elif method_name == "submatriz":
            param_widgets["matriz"] = MatrixEntryFrame(params_frame, "Matriz")
            param_widgets["matriz"].pack(fill=tk.X, pady=5)
            param_widgets["fila_excluida"] = ScalarEntryFrame(params_frame, "Fila a excluir (0-based)")
            param_widgets["fila_excluida"].pack(fill=tk.X, pady=5)
            param_widgets["columna_excluida"] = ScalarEntryFrame(params_frame, "Columna a excluir (0-based)")
            param_widgets["columna_excluida"].pack(fill=tk.X, pady=5)
        
        elif method_name == "determinante" or method_name == "inversa":
            param_widgets["matriz"] = MatrixEntryFrame(params_frame, "Matriz")
            param_widgets["matriz"].pack(fill=tk.X, pady=5)
        
        elif method_name == "gauss_jordan" or method_name == "gauss":
            param_widgets["matriz_aumentada"] = MatrixEntryFrame(params_frame, "Matriz Aumentada [A|b]")
            param_widgets["matriz_aumentada"].pack(fill=tk.X, pady=5)
            
            if method_name == "gauss":
                verbose_frame = ttk.Frame(params_frame)
                verbose_frame.pack(fill=tk.X, pady=5)
                param_widgets["verbose"] = tk.BooleanVar(value=True)
                verbose_check = ttk.Checkbutton(
                    verbose_frame, 
                    text="Mostrar detalles", 
                    variable=param_widgets["verbose"]
                )
                verbose_check.pack(side=tk.LEFT)
        
        elif method_name == "resolver_sistema":
            param_widgets["coeficientes"] = MatrixEntryFrame(params_frame, "Matriz de Coeficientes")
            param_widgets["coeficientes"].pack(fill=tk.X, pady=5)
            param_widgets["terminos_independientes"] = VectorEntryFrame(params_frame, "Términos Independientes")
            param_widgets["terminos_independientes"].pack(fill=tk.X, pady=5)
        
        elif method_name == "calcular_rango":
            param_widgets["matriz"] = MatrixEntryFrame(params_frame, "Matriz")
            param_widgets["matriz"].pack(fill=tk.X, pady=5)
        
        elif method_name == "es_linealmente_independiente":
            # Crear un widget especial para matrices donde cada columna es un vector
            param_widgets["vectores"] = MatrixEntryFrame(params_frame, "Vectores (cada columna es un vector)")
            param_widgets["vectores"].pack(fill=tk.X, pady=5)
        
        elif method_name == "combinacion_lineal":
            param_widgets["vectores"] = MatrixEntryFrame(params_frame, "Vectores (cada columna es un vector)")
            param_widgets["vectores"].pack(fill=tk.X, pady=5)
            param_widgets["coeficientes"] = VectorEntryFrame(params_frame, "Coeficientes")
            param_widgets["coeficientes"].pack(fill=tk.X, pady=5)
        
        elif method_name == "es_combinacion_lineal":
            param_widgets["vector"] = VectorEntryFrame(params_frame, "Vector a comprobar")
            param_widgets["vector"].pack(fill=tk.X, pady=5)
            param_widgets["conjunto_vectores"] = MatrixEntryFrame(params_frame, "Conjunto de Vectores (cada columna es un vector)")
            param_widgets["conjunto_vectores"].pack(fill=tk.X, pady=5)
        
        elif method_name == "graficar_funcion":
            # Para la función, creamos un frame especial
            function_frame = ttk.Frame(params_frame)
            function_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(function_frame, text="Función (ejemplo: 'x**2'):").pack(side=tk.LEFT, padx=(0, 5))
            function_entry = ttk.Entry(function_frame, width=30)
            function_entry.pack(side=tk.LEFT)
            function_entry.insert(0, "x**2")
            param_widgets["f"] = function_entry
            
            param_widgets["x_min"] = ScalarEntryFrame(params_frame, "X mínimo")
            param_widgets["x_min"].pack(fill=tk.X, pady=5)
            param_widgets["x_min"].entry.delete(0, tk.END)
            param_widgets["x_min"].entry.insert(0, "-5")
            
            param_widgets["x_max"] = ScalarEntryFrame(params_frame, "X máximo")
            param_widgets["x_max"].pack(fill=tk.X, pady=5)
            param_widgets["x_max"].entry.delete(0, tk.END)
            param_widgets["x_max"].entry.insert(0, "5")
            
            param_widgets["puntos"] = ScalarEntryFrame(params_frame, "Número de puntos")
            param_widgets["puntos"].pack(fill=tk.X, pady=5)
            param_widgets["puntos"].entry.delete(0, tk.END)
            param_widgets["puntos"].entry.insert(0, "100")
            
            param_widgets["titulo"] = ttk.Entry(params_frame)
            titulo_frame = ttk.Frame(params_frame)
            titulo_frame.pack(fill=tk.X, pady=5)
            ttk.Label(titulo_frame, text="Título:").pack(side=tk.LEFT, padx=(0, 5))
            param_widgets["titulo"] = ttk.Entry(titulo_frame, width=30)
            param_widgets["titulo"].pack(side=tk.LEFT)
            param_widgets["titulo"].insert(0, "Gráfica de función")
            
            # Etiquetas X e Y
            etiqueta_x_frame = ttk.Frame(params_frame)
            etiqueta_x_frame.pack(fill=tk.X, pady=5)
            ttk.Label(etiqueta_x_frame, text="Etiqueta X:").pack(side=tk.LEFT, padx=(0, 5))
            param_widgets["etiqueta_x"] = ttk.Entry(etiqueta_x_frame, width=20)
            param_widgets["etiqueta_x"].pack(side=tk.LEFT)
            param_widgets["etiqueta_x"].insert(0, "x")
            
            etiqueta_y_frame = ttk.Frame(params_frame)
            etiqueta_y_frame.pack(fill=tk.X, pady=5)
            ttk.Label(etiqueta_y_frame, text="Etiqueta Y:").pack(side=tk.LEFT, padx=(0, 5))
            param_widgets["etiqueta_y"] = ttk.Entry(etiqueta_y_frame, width=20)
            param_widgets["etiqueta_y"].pack(side=tk.LEFT)
            param_widgets["etiqueta_y"].insert(0, "f(x)")
            
            # Checkbox para mostrar puntos destacados
            mostrar_puntos_frame = ttk.Frame(params_frame)
            mostrar_puntos_frame.pack(fill=tk.X, pady=5)
            param_widgets["mostrar_puntos_destacados"] = tk.BooleanVar(value=True)
            mostrar_puntos_check = ttk.Checkbutton(
                mostrar_puntos_frame, 
                text="Mostrar puntos destacados", 
                variable=param_widgets["mostrar_puntos_destacados"]
            )
            mostrar_puntos_check.pack(side=tk.LEFT)
        
        elif method_name == "graficar_vectores":
            param_widgets["vectores"] = MatrixEntryFrame(params_frame, "Vectores (cada fila es un vector)")
            param_widgets["vectores"].pack(fill=tk.X, pady=5)
            
            # Etiquetas
            etiquetas_frame = ttk.Frame(params_frame)
            etiquetas_frame.pack(fill=tk.X, pady=5)
            ttk.Label(etiquetas_frame, text="Etiquetas (separadas por comas):").pack(side=tk.LEFT, padx=(0, 5))
            param_widgets["etiquetas"] = ttk.Entry(etiquetas_frame, width=30)
            param_widgets["etiquetas"].pack(side=tk.LEFT)
            param_widgets["etiquetas"].insert(0, "v1,v2,v3")
            
            # Origen
            origen_frame = ttk.Frame(params_frame)
            origen_frame.pack(fill=tk.X, pady=5)
            ttk.Label(origen_frame, text="Origen (separado por comas):").pack(side=tk.LEFT, padx=(0, 5))
            param_widgets["origen"] = ttk.Entry(origen_frame, width=30)
            param_widgets["origen"].pack(side=tk.LEFT)
            param_widgets["origen"].insert(0, "0,0,0")
            
            # Título
            titulo_frame = ttk.Frame(params_frame)
            titulo_frame.pack(fill=tk.X, pady=5)
            ttk.Label(titulo_frame, text="Título:").pack(side=tk.LEFT, padx=(0, 5))
            param_widgets["titulo"] = ttk.Entry(titulo_frame, width=30)
            param_widgets["titulo"].pack(side=tk.LEFT)
            param_widgets["titulo"].insert(0, "Vectores")
            
            # Mostrar ejes
            mostrar_ejes_frame = ttk.Frame(params_frame)
            mostrar_ejes_frame.pack(fill=tk.X, pady=5)
            param_widgets["mostrar_ejes"] = tk.BooleanVar(value=True)
            mostrar_ejes_check = ttk.Checkbutton(
                mostrar_ejes_frame, 
                text="Mostrar ejes", 
                variable=param_widgets["mostrar_ejes"]
            )
            mostrar_ejes_check.pack(side=tk.LEFT)
            
        
        # Botón para ejecutar el método
        execute_button = ttk.Button(
            self.current_method_frame,
            text="Ejecutar",
            command=lambda: self.execute_method(method_name, param_widgets)
        )
        execute_button.pack(pady=10)
    
    def execute_method(self, method_name, param_widgets):
        """Ejecuta el método seleccionado con los parámetros proporcionados"""
        try:
            # Limpiar la consola
            self.clear_console()
            
            # Obtener los valores de los parámetros según el método
            args = {}
            
            # Gestionar cada método específico
            if method_name in ["producto_escalar", "producto_vectorial", "suma_vectores", "resta_vectores"]:
                args["v1"] = param_widgets["v1"].get_vector()
                args["v2"] = param_widgets["v2"].get_vector()
            
            elif method_name == "escalar_por_vector":
                args["escalar"] = param_widgets["escalar"].get_value()
                args["vector"] = param_widgets["vector"].get_vector()
            
            elif method_name in ["norma", "normalizar"]:
                args["vector"] = param_widgets["vector"].get_vector()
            
            elif method_name in ["angulo_entre_vectores", "proyeccion"]:
                args["v1"] = param_widgets["v1"].get_vector()
                args["v2"] = param_widgets["v2"].get_vector()
            
            elif method_name == "crear_matriz":
                args["filas"] = int(param_widgets["filas"].get_value())
                args["columnas"] = int(param_widgets["columnas"].get_value())
                args["valor_inicial"] = param_widgets["valor_inicial"].get_value()
            
            elif method_name == "crear_matriz_identidad":
                args["n"] = int(param_widgets["n"].get_value())
            
            elif method_name in ["suma_matrices", "resta_matrices", "mult_matrices"]:
                args["m1"] = param_widgets["m1"].get_matrix()
                args["m2"] = param_widgets["m2"].get_matrix()
            
            elif method_name == "escalar_por_matriz":
                args["escalar"] = param_widgets["escalar"].get_value()
                args["matriz"] = param_widgets["matriz"].get_matrix()
            
            elif method_name in ["transpuesta", "determinante", "inversa"]:
                args["matriz"] = param_widgets["matriz"].get_matrix()
            
            elif method_name == "submatriz":
                args["matriz"] = param_widgets["matriz"].get_matrix()
                args["fila_excluida"] = int(param_widgets["fila_excluida"].get_value())
                args["columna_excluida"] = int(param_widgets["columna_excluida"].get_value())
            
            elif method_name == "gauss_jordan":
                args["matriz_aumentada"] = param_widgets["matriz_aumentada"].get_matrix()
            
            elif method_name == "gauss":
                args["matriz_aumentada"] = param_widgets["matriz_aumentada"].get_matrix()
                args["verbose"] = param_widgets["verbose"].get()
            
            elif method_name == "resolver_sistema":
                args["coeficientes"] = param_widgets["coeficientes"].get_matrix()
                args["terminos_independientes"] = param_widgets["terminos_independientes"].get_vector()
            
            elif method_name == "calcular_rango":
                args["matriz"] = param_widgets["matriz"].get_matrix()
            
            elif method_name == "es_linealmente_independiente":
                # Transponer la matriz para obtener los vectores como filas
                matriz = param_widgets["vectores"].get_matrix()
                args["vectores"] = [list(x) for x in zip(*matriz)]
            
            elif method_name == "combinacion_lineal":
                matriz = param_widgets["vectores"].get_matrix()
                args["vectores"] = [list(x) for x in zip(*matriz)]
                args["coeficientes"] = param_widgets["coeficientes"].get_vector()
            
            elif method_name == "es_combinacion_lineal":
                args["vector"] = param_widgets["vector"].get_vector()
                matriz = param_widgets["conjunto_vectores"].get_matrix()
                args["conjunto_vectores"] = [list(x) for x in zip(*matriz)]
            
            elif method_name == "graficar_funcion":
                # Crear una función a partir de la cadena
                func_str = param_widgets["f"].get()
                args["f"] = lambda x: eval(func_str)
                args["x_min"] = param_widgets["x_min"].get_value()
                args["x_max"] = param_widgets["x_max"].get_value()
                args["puntos"] = int(param_widgets["puntos"].get_value())
                args["titulo"] = param_widgets["titulo"].get()
                args["etiqueta_x"] = param_widgets["etiqueta_x"].get()
                args["etiqueta_y"] = param_widgets["etiqueta_y"].get()
                args["mostrar_puntos_destacados"] = param_widgets["mostrar_puntos_destacados"].get()
            
            elif method_name == "graficar_vectores":
                args["vectores"] = param_widgets["vectores"].get_matrix()
                
                # Procesar etiquetas
                etiquetas_str = param_widgets["etiquetas"].get()
                if etiquetas_str.strip():
                    args["etiquetas"] = etiquetas_str.split(",")
                else:
                    args["etiquetas"] = None
                
                # Procesar origen
                origen_str = param_widgets["origen"].get()
                if origen_str.strip():
                    args["origen"] = [float(x) for x in origen_str.split(",")]
                else:
                    args["origen"] = None
                
                args["titulo"] = param_widgets["titulo"].get()
                args["mostrar_ejes"] = param_widgets["mostrar_ejes"].get()
            
            # Verificar que todos los argumentos necesarios están presentes
            if None in args.values():
                messagebox.showerror("Error", "Por favor complete todos los campos correctamente.")
                return
            
            # Ejecutar el método
            method = getattr(AlgebraLineal, method_name)
            result = method(**args)
            
            # Mostrar el resultado
            if result is not None:
                if method_name in ["es_linealmente_independiente", "es_combinacion_lineal"]:
                    # Estos métodos devuelven una tupla (booleano, explicación)
                    is_valid, explanation = result
                    result_str = f"Resultado: {'Sí' if is_valid else 'No'}\n\nExplicación: {explanation}"
                    self.display_result(result_str)
                elif method_name in ["gauss_jordan", "gauss", "resolver_sistema"]:
                    # Estos métodos devuelven una tupla (solución, tipo)
                    solucion, tipo = result
                    if tipo == "unica":
                        result_str = "Solución única:\n"
                        for i, val in enumerate(solucion):
                            result_str += f"x{i+1} = {val}\n"
                    elif tipo == "infinitas":
                        result_str = "El sistema tiene infinitas soluciones."
                    else:
                        result_str = "El sistema no tiene solución."
                    self.display_result(result_str)
                else:
                    # Para otros métodos, mostrar el resultado directamente
                    if isinstance(result, list):
                        if isinstance(result[0], list):
                            # Es una matriz
                            result_str = "Resultado:\n"
                            for row in result:
                                result_str += str(row) + "\n"
                        else:
                            # Es un vector
                            result_str = f"Resultado: {result}"
                    else:
                        result_str = f"Resultado: {result}"
                    
                    self.display_result(result_str)
            else:
                # Algunos métodos como graficar_* no devuelven nada
                self.display_result("Operación completada con éxito.")
        
        except Exception as e:
            self.display_result(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def __del__(self):
        """Restaurar stdout original al cerrar la aplicación"""
        if hasattr(self, 'original_stdout'):
            sys.stdout = self.original_stdout

def main():
    """Función principal para iniciar la aplicación"""
    root = tk.Tk()
    app = AlgebraLinealGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()