import sympy
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as iw
import macti.vis as mvis

# TODO: Bug in matlplotlib with fonts. With this the warning is avoided
plt.rcParams['font.weight'] = 'normal'

class QuadraticFunc():

    def __init__(self, A, b, xlim, ylim, sizegrid, vecscale = 23000):
        # Malla para gráficos
        self.__xg, self.__yg = np.meshgrid(np.linspace(xlim[0], xlim[1], sizegrid), 
                                           np.linspace(ylim[0], ylim[1], sizegrid))

        # Dominio de evaluación
        self.__x = np.linspace(xlim[0], xlim[1], 11)

        # Limites para las gráficas
        xoff = np.fabs(xlim[1] - xlim[0]) * 0.1
        self.__xlim = (xlim[0] - xoff, xlim[1] + xoff)
        
        yoff = np.fabs(ylim[1] - ylim[0]) * 0.1
        self.__ylim = (ylim[0] - yoff, ylim[1] + yoff)

        # Calculamos las líneas rectas
        self.__y1, self.__y2 = self.__calc_lines(A, b, self.__x)

        # Resolvemos el sistema lineal
        self.__sol = np.linalg.solve(A, b)

        # Calculamos la función cuadrática en una malla
        self.__z = self.__calc_surface(A, b, sizegrid, self.__xg, self.__yg)

        # Calculamos el campo vectorial del gradiente
        self.__U, self.__V = self.__calc_gradient(A, b, self.__xg, self.__yg)

        # Calculamos los eigenvectores
        __, self.__vA = np.linalg.eig(A)
        
        self.__vecscale = vecscale
        
    def __calc_lines(self, A, b, x):
        # Línea recta 1
        m1 = -A[0,0] / A[0,1]
        b1 = b[0] / A[0,1]
        y1 = m1 * x + b1 
        
        # Línea recta 2
        m2 = -A[1,0] / A[1,1]
        b2 = b[1] / A[1,1]
        y2 = m2 * x + b2 

        return y1, y2

    def __calc_gradient(self, A, b, xg, yg):
        xs, ys = sympy.symbols("x y")
        X = sympy.Matrix([xs, ys])

        # Función cuadrática
        f = 0.5 * X.T @ A @ X - X.T @ b

        # Cálculo del gradiente 
        Df = sympy.Matrix(f).jacobian(X)
    
        # Transformación a arreglos numpy
        f1 = sympy.lambdify(X, Df[0])
        f2 = sympy.lambdify(X, Df[1])

        # Cálculo del campo vectorial
        U=[f1(x1, y1) for x1,y1 in zip(xg, yg)]
        V=[f2(x1, y1) for x1,y1 in zip(xg, yg)]

        return U, V

    def __calc_surface(self, A, b, sizegrid, xg, yg):
        # Arreglo para almacenar los valores de la función cuadrática
        z = np.zeros((sizegrid, sizegrid))

        # Función cuadrática
        f = lambda A, b, x: 0.5 * x @ A @ x.T - x @ b.T
    
        # Cálculo de la función cuadrática
        for i in range(sizegrid):
            for j in range(sizegrid):
                xc = np.array([xg[i,j], yg[i,j]])
                z[i,j] = f(A, b, xc)
                
        return z        

    def plot_sol(self):
        x = self.__x
        y1 = self.__y1
        y2 = self.__y2
        sol = self.__sol
        
        # Definición de los parámetros de la figura
        fig_par = {}# "figsize":(10, 5)

        # Definición de los parámetros de las gráficas (2)
        axis_par = [
            {"aspect":"auto"}
        ]

        # Definición del espacio de trabajo
        v = mvis.Plotter(1, 1, axis_par, fig_par, title="")

        # --- GRAFICA 1 ---
        # Definimos el sistema de coordenadas
        v.set_coordsys(1)

        # Graficamos las líneas
        v.plot(1, x, y1, lw = 3, c = "blue", label = "Ecuación (1)") # Línea recta 1
        v.plot(1, x, y2, lw = 3, c = "red", label = "Ecuación (2)") # Línea recta 2
        
        # Graficamos la solución
        v.scatter(1, sol[0], sol[1], fc="sandybrown", ec="k", s = 75, alpha=0.75, zorder=5, label="Solución final.")

        v.axes(1).set_xlim(self.__xlim)
        v.axes(1).set_ylim(self.__ylim)
        v.legend([1], ncol = 1, frameon=True, loc="upper right", fontsize=8)
        v.grid()
        v.show()
            
    def plot_qf(self, azim = 60, solution = False, contour = False, gradient = False, eigen = False):

        # Nombres de variables para un uso más simple
        xg = self.__xg
        yg = self.__yg
        x  = self.__x
        y1, y2 = self.__y1, self.__y2
        sol = self.__sol
        z = self.__z
        U, V = self.__U, self.__V
        vA = self.__vA

        # Definición de los parámetros de la figura
        fig_par = {
            "figsize":(10, 5)
        }

        # Definición de los parámetros de las gráficas (2)
        axis_par = [
            {"projection":"3d", "aspect":"equal", "xlabel":"$x$", "ylabel":"$y$", "zlabel":""},
            {"aspect":"equal"}
        ]

        # Definición del espacio de trabajo
        v = mvis.Plotter(1, 2, axis_par, fig_par, title="Función cuadrática $f(x)$ y sistema lineal")

        # --- GRAFICA 1 ---
        # Graficamos la función cuadrática en 3D
        v.plot_surface(1, xg, yg, z, cmap="Spectral_r", alpha=0.75, lw=0.1,ec="k") # f(x)
        v.axes(1).view_init(azim = azim)

        # --- GRAFICA 2 ---
        # Definimos el sistema de coordenadas
        v.set_coordsys(2)

        # Graficamos las líneas
        v.plot(2, x, y1, lw = 3, c = "blue", label = "Ecuación (1)") # Línea recta 1
        v.plot(2, x, y2, lw = 3, c = "red", label = "Ecuación (2)") # Línea recta 2

        if solution:
            # Graficamos la solución
            v.scatter(2, sol[0], sol[1], fc="sandybrown", ec="k", s = 75, alpha=0.75, zorder=5, label="Solución final.")
            
        if contour:
            # Graficamos los contornos
            v.contour(2, xg, yg, z, levels = 25, cmap="Spectral_r", linewidths=1.5, zorder=1) 

        if gradient:
            # Graficamos los vectores del gradiente
            v.quiver(2, xg, yg, U, V, color="black",scale = self.__vecscale)

        if eigen:
            # Graficamos los eigenvectores
            q = v.quiver(2, [sol[0], sol[0]], [sol[1], sol[1]], vA[0], vA[1], scale = 5, zorder = 10)
            v.axes(2).quiverkey(q, X=0.35, Y=1.05, U=1, label="Eigenvectors", labelpos="E")
            
        v.axes(2).set_xlim(self.__xlim)
        v.axes(2).set_ylim(self.__ylim)
        v.legend([2], ncol = 1, frameon=True, loc="upper right", fontsize=8)
        v.grid()
        v.show()

    def interactive_qf(self):
        azim = iw.IntSlider(min=0, max = 180, value = 60, step = 10, description="Azimut", layout=iw.Layout(width="400px"))
        solution = iw.Checkbox(value=False, description="Solución", disabled=False, indent=True, layout=iw.Layout(width="180px"))
        contour = iw.Checkbox(value=False, description="Contornos", disabled=False, indent=True, layout=iw.Layout(width="180px"))
        gradient = iw.Checkbox(value=False, description="Gradiente", disabled=False, indent=True, layout=iw.Layout(width="180px"))
        eigen = iw.Checkbox(value=False, description="Eigenvectors", disabled=False, indent=True, layout=iw.Layout(width="190px"))
        
        ui = iw.VBox((iw.HBox((solution, contour)),
                      iw.HBox((eigen, gradient)),
                      iw.HBox((azim, ))),
                     layout=iw.Layout(align_items = "flex-end"))
        ui.layout = iw.Layout(border="solid 1px gray")
        ui.layout.width = "450px"

        par = {"azim":azim, "solution": solution, 
               "contour": contour, "gradient": gradient, "eigen": eigen}
        
        out = iw.interactive_output(self.plot_qf, par)

        display(ui, out)   

    def set_steps(self, xs, ys):
        self.__xs = xs
        self.__ys = ys
        
    def plot_steps(self, steps = False, ini_sol = False, fin_sol = False, contour = False, gradient = False):

        # Nombres de variables para un uso más simple
        xs = self.__xs
        ys = self.__ys
        xg = self.__xg
        yg = self.__yg
        x  = self.__x
        y1, y2 = self.__y1, self.__y2
        sol = self.__sol
        z = self.__z
        U, V = self.__U, self.__V
        vA = self.__vA

        # Definición de los parámetros de la figura
        fig_par = {}# "figsize":(10, 5)

        # Definición de los parámetros de las gráficas (2)
        axis_par = [
            {"aspect":"auto"}
        ]

        # Definición del espacio de trabajo
        v = mvis.Plotter(1, 1, axis_par, fig_par, title="Pasos hacia la solución")

        # --- GRAFICA 1 ---
        # Definimos el sistema de coordenadas
        v.set_coordsys(1)

        # Graficamos las líneas
        v.plot(1, x, y1, lw = 3, c = "blue", label = "Ecuación (1)") # Línea recta 1
        v.plot(1, x, y2, lw = 3, c = "red", label = "Ecuación (2)") # Línea recta 2

        
        if fin_sol:
            # Graficamos la solución
            v.scatter(1, sol[0], sol[1], fc="sandybrown", ec="k", s = 75, alpha=0.75, zorder=5, label="Solución final.")
            
        if contour:
            # Graficamos los contornos
            v.contour(1, xg, yg, z, levels = 25, cmap="Spectral_r", linewidths=1.5, zorder=1) 

        if gradient:
            # Graficamos los vectores del gradiente
            v.quiver(1, xg, yg, U, V, color="dimgray",scale = self.__vecscale)

        if ini_sol:
            v.scatter(1, xs[0], ys[0], fc="yellow", ec="k", s = 75, alpha=0.75, zorder=8, label="Solución inicial")
            
        if steps:
            v.scatter(1, xs[1:], ys[1:], c="navy", s = 10, alpha=0.5, zorder=8)
            v.plot(1, xs, ys, c="black", marker = ".", ls = "--", lw=1.0, zorder=7, label="Pasos del método")
            
        v.axes(1).set_xlim(self.__xlim)
        v.axes(1).set_ylim(self.__ylim)
        v.legend([1], ncol = 1, frameon=True, loc="upper right", fontsize=8)
        v.grid()
        v.show()

    def interactive_steps(self):
        steps = iw.Checkbox(value=False, description="Pasos", disabled=False, indent=True, layout=iw.Layout(width="160px"))
        ini_sol = iw.Checkbox(value=False, description="Sol. ini.", disabled=False, indent=True, layout=iw.Layout(width="170px"))
        fin_sol = iw.Checkbox(value=False, description="Sol. fin", disabled=False, indent=True, layout=iw.Layout(width="170px"))
        contour = iw.Checkbox(value=False, description="Contornos", disabled=False, indent=True, layout=iw.Layout(width="180px"))
        gradient = iw.Checkbox(value=False, description="Gradiente", disabled=False, indent=True, layout=iw.Layout(width="180px"))
        
        ui = iw.VBox((iw.HBox((ini_sol, fin_sol, steps)),
                      iw.HBox((contour, gradient))),
                      layout=iw.Layout(align_items = "flex-end"))
        ui.layout = iw.Layout(border="solid 1px gray")
        ui.layout.width = "500px"

        par = {"steps":steps, "ini_sol": ini_sol, "fin_sol": fin_sol,
               "contour": contour, "gradient": gradient}
        
        out = iw.interactive_output(self.plot_steps, par)

        display(ui, out)  
        
    def __repr__(self):
        lines = []
        lines.append(f"QuadraticFunc(sol={self.__sol})")
        output = " --- \n"
        for l in lines:
            output += l + "\n"
        return output

if __name__ == "__main__" :
    import dfsol
    # Definimos los parámetros físicos del problema
    k = 1000
    L = 0.5
    TA = 500
    TB = 100
    q = 5e+6
    N = 2
    
    A, b = dfsol.sistema_lineal(TA, TB, q, L, k, N)
    
    f = QuadraticFunc(A, b, (100, 800), (-400, 1000), 40)

    sol = int(input("¿Solución? "))
    contour = int(input("¿Contornos? "))
    gradient = int(input("¿Gradiente? "))
    eigen = int(input("¿Eigenvectors? "))
    f.plot(azim= 60, solution = sol, contour = contour, gradient = gradient, eigen = eigen)