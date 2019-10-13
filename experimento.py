
'''
quiero hacer una clase que tenga metodos utiles para mediciones
metodos
   pasarle un directorio y que me devuelva archivos
       # archivos = med.dir("directorio")
   # pasarle un archivo y que devuelva array
       # array = med.arr(archivo)
       # que ese array sea de instancias de la clase variables
   # asociarle errores
       # necesito que cada variable de una medicion exista por si misma
       # X.error(error)
   # grficarlo

# experimento
   # medicion
       # variables, array
# '''
import matplotlib.pyplot as plt
from uncertainties import unumpy as un
import numpy as np
import os

class experimento():
    "una clase para agilizar la carga de datos de labo"
    def __init__(self, dire, claves=[]):
        if os.path.isabs(dire):
            self.absdire = dire
        else:
            self.absdire = os.path.abspath(dire)
        self.archivos = sorted(os.listdir(self.absdire))
        if claves!=[]:
            self.archivos = [i for i in self.archivos if all(clave in i for clave in claves)]
        else:
            pass
    def cargar(self,archivo):
        arch = os.path.join(self.absdire, archivo)
        return np.loadtxt(arch, delimiter=';', unpack = True, comments='$')
    @staticmethod
    def titular(archivo):
        return archivo.split('.')[0].replace('_',' ')
    @staticmethod
    def set_e(var:np.ndarray, error:float) -> np.ndarray:
        error = np.ones(np.shape(var))*error
        var = un.uarray(var, error)
        return var
    @staticmethod
    def get_e(var:np.ndarray):
        assert var.ndim == 1
        return un.std_devs(var)
    @staticmethod
    def get_v(var:np.ndarray):
        assert var.ndim == 1
        return un.nominal_values(var)
    def plotear(self,x, var, fill=True, alpha=0.5, label=None, orden=None):
        xvals = self.get_v(x)
        xerr = self.get_e(x)
        varvals = self.get_v(var)
        varerr = self.get_e(var)
        plt.plot(xvals, varvals, '.', label=label)
        if fill:
            varmenos = varvals - varerr
            varmas = varvals + varerr
            plt.fill_between(xvals, varmenos, varmas, alpha=alpha)
        else:
            plt.errorbar(xvals, varvals, xerr=xerr, yerr=varerr, fmt='.')
        if orden:
            try:
                z,cov = np.polyfit(xvals, varvals, orden, w=varerr, cov=True)
            except ValueError:
                z,cov = np.polyfit(xvals, varvals, orden, cov=True)
            zerr = np.sqrt(cov[0,0] * np.sqrt(len(varvals)))
            polinomio = np.poly1d(z)
            h = np.linspace(min(xvals),max(xvals),100)
            plt.plot(h,polinomio(h),'--r',label=f'({z[0]:.3f} +- {zerr:.3f})')
        if label:
            plt.legend(loc='upper left', framealpha=1)
    def ver_todas(self, orden=None, amnt=0):
        for archivo in self.archivos[amnt:]:
            i, x, *vars = self.cargar(archivo)
            for var in vars:
                # plt.ion()
                print(f'ploteando {archivo}')
                self.plotear(x, var, orden=orden)
                plt.title(self.titular(archivo))
                # plt.ioff()
            plt.show()

if __name__ == '__main__':
    termo = experimento("/home/marco/Documents/fac/labo4/termometria/diados/tempsposta/")
    termo.get_archivos(claves=["temp",'csv'])
    termo.ver_todas()
    # x, y, z = termo.cargar(termo.archivos[0])
