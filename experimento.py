
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
    def __init__(self, dire):
        if os.path.isabs(dire):
            self.absdire = dire
        else:
            self.absdire = os.path.abspath(dire)
    def get_archivos(self, claves=[]):
        self.archivos = sorted(os.listdir(self.absdire))
        if claves!=[]:
            self.archivos = [i for i in self.archivos if all(clave in i for clave in claves)]
        else:
            pass
        return self.archivos
    def cargar(self,archivo):
        arch = os.path.join(self.absdire, archivo)
        return np.loadtxt(arch, delimiter=',', unpack = True)
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
    def plotear(self,X, var, fill=True, alpha=0.5, label=None):
        Xvals = self.get_v(X)
        Xerr = self.get_e(X)
        varvals = self.get_v(var)
        varerr = self.get_e(var)
        plt.plot(Xvals, varvals, '.', label=label)
        if fill:
            varmenos = varvals - varerr
            varmas = varvals + varerr
            plt.fill_between(Xvals, varmenos, varmas, alpha=alpha)
        else:
            plt.errorbar(Xvals, varvals, xerr=Xerr, yerr=varerr, fmt='.')
        if label:
            plt.loc('best')
    def ver_todas(self,**kwargs):
        for archivo in self.archivos:
            *vars, x = self.cargar(archivo)
            for var in vars:
                plt.ion()
                print(f'ploteando {archivo}')
                self.plotear(x, var)
                plt.title(self.titular(archivo))
                plt.ioff()
                plt.show()

if __name__ == '__main__':
    termo = experimento("/home/marco/Documents/fac/labo4/termometria/diados/tempsposta/")
    termo.get_archivos(claves=["temp",'csv'])
    termo.ver_todas()
    # x, y, z = termo.cargar(termo.archivos[0])

