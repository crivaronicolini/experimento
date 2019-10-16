
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
import pandas as pd
import os

class experimento():
    "una clase para agilizar la carga de datos de labo"
    def __init__(self, dire:str, claves:list =[]):
        '''Se inicializa con un directorio donde estan las mediciones.
        Setea ese directorio y la lista de archivos en el.
        Opcionalmente las mediciones se pueden filtrar por claves'''
        if os.path.isabs(dire):
            self.absdire = dire
        else:
            self.absdire = os.path.abspath(dire)
        self.archivos = sorted(os.listdir(self.absdire))
        if claves!=[]:
            self.archivos = [i for i in self.archivos if all(clave in i.lower() for clave in claves)]

    def _hacer_absarch(self):
        '''Usado para add y sub, porque si creo un nuevo experimento
        con ese resultado los archivos deben tener path absoluto.
        No cambio los archivos originales por si los quiero seguir
        usando'''
        self.absarch = [os.path.join(self.absdire, a) for a in self.archivos]

    def __len__(self):
        return list.__len__(self.archivos)

    def __contains__(self, archivo):
        # return True if archivo in self.archivos else False
        return self.archivos.__contains__(archivo)

    def __getitem__(self, archivo):
        return self.archivos.__getitem__(archivo)

    def __add__(self, exp):
        a = experimento('.')
        self._hacer_absarch()
        exp._hacer_absarch()
        a.archivos = self.absarch + exp.absarch
        return a

    def __sub__(self, exp):
        s = experimento('.')
        self._hacer_absarch()
        exp._hacer_absarch()
        s.archivos = list( set(self.absarch) - set(exp.absarch) )
        return s

    def __iter__(self):
        return self.archivos.__iter__()

    def cargar(self,archivo, delimiter:str =';'):
        arch = os.path.join(self.absdire, archivo)
        return np.loadtxt(arch, delimiter=delimiter, unpack = True, comments='$')

    @staticmethod
    def titular(archivo):
        return archivo.split('.')[0].replace('_',' ')

    @staticmethod
    def set_e(var, error:float, df: pd.core.frame.DataFrame =None):
        if df == None:
            error = np.ones(np.shape(var))*error
            var = un.uarray(var, error)
            return var
        else:
            error = np.ones(df[var].shape)*error
            df[var] = un.uarray(df[var], error)

    @staticmethod
    def get_e(var:np.ndarray):
        assert var.ndim == 1
        return un.std_devs(var)

    @staticmethod
    def get_v(var:np.ndarray):
        assert var.ndim == 1
        return un.nominal_values(var)

    def plotear(self,x, var, df:pd.core.frame.DataFrame=None, fill:bool =True, alpha:float =0.5, label:str =None,
            orden:int =None):
        if df != None:
            x = df[x]
            var = df[var]
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

    def ver_todas(self, orden:int =None, amnt: int =0):
        amnt = amnt - len(self.archivos) if amnt > len(self.archivos) else amnt
        for archivo in self.archivos[amnt:]:
            i, x, *vars = self.cargar(archivo)
            for var in vars:
                # plt.ion()
                print(f'ploteando {archivo}')
                self.plotear(x, var, orden=orden)
                plt.title(self.titular(archivo))
                # plt.ioff()
            plt.show()

    def cargar_pd(self,archivo:str):
        arch = os.path.join(self.absdire, archivo)
        return pd.read_csv(arch,  delimiter=',', skipinitialspace=True)

if __name__ == '__main__':
    termo = experimento(
            "/home/marco/Documents/fac/labo4/termometria/diados/tempsposta/",
            claves=["temp",'csv'])
    # termo.ver_todas(amnt=2)
    if termo.archivos[0] not in termo:
        print('hola')
    print(len(termo))
    vacio = experimento(
            "/home/marco/Documents/fac/labo4/vacio/",
            claves=['txt'])
    print(len(vacio))
    a = termo + vacio
    print(a.archivos)
