import os
import pandas as pd
import math
import numpy as np
import heapq


##parte archivo
directorio_actual = os.path.dirname(__file__)
BuenosAires="S1_buenosAires.csv"
Bogota = "S2_bogota.csv"
Vancouver = "S3_vancouver.csv"
ruta_archivo_s1 = os.path.join(directorio_actual, BuenosAires)
ruta_archivo_s2 = os.path.join(directorio_actual, Bogota)
ruta_archivo_s3 = os.path.join(directorio_actual, Vancouver)

###  hufman

class Heap:
    def __init__(self):
        self.heap = []
    
    def agregar_valor(self, valor, simbolo): #valor es probabilidad
        if(not valor == 0.0):
            if not simbolo:
                simbolo = ""
            heapq.heappush(self.heap, (valor, simbolo))
    
    def obtener_siguiente(self):
        valor, simbolo = heapq.heappop(self.heap)
        return valor, simbolo
    
    def tamaño_heap(self):
        return len(self.heap)

def crear_heap(lista_pares):
    heap = Heap()
    for par in lista_pares:
        valor, simbolo = par
        heap.agregar_valor(valor, simbolo)
    
    return heap

class NodoArbolHuffman:
    def __init__(self, símbolo=None, prob=None, izq=None, der=None, código=None, Real = False):
        self.heap = []
        self.símbolo = símbolo
        self.prob = prob
        self.izq = izq
        self.der = der
        self.código = código
        self.Real= Real
    
    def set_Real(self, Real):
        self.Real = Real
    
    def get_Real(self):
        return self.Real

    def set_símbolo(self, símbolo):
        self.símbolo = símbolo
    
    def get_símbolo(self):
        return self.símbolo
    
    def set_prob(self, prob):
        self.prob = prob
    
    def get_prob(self):
        return self.prob
    
    def set_izq(self, izq):
        self.izq = izq
    
    def get_izq(self):
        return self.izq
    
    def set_der(self, der):
        self.der = der
    
    def get_der(self):
        return self.der
    
    def set_código(self, código):
        self.código = código
    
    def get_código(self):
        return self.código

    def isHoja(self):
        LadoIzq = (self.izq == None)
        LadoDer = (self.der == None)
        return LadoIzq and LadoDer
    def __eq__(self, other):
        if isinstance(other, NodoArbolHuffman):
            return (self.símbolo == other.símbolo and
                    self.prob == other.prob and
                    self.izq == other.izq and
                    self.der == other.der and
                    self.código == other.código and
                    self.Real == other.Real)
        return False
    def imprimir_nodos(self,Devolver, codigo_actual=""):
        if self.Real:
            self.código = codigo_actual
            Devolver[self.símbolo]=self.código
            #Devolver.append( (self.símbolo, self.código, self.prob) )
            #print(f"Símbolo: {self.símbolo}, Código: {self.código}, Probabilidad: {self.prob}")
        
        if self.izq:
            self.izq.imprimir_nodos(Devolver,codigo_actual + "0")  # Agregar 1 si se va a la izquierda
        if self.der:
            self.der.imprimir_nodos(Devolver,codigo_actual + "1")  # Agregar 0 si se va a la derecha
##########################################

def CalcularMedia(RutaArchivo):
    contenido=pd.read_csv(RutaArchivo)
    Suma=0
    N=0
    Media=0
    for indice, fila in contenido.iterrows():
        Valor = fila.iloc[0]
        Suma+=Valor
        N+=1
        Media=Suma/N
    return Media

def CalcularDesvio(RutaArchivo):
    contenido=pd.read_csv(RutaArchivo)
    suma_cuadrados_diff = 0
    Media= CalcularMedia(RutaArchivo)
    for indice, fila in contenido.iterrows():
        Valor = fila.iloc[0]
        suma_cuadrados_diff += (Valor - Media)**2
    varianza = suma_cuadrados_diff /  len(contenido)
    return math.sqrt(varianza)


def CalcularCovarianzaAB(RutaArchivo1,RutaArchivo2):
    contenido1=pd.read_csv(RutaArchivo1)
    contenido2=pd.read_csv(RutaArchivo2)
    MediaA=CalcularMedia(RutaArchivo1)
    MediaB=CalcularMedia(RutaArchivo2)
    longitud = len(contenido1)
    cov=0
    for i in range(longitud):
        ValorA = contenido1.iloc[i].values[0]
        ValorB = contenido2.iloc[i].values[0]
        cov+= (ValorA - MediaA) * (ValorB - MediaB)

    return cov / longitud

def CalcularFactorCorrelacion(RutaArchivo1,RutaArchivo2):
    cov=CalcularCovarianzaAB(RutaArchivo1,RutaArchivo2)
    desv1=CalcularDesvio(RutaArchivo1)
    desv2=CalcularDesvio(RutaArchivo2)
    return cov / (desv1*desv2)

def DevolverCategoria(numero):
    if(numero < 10):
        return 0
    elif(numero < 20):
        return 1
    else:
        return 2

def GenerarProbabilidadesSinMemoria(RutaArchivo): # B = 0 , M = 1 , A = 2
    contenido=pd.read_csv(RutaArchivo)
    CantidadPorSimbolo = [0,0,0]
    CantidadTotal= len(contenido)
    for i in range(CantidadTotal):
        Simbolo = contenido.iloc[i].values[0]
        CantidadPorSimbolo[DevolverCategoria(Simbolo)]+=1
    for i in range(3):
        CantidadPorSimbolo[i]= CantidadPorSimbolo[i] / CantidadTotal
    return CantidadPorSimbolo

def GenerarProbabilidadesConMemoria(RutaArchivo): # B = 0 , M = 1 , A = 2
    contenido=pd.read_csv(RutaArchivo)
    CantidadPorSimbolo = [0,0,0]
    ProbabilidadCadaSimbolo = [ [0 , 0 , 0 ] , [ 0 , 0 , 0], [0 , 0 , 0 ] ]
    SimboloAnterior = contenido.iloc[0].values[0]   
    CantidadPorSimbolo[DevolverCategoria(SimboloAnterior)]+=1
    longitud = len(contenido)
    for i in range(1, longitud):
        Simbolo = contenido.iloc[i].values[0]
        CantidadPorSimbolo[DevolverCategoria(SimboloAnterior)]+=1
        ProbabilidadCadaSimbolo[DevolverCategoria(Simbolo)][DevolverCategoria(SimboloAnterior)]+=1
        SimboloAnterior=Simbolo
    for i in range(3):
        for j in range(3):
            if(not CantidadPorSimbolo[i]== 0):
                ProbabilidadCadaSimbolo[j][i]= ProbabilidadCadaSimbolo[j][i] / CantidadPorSimbolo[i]

    return ProbabilidadCadaSimbolo

def GenerarFuenteMarkoviana(MatrizProbabilidad): ##arreglar xd
    MatrizProbabilidad[1][0]= MatrizProbabilidad[1][0] + MatrizProbabilidad[0][0]
    MatrizProbabilidad[2][0]= 1.0
    MatrizProbabilidad[1][1]= MatrizProbabilidad[1][1] + MatrizProbabilidad[0][1]
    MatrizProbabilidad[2][1]= 1.0
    MatrizProbabilidad[1][2]= MatrizProbabilidad[1][2] + MatrizProbabilidad[0][2]
    MatrizProbabilidad[2][2]= 1.0
    return MatrizProbabilidad

def GenerarFuente(VectorProbabilidades): ##arreglar xd
    VectorProbabilidades[1]= VectorProbabilidades[1] + VectorProbabilidades[0]
    VectorProbabilidades[2]=1.0
    return VectorProbabilidades

def CalcularEntropiaSinMemoria(Fuente):
    Suma=0
    for i in range(len(Fuente)):
        if(not Fuente[i]==0):
            Suma += -(Fuente[i] * math.log2(Fuente[i]))
    return Suma

def CalcularEntropiaConMemoria(Estacionario,Fuente):
    # hc= sum prob estacionario * hi 
    # hi= entropia de la columna
    hi = [0,0,0]
    for i in range(len(Estacionario)):
        for j in range(len(Estacionario)):
            if(not Fuente[j][i]==0):
                hi[i] += -(Fuente[j][i] * math.log2(Fuente[j][i]))
    hc=0
    for i in range(len(Estacionario)):
        hc += Estacionario[i] * hi[i]
    return hc

def CreararbolHuffman(prob_pi): ##prob_pi es un heap
    Partes_Arboles = []
    while prob_pi.tamaño_heap() != 1:
        Valor_Izq,Simbolo_Izq = prob_pi.obtener_siguiente()     #getPrimero lo saca del Heap
        Nodo_Izq=None
        indice=0
        for Elementos in Partes_Arboles:
            if(Valor_Izq== Elementos.get_prob()):
                Nodo_Izq=Elementos
            if(Nodo_Izq==None):
                indice+=1
        if(Nodo_Izq==None):
            Nodo_Izq = NodoArbolHuffman(Simbolo_Izq,Valor_Izq, None, None, "1",True)
        else:
            Partes_Arboles.pop(indice)
                

        Valor_Der,Simbolo_Der = prob_pi.obtener_siguiente() ##siempre da el menor
        Nodo_Der=None
        indice=0
        for Elementos in Partes_Arboles:
            if(Valor_Der== Elementos.get_prob()):
                Nodo_Der=Elementos
            if(Nodo_Der==None):
                indice+=1
        if(Nodo_Der==None):
            Nodo_Der = NodoArbolHuffman(Simbolo_Der,Valor_Der, None, None, "0",True)
        else:
            Partes_Arboles.pop(indice)

        Valor_Suma = Valor_Izq + Valor_Der
        Nodo_Nuevo = NodoArbolHuffman(None,Valor_Suma,Nodo_Izq,Nodo_Der,None,False)
        prob_pi.agregar_valor(Valor_Suma, None)
        Partes_Arboles.append(Nodo_Nuevo)

    Nodo_Raiz = Partes_Arboles[0]
    return Nodo_Raiz

def generarParesProbSimb(Probabilidades):
    devolver = [(Probabilidades[0], "B"), (Probabilidades[1], "M"), (Probabilidades[2], "A")]
    return devolver

def generarParesProbSimbOrden2(Estacionario, Probabilidades):
    devolver = [  (Estacionario[0] * Probabilidades[0][0], "BB"), (Estacionario[1] * Probabilidades[0][1], "MB"), (Estacionario[2] * Probabilidades[0][2], "AB") ,
                 (Estacionario[0] * Probabilidades[1][0], "BM"), (Estacionario[1] * Probabilidades[1][1], "MM"), (Estacionario[2] * Probabilidades[1][2], "AM"),
                 (Estacionario[0] * Probabilidades[2][0], "BA"), (Estacionario[1] * Probabilidades[2][1], "MA"), (Estacionario[2] * Probabilidades[2][2], "AA") ]
    return devolver

def CalcularLongitudPromedioCodificacion(Codificacion):
    sum=0
    longitud=len(Codificacion)
    for i in range(longitud):
        sum+= len(Codificacion[i][1])
    return sum / longitud

def DevolverCategoriaSimbolo(numero):
    if(numero < 10):
        return "B"
    elif(numero < 20):
        return "M"
    else:
        return "A"

def GenerarCadena(RutaArchivo):
    contenido=pd.read_csv(RutaArchivo)
    Cadena= []
    for indice, fila in contenido.iterrows():
        Cadena.append(DevolverCategoriaSimbolo(fila.iloc[0]))
    return Cadena

def GenerarCadenaOrden2(RutaArchivo):
    contenido=pd.read_csv(RutaArchivo)
    Cadena= []
    for i in range(0, len(contenido)-1, 2):
        Simbolo1 = DevolverCategoriaSimbolo(contenido.iloc[i].values[0])
        Simbolo2 = DevolverCategoriaSimbolo(contenido.iloc[i+1].values[0])
        SimboloFinal= Simbolo1 + Simbolo2
        Cadena.append(SimboloFinal)
    return Cadena


def CalcularCantidadBits(Cadena,Codificacion):
    cantidad=0
    for elemento in Cadena:
        cantidad+= len(Codificacion[elemento])
    return cantidad

##### MAIN ###3
#print("Media Buenos Aires", CalcularMedia(ruta_archivo_s1))
#print("Media Bogota", CalcularMedia(ruta_archivo_s2))
#print("Media Vancouver", CalcularMedia( ruta_archivo_s3))

#print("Desvio Buenos Aires", CalcularDesvio(ruta_archivo_s1))
#print("Desvio Bogota", CalcularDesvio(ruta_archivo_s2))
#print("Desvio Vancouver", CalcularDesvio( ruta_archivo_s3))

#print("Calcular Factor Correlacion BuenosAires-Bogota", CalcularFactorCorrelacion(ruta_archivo_s1,ruta_archivo_s2))
#print("Calcular Factor Correlacion BuenosAires-Vancouver", CalcularFactorCorrelacion(ruta_archivo_s1,ruta_archivo_s3))
#print("Calcular Factor Correlacion Bogota-Vancouver", CalcularFactorCorrelacion(ruta_archivo_s2,ruta_archivo_s3))

### sin memoria es igual al estacionario
ProbabilidadesBuenosAires = GenerarProbabilidadesSinMemoria(ruta_archivo_s1)
ProbabilidadesBogota = GenerarProbabilidadesSinMemoria(ruta_archivo_s2)
ProbabilidadesVancouver = GenerarProbabilidadesSinMemoria(ruta_archivo_s3)

#FuenteSinMemoriaBuenosAires = GenerarFuente(ProbabilidadesBuenosAires)
#FuenteSinMemoriaBogota =GenerarFuente(ProbabilidadesBogota)
#FuenteSinMemoriaVancouver = GenerarFuente(ProbabilidadesVancouver)

###### con memoria

ProbabilidadesMemoriaBuenosAires = GenerarProbabilidadesConMemoria(ruta_archivo_s1)
ProbabilidadesMemoriaBogota = GenerarProbabilidadesConMemoria(ruta_archivo_s2)
ProbabilidadesMemoriaVancouver = GenerarProbabilidadesConMemoria(ruta_archivo_s3)


#FuenteMarkovianaBuenosAires=GenerarFuenteMarkoviana(ProbabilidadesMemoriaBuenosAires)
#FuenteMarkovianaBogota=GenerarFuenteMarkoviana(ProbabilidadesMemoriaBogota)
#FuenteMarkovianaVancouver=GenerarFuenteMarkoviana(ProbabilidadesMemoriaVancouver)


#### entropias

#print(CalcularEntropiaSinMemoria(ProbabilidadesBuenosAires))
#print(CalcularEntropiaSinMemoria(ProbabilidadesBogota))
#print(CalcularEntropiaSinMemoria(ProbabilidadesVancouver))


#print(CalcularEntropiaConMemoria(ProbabilidadesBuenosAires,ProbabilidadesMemoriaBuenosAires))
#print(CalcularEntropiaConMemoria(ProbabilidadesBogota,ProbabilidadesMemoriaBogota))
#print(CalcularEntropiaConMemoria(ProbabilidadesVancouver,ProbabilidadesMemoriaVancouver))


########################## hufman
ParesBuenosAires = generarParesProbSimb(ProbabilidadesBuenosAires)
ParesBogota = generarParesProbSimb(ProbabilidadesBogota)
ParesVancouver = generarParesProbSimb(ProbabilidadesVancouver)
heapBS = crear_heap(ParesBuenosAires)
heapBT = crear_heap(ParesBogota)
heapVC = crear_heap(ParesVancouver)
ArbolBS = CreararbolHuffman(heapBS)
ArbolBT = CreararbolHuffman(heapBT)
ArbolVC = CreararbolHuffman(heapVC)
ListaSimbolosBS = {}
ArbolBS.imprimir_nodos(ListaSimbolosBS)
ListaSimbolosBT = {}
ArbolBT.imprimir_nodos(ListaSimbolosBT)
ListaSimbolosVC = {}
ArbolVC.imprimir_nodos(ListaSimbolosVC)

print("Tamaño En Bits Bs ",CalcularCantidadBits( GenerarCadena(ruta_archivo_s1), ListaSimbolosBS) )
print("Tamaño En Bits BT ", CalcularCantidadBits( GenerarCadena(ruta_archivo_s2), ListaSimbolosBT) )
print("Tamaño En Bits VC ",CalcularCantidadBits( GenerarCadena(ruta_archivo_s3), ListaSimbolosVC) )

ParesBuenosAiresOrden2 = generarParesProbSimbOrden2(ProbabilidadesBuenosAires,ProbabilidadesMemoriaBuenosAires)
ParesBogotaOrden2 = generarParesProbSimbOrden2(ProbabilidadesBogota,ProbabilidadesMemoriaBogota)
ParesVancouverOrden2 = generarParesProbSimbOrden2(ProbabilidadesVancouver,ProbabilidadesMemoriaVancouver)
heapBSO2 = crear_heap(ParesBuenosAiresOrden2)
heapBTO2 = crear_heap(ParesBogotaOrden2)
heapVCO2 = crear_heap(ParesVancouverOrden2)
ArbolBSO2 = CreararbolHuffman(heapBSO2)
ArbolBTO2 = CreararbolHuffman(heapBTO2)
ArbolVCO2 = CreararbolHuffman(heapVCO2)
ListaSimbolosBSO2 = {}
ArbolBSO2.imprimir_nodos(ListaSimbolosBSO2)
ListaSimbolosBTO2 = {}
ArbolBTO2.imprimir_nodos(ListaSimbolosBTO2)
ListaSimbolosVCO2 = {}
ArbolVCO2.imprimir_nodos(ListaSimbolosVCO2)

print("Tamaño En Bits BsO2 ", CalcularCantidadBits( GenerarCadenaOrden2(ruta_archivo_s1), ListaSimbolosBSO2) )
print("Tamaño En Bits BTO2 ", CalcularCantidadBits( GenerarCadenaOrden2(ruta_archivo_s2), ListaSimbolosBTO2) )
print("Tamaño En Bits VCO2 ", CalcularCantidadBits( GenerarCadenaOrden2(ruta_archivo_s3), ListaSimbolosVCO2) )

#print(CalcularLongitudPromedioCodificacion(ListaSimbolosBS))
#print(CalcularLongitudPromedioCodificacion(ListaSimbolosBT))
#print(CalcularLongitudPromedioCodificacion(ListaSimbolosVC))

