import os
import pandas as pd
import math
import numpy as np
import heapq
import random
from typing import List

### RUTAS ###
directorio_actual = os.path.dirname(__file__)
buenos_aires="S1_buenosAires.csv"
bogota = "S2_bogota.csv"
vancouver = "S3_vancouver.csv"
buenos_aires_out = "S4_buenosAiresR.csv"
ruta_archivo_s1 = os.path.join(directorio_actual, buenos_aires)
ruta_archivo_s2 = os.path.join(directorio_actual, bogota)
ruta_archivo_s3 = os.path.join(directorio_actual, vancouver)
ruta_archivo_s4 = os.path.join(directorio_actual, buenos_aires_out)

CTEMIN     = 1000000
ERROR_LISTA = 0.001

###  clases requeridas

class Heap:
    def __init__(self):
        self.heap = []
    
    def add(self, valor : float, simbolo : str):
        """valor es una probabilidad"""

        if(not valor == 0.0):
            if not simbolo:
                simbolo = ""
            heapq.heappush(self.heap, (valor, simbolo))
    
    def pop(self):
        valor, simbolo = heapq.heappop(self.heap)
        return valor, simbolo
    
    def __len__(self):
        return len(self.heap)

class NodoArbolHuffman:
    def __init__(self, símbolo=None, prob=None, izq=None, der=None, Real = False):
        self.heap = []
        self.símbolo = símbolo
        self.prob = prob
        self.izq = izq
        self.der = der
        self.código = None
        self.real= Real
    
    def set_real(self, real : bool):
        self.real = real
    
    def get_real(self) -> bool:
        return self.real

    def set_símbolo(self, símbolo : str):
        self.símbolo = símbolo
    
    def get_símbolo(self) -> str:
        return self.símbolo
    
    def set_prob(self, prob : float):
        self.prob = prob
    
    def get_prob(self) -> float:
        return self.prob
    
    def set_izq(self, izq):
        self.izq = izq
    
    def get_izq(self):
        return self.izq
    
    def set_der(self, der):
        self.der = der
    
    def get_der(self):
        return self.der
    
    def set_código(self, código : str):
        self.código = código
    
    def get_código(self) -> str:
        return self.código

    def es_hoja(self):
        izq = (self.izq == None)
        der = (self.der == None)
        return izq and der

    def imprimir_nodos(self, lista_simbolos, codigo_actual=""):
        if self.real:
            self.código = codigo_actual
            lista_simbolos[self.símbolo]= (self.código,self.prob) 
        if self.izq:
            self.izq.imprimir_nodos(lista_simbolos,codigo_actual + "0")  # Agregar 1 si se va a la izquierda
        if self.der:
            self.der.imprimir_nodos(lista_simbolos,codigo_actual + "1")  # Agregar 0 si se va a la derecha

def crear_heap(lista_pares : list):
    heap = Heap()
    for par in lista_pares:
        valor, simbolo = par
        heap.add(valor, simbolo)
    return heap

def calcular_media(ruta_archivo) -> float:
    contenido=pd.read_csv(ruta_archivo)
    suma=0
    n=0
    media=0
    for indice, fila in contenido.iterrows():
        valor = fila.iloc[0]
        suma+=valor
        n+=1
        media=suma/n
    return media

def calcular_desvio(ruta_archivo) -> float:
    contenido=pd.read_csv(ruta_archivo)
    suma_cuadrados_diff = 0
    media= calcular_media(ruta_archivo)
    for indice, fila in contenido.iterrows():
        valor = fila.iloc[0]
        suma_cuadrados_diff += (valor - media)**2
    varianza = suma_cuadrados_diff /  len(contenido)
    return math.sqrt(varianza)

def covarianza_AB(ruta_archivo_1,ruta_archivo_2) -> float:
    contenido1 = pd.read_csv(ruta_archivo_1)
    contenido2 = pd.read_csv(ruta_archivo_2)
    media_A = calcular_media(ruta_archivo_1)
    media_B = calcular_media(ruta_archivo_2)
    longitud = len(contenido1)
    cov = 0
    for i in range(longitud):
        valor_A = contenido1.iloc[i].values[0]
        valor_B = contenido2.iloc[i].values[0]
        cov+= (valor_A - media_A) * (valor_B - media_B)
    return cov / longitud

def factor_correlacion(ruta_archivo_1,ruta_archivo_2) -> float:
    cov = covarianza_AB(ruta_archivo_1,ruta_archivo_2)
    desv1 = calcular_desvio(ruta_archivo_1)
    desv2 = calcular_desvio(ruta_archivo_2)
    return cov / (desv1*desv2)

def categoria(numero) -> int:
    # devuelve la categoría segun la temperatura
    if(numero < 10):
        return 0
    elif(numero < 20):
        return 1
    else:
        return 2

def probabilidades_sin_memoria(ruta_archivo) -> List[float]: # B = 0 , M = 1 , A = 2
    contenido = pd.read_csv(ruta_archivo)
    cant_por_simbolo = [0,0,0]
    cant_total = len(contenido)
    for i in range(cant_total):
        simbolo = contenido.iloc[i].values[0]
        cant_por_simbolo[categoria(simbolo)]+=1
    for i in range(3):
        cant_por_simbolo[i]= cant_por_simbolo[i] / cant_total
    return cant_por_simbolo

def markoviana(ruta_archivo): # B = 0 , M = 1 , A = 2
    contenido = pd.read_csv(ruta_archivo)
    cant_por_simbolo = [0,0,0]
    prob_simbolo = [ [0 , 0 , 0 ], 
                    [ 0 , 0 , 0], 
                    [0 , 0 , 0 ] ]
    simbolo_anterior = contenido.iloc[0].values[0]   
    cant_por_simbolo[categoria(simbolo_anterior)]+=1
    longitud = len(contenido)
    for i in range(1, longitud):
        Simbolo = contenido.iloc[i].values[0]
        cant_por_simbolo[categoria(simbolo_anterior)]+=1
        prob_simbolo[categoria(Simbolo)][categoria(simbolo_anterior)]+=1
        simbolo_anterior=Simbolo
    for i in range(3):
        for j in range(3):
            if(cant_por_simbolo[i] != 0):
                prob_simbolo[j][i]= prob_simbolo[j][i] / cant_por_simbolo[i]

    return prob_simbolo

def entropia_sin_memoria(fuente) -> float:
    suma = 0
    for i in range(len(fuente)):
        if(fuente[i] != 0):
            suma += -(fuente[i] * math.log2(fuente[i]))
    return suma

def entropia_con_memoria(estacionario : List[float], fuente) -> float:
    # hc = sum prob estacionario * hi 
    # hi = entropia de la columna
    hi = [0,0,0]
    for i in range(len(estacionario)):
        for j in range(len(estacionario)):
            if(fuente[j][i] != 0):
                hi[i] += -(fuente[j][i] * math.log2(fuente[j][i]))
    hc=0
    for i in range(len(estacionario)):
        hc += estacionario[i] * hi[i]
    return hc

def arbol_Huffman(prob_pi : Heap) -> NodoArbolHuffman: ##prob_pi es un heap
    partes_arboles = []
    while len(prob_pi) != 1:
        valor_izq,simbolo_izq = prob_pi.pop()  
        nodo_izq = None
        indice = 0
        for elemento in partes_arboles:
            if(valor_izq == elemento.get_prob()):
                nodo_izq = elemento
            if(nodo_izq is None):
                indice+=1
        if(nodo_izq is None):
            nodo_izq = NodoArbolHuffman(símbolo=simbolo_izq,prob=valor_izq,Real=True)
        else:
            partes_arboles.pop(indice)

        valor_der,simbolo_der = prob_pi.pop() 
        nodo_der = None
        indice = 0
        for elemento in partes_arboles:
            if(valor_der == elemento.get_prob()):
                nodo_der = elemento
            if(nodo_der == None):
                indice+=1
        if(nodo_der is None):
            nodo_der = NodoArbolHuffman(símbolo=simbolo_der, prob=valor_der, Real=True)
        else:
            partes_arboles.pop(indice)

        suma_valores = valor_izq + valor_der
        nodo_nuevo = NodoArbolHuffman(símbolo=None, prob=suma_valores, izq=nodo_izq, der=nodo_der, Real=False)
        prob_pi.add(suma_valores, None)
        partes_arboles.append(nodo_nuevo)
        if len(prob_pi) == 1:
            return partes_arboles[0] # nodo raiz
    return []

def pares_prob_simb(probabilidades) -> List[map]:
    return [(probabilidades[0], "B"), (probabilidades[1], "M"), (probabilidades[2], "A")]

def pares_prob_simb_orden2(estacionario, probabilidades) -> List[map]:
    return [  (estacionario[0] * probabilidades[0][0], "BB"), (estacionario[1] * probabilidades[0][1], "MB"), (estacionario[2] * probabilidades[0][2], "AB") ,
                 (estacionario[0] * probabilidades[1][0], "BM"), (estacionario[1] * probabilidades[1][1], "MM"), (estacionario[2] * probabilidades[1][2], "AM"),
                 (estacionario[0] * probabilidades[2][0], "BA"), (estacionario[1] * probabilidades[2][1], "MA"), (estacionario[2] * probabilidades[2][2], "AA") ]

def long_promedio_codificacion(codificacion: map) -> float:
    sum = 0
    for clave, valor in codificacion.items():
        sum+= len(valor[0]) * valor[1] 
    return sum 

def categoria_simbolo(numero : int) -> str:
    # retorna el simbolo de la categoria segun la temperatura
    if(numero < 10):
        return "B"
    elif(numero < 20):
        return "M"
    else:
        return "A"

def generar_cadena(ruta_archivo) -> List[str]:
    contenido=pd.read_csv(ruta_archivo)
    cadena= []
    for indice, fila in contenido.iterrows():
        cadena.append(categoria_simbolo(fila.iloc[0]))
    return cadena

def generar_cadena_orden2(ruta_archivo) -> List[str]:
    contenido=pd.read_csv(ruta_archivo)
    cadena= []
    for i in range(0, len(contenido)-1, 2):
        simbolo_1 = categoria_simbolo(contenido.iloc[i].values[0])
        simbolo_2 = categoria_simbolo(contenido.iloc[i+1].values[0])
        simbolo_final= simbolo_1 + simbolo_2
        cadena.append(simbolo_final)
    return cadena
    
def cant_bits(cadena,codificacion):
    cantidad = 0
    for elemento in cadena:
        cantidad+= len(codificacion[elemento][0])
    return cantidad

def teorema_shannon_sin_memoria(entropia, longitud, orden):
    print(entropia, "<=" , longitud/orden,"<", entropia + 1/orden)

def teorema_shannon_con_memoria(h1, h_cond, longitud, orden):
    print(h1/orden+(1-1/orden)*h_cond, "<=" , longitud/orden,"<", h1/orden+(1-1/orden)*h_cond + 1/orden)

def matriz_canal(entrada, salida) -> List[List[float]]:
    contenido1 = pd.read_csv(entrada)
    contenido2 = pd.read_csv(salida)
    canal_entrada = [0, 0, 0]  # 0 B 1 M 2 A
    mat_canal = [[0, 0, 0], 
                 [0, 0, 0], 
                 [0, 0, 0]]
    longitud = len(contenido1)
    
    for i in range(longitud):
        valor_A = categoria(contenido1.iloc[i].values[0])
        valor_B = categoria(contenido2.iloc[i].values[0])
        canal_entrada[valor_A] += 1
        mat_canal[valor_B][valor_A] += 1
    
    for j in range(3):
        mat_canal[0][j] = (mat_canal[0][j] / canal_entrada[j])
        mat_canal[1][j] = (mat_canal[1][j] / canal_entrada[j])
        mat_canal[2][j] = (mat_canal[2][j] / canal_entrada[j])

    return mat_canal

def ruido_de_entradas(mat):
    suma = [0,0,0]
    for i in range (len(mat)):
        for j in range (len(mat)):
            if(mat[j][i] != 0):
                suma[i] += -(mat[j][i] * math.log2(mat[j][i]))
    return suma

def ruido(prob,ruido_de_entradas) -> float:
    suma = 0
    for i in range(len(ruido_de_entradas)):
        suma += ruido_de_entradas[i] * prob[i]
    return suma

def informacion_mutua(ruido_de_entradas) -> float:
    probHy = probabilidades_sin_memoria(ruta_archivo_s4)
    Estacionario = probabilidades_sin_memoria(ruta_archivo_s1)
    Hy = entropia_sin_memoria(probHy)
    Hyx = ruido(Estacionario,ruido_de_entradas)
    return Hy-Hyx

def print_matriz(matriz):
    for fila in matriz:
        print(f"{fila}")

def matriz_acumulada(matriz_probabilidad):
    matriz_probabilidad[1][0]= matriz_probabilidad[1][0] + matriz_probabilidad[0][0]
    matriz_probabilidad[2][0]= 1.0
    matriz_probabilidad[1][1]= matriz_probabilidad[1][1] + matriz_probabilidad[0][1]
    matriz_probabilidad[2][1]= 1.0
    matriz_probabilidad[1][2]= matriz_probabilidad[1][2] + matriz_probabilidad[0][2]
    matriz_probabilidad[2][2]= 1.0
    return matriz_probabilidad

def get_prox_simbolo(markov_acumulada,simb_anterior) -> int:
    x = random.random()
    for i in range (len(markov_acumulada)):
        if (x < markov_acumulada[i][simb_anterior]):
            return i
        
def converge(prob_ant,prob_act) -> bool:
    for i in range(0,len(prob_ant)):
        if (abs(prob_ant[i] - prob_act[i]) > ERROR_LISTA):
            return False
    return True

def prob_apariciones_entre_simb(simbolo, N, markov_acumulada, canal_acumulada) -> List[float]:
    probact = [0] * (N+1)
    probant = [-1] * (N+1)
    cant_veces_simbolo = [0] * (N+1)
    cant = 0
    while(not converge(probact, probant) or (cant < CTEMIN)) : 
        contador = 0
        simb = get_prox_simbolo(markov_acumulada,simbolo)
        simbSalida = get_prox_simbolo(canal_acumulada,simb)
        while (contador < N) and (simbSalida != simbolo):
            simb = get_prox_simbolo(markov_acumulada,simbolo) 
            simbSalida = get_prox_simbolo(canal_acumulada,simb)
            contador +=1
        if (simbSalida == simbolo):
            cant +=1
            cant_veces_simbolo[contador] += 1
            for i in range (N+1):
                probant[i] = probact[i]
                probact[i] = cant_veces_simbolo[i]/cant
    return probact


##### MAIN ###############

### Media y desvío de cada una de las señales ( 1.a )print("Media Buenos Aires", calcular_media(ruta_archivo_s1))
print("Media Bogota", calcular_media(ruta_archivo_s2))
print("Media Vancouver", calcular_media( ruta_archivo_s3))

print("Desvio Buenos Aires", calcular_desvio(ruta_archivo_s1))
print("Desvio Bogota", calcular_desvio(ruta_archivo_s2))
print("Desvio Vancouver", calcular_desvio( ruta_archivo_s3))


### Factor de correlación cruzada para cada par de señales ( 1.b )
print("Calcular Factor Correlacion BuenosAires-Bogota", factor_correlacion(ruta_archivo_s1,ruta_archivo_s2))
print("Calcular Factor Correlacion BuenosAires-Vancouver", factor_correlacion(ruta_archivo_s1,ruta_archivo_s3))
print("Calcular Factor Correlacion Bogota-Vancouver", factor_correlacion(ruta_archivo_s2,ruta_archivo_s3))


### Cálculo de entropía con y sin memoria ( 2.a )

# probabilidades sin memoria
probabilidades_buenos_aires = probabilidades_sin_memoria(ruta_archivo_s1)
probabilidades_bogota = probabilidades_sin_memoria(ruta_archivo_s2)
probabilidades_vancouver = probabilidades_sin_memoria(ruta_archivo_s3)

print('Fuente sin memoria, Buenos Aires:' , probabilidades_buenos_aires)
print('Fuente sin memoria, Bogota:' , probabilidades_bogota)
print('Fuente sin memoria, Vancouver:' , probabilidades_vancouver)

# probabilidades con memoria

markoviana_buenos_aires = markoviana(ruta_archivo_s1)
markoviana_bogota = markoviana(ruta_archivo_s2)
markoviana_vancouver = markoviana(ruta_archivo_s3)

print('Fuente con memoria, Buenos Aires:' , markoviana_buenos_aires)
print('Fuente con memoria, Bogota:' , markoviana_bogota)
print('Fuente con memoria, Vancouver:' , markoviana_vancouver)

# entropias

entropia_BS = entropia_sin_memoria(probabilidades_buenos_aires)
entropia_BT = entropia_sin_memoria(probabilidades_bogota)
entropia_VC = entropia_sin_memoria(probabilidades_vancouver)

print('Entropia orden 1 de Buenos Aires:', entropia_BS)
print('Entropia orden 1 de Bogota:', entropia_BT)
print('Entropia orden 1 de Vacnouver:', entropia_VC)

entropia_BSO2 = entropia_con_memoria(probabilidades_buenos_aires,markoviana_buenos_aires)
entropia_BTO2 = entropia_con_memoria(probabilidades_bogota,markoviana_bogota)
entropia_VCO2 = entropia_con_memoria(probabilidades_vancouver,markoviana_vancouver)

print('Entropia orden 2 de Buenos Aires:', entropia_BSO2)
print('Entropia orden 2 de Bogota:', entropia_BTO2)
print('Entropia orden 2 de Vacnouver:', entropia_VCO2)

### Generación de códigos de Huffman ( 2.b )
### Cálculo de longitud total en bits y tasa de compresión obtenida ( 2.c )
pares_buenos_aires = pares_prob_simb(probabilidades_buenos_aires)
pares_bogota = pares_prob_simb(probabilidades_bogota)
pares_vancouver = pares_prob_simb(probabilidades_vancouver)
heap_BS = crear_heap(pares_buenos_aires)
heap_BT = crear_heap(pares_bogota)
heap_VC = crear_heap(pares_vancouver)
arbol_BS = arbol_Huffman(heap_BS)
arbol_BT = arbol_Huffman(heap_BT)
arbol_VC = arbol_Huffman(heap_VC)
simbolos_BS = {}
arbol_BS.imprimir_nodos(simbolos_BS)
simbolos_BT = {}
arbol_BT.imprimir_nodos(simbolos_BT)
simbolos_VC = {}
arbol_VC.imprimir_nodos(simbolos_VC)
print(simbolos_BS)
print(simbolos_BT)
print(simbolos_VC)


print("Tamaño En Bits Bs ",cant_bits(generar_cadena(ruta_archivo_s1), simbolos_BS))
print("Tamaño En Bits BT ", cant_bits(generar_cadena(ruta_archivo_s2), simbolos_BT))
print("Tamaño En Bits VC ",cant_bits(generar_cadena(ruta_archivo_s3), simbolos_VC))


pares_buenos_aires_O2 = pares_prob_simb_orden2(probabilidades_buenos_aires,markoviana_buenos_aires)
pares_bogota_O2 = pares_prob_simb_orden2(probabilidades_bogota,markoviana_bogota)
pares_vancouver_O2 = pares_prob_simb_orden2(probabilidades_vancouver,markoviana_vancouver)
heap_BSO2 = crear_heap(pares_buenos_aires_O2)
heap_BTO2 = crear_heap(pares_bogota_O2)
heap_VCO2 = crear_heap(pares_vancouver_O2)
arbol_BSO2 = arbol_Huffman(heap_BSO2)
arbol_BTO2 = arbol_Huffman(heap_BTO2)
arbol_VCO2 = arbol_Huffman(heap_VCO2)
simbolos_BSO2 = {}
arbol_BSO2.imprimir_nodos(simbolos_BSO2)
simbolos_BTO2 = {}
arbol_BTO2.imprimir_nodos(simbolos_BTO2)
simbolos_VCO2 = {}
arbol_VCO2.imprimir_nodos(simbolos_VCO2)
print(simbolos_BSO2)
print(simbolos_BTO2)
print(simbolos_VCO2)

print("Tamaño En Bits BsO2 ", cant_bits(generar_cadena_orden2(ruta_archivo_s1), simbolos_BSO2))
print("Tamaño En Bits BTO2 ", cant_bits(generar_cadena_orden2(ruta_archivo_s2), simbolos_BTO2))
print("Tamaño En Bits VCO2 ", cant_bits(generar_cadena_orden2(ruta_archivo_s3), simbolos_VCO2))

long_BS = long_promedio_codificacion(simbolos_BS)
long_BT = long_promedio_codificacion(simbolos_BT)
long_VC = long_promedio_codificacion(simbolos_VC)
long_BSO2 = long_promedio_codificacion(simbolos_BSO2)
long_BTO2 = long_promedio_codificacion(simbolos_BTO2)
long_VCO2 = long_promedio_codificacion(simbolos_VCO2)

### Cálculo de teorema de shannon
teorema_shannon_sin_memoria(entropia_BS,long_BS,1)
teorema_shannon_sin_memoria(entropia_BT,long_BT,1)
teorema_shannon_sin_memoria(entropia_VC,long_VC,1)

teorema_shannon_con_memoria(entropia_BS,entropia_BSO2,long_BSO2,2)
teorema_shannon_con_memoria(entropia_BT,entropia_BTO2,long_BTO2,2)
teorema_shannon_con_memoria(entropia_VC,entropia_VCO2,long_VCO2,2)


###  Cálculo de la matriz del canal ( 3.a )
matriz_yx = matriz_canal(ruta_archivo_s1,ruta_archivo_s4)

### Cálculo del ruido del canal e Información Mutua ( 3.b )

ruiditos_buenos_aires = ruido_de_entradas(matriz_yx)
print(ruido(probabilidades_buenos_aires,ruiditos_buenos_aires))
markov_acumulada=matriz_acumulada(markoviana_buenos_aires)
canal_acumulada = matriz_acumulada(matriz_yx)
print(informacion_mutua(ruiditos_buenos_aires))

### Simulación computacional de aparición entre simbolos ( 3.c )
simbolo = 0 # 0 = B, 1 = M, 2 = A
n = 2
print(prob_apariciones_entre_simb(simbolo,n,markov_acumulada,canal_acumulada))