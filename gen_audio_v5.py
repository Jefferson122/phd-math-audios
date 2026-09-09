#!/usr/bin/env python3
"""
gen_audio_v5.py — Audio matemático tipo profesor, voz neuronal kokoro pm_alex.
Explicaciones pre-escritas en español natural + texto limpio de la solución.
NO intenta leer LaTeX símbolo a símbolo.
"""
import os, re, subprocess, glob, json, wave
import numpy as np
import kokoro_onnx

SOLUTIONS_DIR = '/home/user/phd-math-audios/NumericalAnalysis/solutions'
OUT_DIR       = '/home/user/phd-math-audios/NumericalAnalysis/solutions_pdf'
ENUNCIADOS    = json.loads(open(
    '/tmp/claude-0/-home-user/1e960dbe-2a99-5b49-9919-5071cfede866/scratchpad/enunciados.json'
).read())

KOKORO_MODEL  = '/tmp/claude-0/-home-user/1e960dbe-2a99-5b49-9919-5071cfede866/scratchpad/kokoro/kokoro-v1.0.onnx'
KOKORO_VOICES = '/tmp/claude-0/-home-user/1e960dbe-2a99-5b49-9919-5071cfede866/scratchpad/kokoro/voices-v1.1.bin'
KOKORO_RATE   = 24000

LABELS = {
    '2011-01': 'Enero de 2011',    '2011-08': 'Agosto de 2011',
    '2012-01': 'Enero de 2012',    '2012-08': 'Agosto de 2012',
    '2013-01': 'Enero de 2013',    '2013-08': 'Agosto de 2013',
    '2014-01': 'Enero de 2014',    '2014-08': 'Agosto de 2014',
    '2015-08': 'Agosto de 2015',
    '2016-01': 'Enero de 2016',    '2016-08': 'Agosto de 2016',
    '2017-01': 'Enero de 2017',    '2017-08': 'Agosto de 2017',
    '2018-01': 'Enero de 2018',    '2018-08': 'Agosto de 2018',
    '2019-01': 'Enero de 2019',    '2019-08': 'Agosto de 2019',
    '2019-12': 'Diciembre de 2019',
}

# ─── Kokoro TTS ────────────────────────────────────────────────────────────

_kokoro = None

def get_kokoro():
    global _kokoro
    if _kokoro is None:
        _kokoro = kokoro_onnx.Kokoro(KOKORO_MODEL, KOKORO_VOICES)
    return _kokoro


def save_wav(samples, rate, path):
    data = (samples * 32767).astype(np.int16)
    with wave.open(path, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(data.tobytes())


def make_silence(path, duration=0.6):
    subprocess.run(
        ['sox', '-n', '-r', str(KOKORO_RATE), '-c', '1', path, 'trim', '0.0', str(duration)],
        capture_output=True
    )


def kokoro_speak(text, wav_path):
    text = clean_for_tts(text)
    if not text:
        make_silence(wav_path, 0.3)
        return

    kokoro = get_kokoro()

    MAX = 800
    if len(text) <= MAX:
        samples, rate = kokoro.create(text, voice='pm_alex', speed=0.85, lang='es')
        save_wav(samples, rate, wav_path)
        return

    # Split into sentences, merge into chunks ≤ MAX chars
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur = [], ''
    for s in sentences:
        if len(cur) + len(s) + 1 <= MAX:
            cur = (cur + ' ' + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)

    chunk_wavs = []
    for idx, chunk in enumerate(chunks):
        cw = wav_path.replace('.wav', f'_ck{idx}.wav')
        samples, rate = kokoro.create(chunk, voice='pm_alex', speed=0.85, lang='es')
        save_wav(samples, rate, cw)
        chunk_wavs.append(cw)

    subprocess.run(['sox'] + chunk_wavs + [wav_path], capture_output=True)
    for cw in chunk_wavs:
        if os.path.exists(cw):
            os.unlink(cw)


# ─── Comprehensive topic explanations ─────────────────────────────────────
TOPIC_SCRIPTS = {

'newton': [
    "El método de Newton-Raphson es el algoritmo más potente para encontrar raíces de funciones. "
    "La idea central es geométrica y elegante. En el punto actual, calculamos la tangente a la curva "
    "y buscamos dónde esa tangente cruza el eje horizontal. Ese punto de cruce es nuestra siguiente aproximación.",

    "La fórmula de iteración se obtiene directamente de la geometría. "
    "Tomamos el punto actual, y le restamos el cociente entre la función evaluada en ese punto "
    "y su derivada evaluada en ese mismo punto. "
    "Repetimos este proceso hasta que la función sea suficientemente pequeña o hasta que "
    "dos iteraciones consecutivas difieran menos que la tolerancia dada.",

    "La convergencia de Newton-Raphson es cuadrática cerca de la raíz. "
    "Esto significa que si en la iteración actual cometemos un error de orden épsilon, "
    "en la siguiente iteración el error es del orden del cuadrado de épsilon. "
    "Dicho de otro modo: si tenemos dos decimales correctos, en la siguiente iteración tenemos cuatro, "
    "luego ocho, y así sucesivamente. Esta convergencia acelerada es lo que hace el método tan eficiente.",

    "La demostración de la convergencia cuadrática usa el desarrollo de Taylor. "
    "Si alfa es la raíz exacta y e sub n es el error en la iteración n, "
    "entonces el error en la siguiente iteración es proporcional al cuadrado del error actual, "
    "con un factor que involucra la segunda derivada de la función dividida entre dos veces la primera derivada, "
    "todo evaluado en la raíz. Para que el método funcione bien, es crucial que la derivada en la raíz "
    "sea distinta de cero. Cuando la derivada se anula, la raíz es múltiple "
    "y la convergencia se degrada a lineal.",

    "Para aplicar Newton en la práctica, necesitamos un punto de partida suficientemente cercano a la raíz. "
    "Si estamos demasiado lejos, el método puede divergir o converger a una raíz diferente. "
    "Una buena estrategia es usar primero el método de bisección para localizar la raíz aproximadamente, "
    "y luego cambiar a Newton para lograr alta precisión con pocas iteraciones adicionales.",
],

'fixed': [
    "La iteración de punto fijo es la base teórica de muchos métodos numéricos, incluido Newton-Raphson. "
    "La idea es reformular el problema de encontrar una raíz como un problema equivalente: "
    "buscamos un punto donde una función g de x sea igual a x. "
    "Ese punto, donde la función g no mueve al punto, se llama punto fijo.",

    "La condición de convergencia es fundamental. "
    "Si queremos que la iteración converja, la derivada de g debe ser menor que uno en valor absoluto "
    "en toda una vecindad de la solución. "
    "Cuanto más pequeño sea ese valor absoluto, más rápida será la convergencia. "
    "Si la derivada es exactamente cero en la solución, la convergencia es cuadrática, "
    "lo cual es precisamente lo que ocurre con Newton-Raphson.",

    "El Teorema del Punto Fijo de Banach garantiza la existencia y unicidad de la solución. "
    "Dice que si g es una contracción en un intervalo cerrado, es decir, "
    "si acerca los puntos entre sí con un factor estrictamente menor que uno, "
    "entonces existe exactamente un punto fijo en ese intervalo, "
    "y la iteración converge a él desde cualquier punto inicial del intervalo.",

    "Para demostrar que un método de iteración tiene un determinado orden de convergencia, "
    "se expande g en serie de Taylor alrededor del punto fijo. "
    "El orden de convergencia viene determinado por la primera derivada no nula de g en ese punto. "
    "Si la primera derivada no es nula, la convergencia es lineal. "
    "Si la primera derivada es nula pero la segunda no lo es, la convergencia es cuadrática. "
    "Y así sucesivamente para órdenes superiores.",
],

'bisection': [
    "El método de bisección es el más simple y confiable para localizar raíces. "
    "Se basa en el Teorema del Valor Intermedio: si una función continua cambia de signo "
    "en los extremos de un intervalo, entonces existe al menos una raíz en su interior.",

    "El algoritmo es directo. Evaluamos la función en el punto medio del intervalo. "
    "Si la función vale cero exactamente, encontramos la raíz. "
    "Si la función en el punto medio tiene el mismo signo que en el extremo derecho, "
    "entonces la raíz está en la mitad izquierda, y nos quedamos con esa mitad. "
    "Si tiene el mismo signo que en el extremo izquierdo, la raíz está en la derecha. "
    "En cualquier caso, después de cada iteración el intervalo se reduce a la mitad.",

    "La convergencia es lineal: el error se reduce a la mitad en cada iteración. "
    "Esto es más lento que Newton-Raphson, pero tiene una ventaja enorme: "
    "el método siempre converge si empezamos con un intervalo que contiene una raíz. "
    "No necesitamos información sobre la derivada, ni preocuparnos por divergencia.",

    "Para estimar el número de iteraciones necesarias, usamos la siguiente regla. "
    "Si queremos que el error sea menor que una tolerancia dada, necesitamos "
    "aproximadamente el logaritmo en base dos del cociente entre el tamaño inicial del intervalo "
    "y la tolerancia. Por ejemplo, para reducir el intervalo de longitud uno a una precisión de diez a la menos diez, "
    "necesitamos alrededor de treinta y tres iteraciones.",
],

'jacobi-gs': [
    "Los métodos iterativos de Jacobi y Gauss-Seidel resuelven sistemas de ecuaciones lineales "
    "de forma iterativa, refinando sucesivamente una solución aproximada. "
    "Son especialmente útiles para sistemas grandes y dispersos, donde los métodos directos "
    "como la eliminación gaussiana serían demasiado costosos.",

    "En el método de Jacobi, actualizamos todas las incógnitas simultáneamente. "
    "Para cada variable, despejamos en la ecuación correspondiente "
    "y sustituimos los valores de la iteración anterior. "
    "El resultado es que la nueva aproximación se calcula completamente separada de la anterior, "
    "lo que facilita la implementación en paralelo.",

    "Gauss-Seidel mejora a Jacobi usando los valores más recientes tan pronto como están disponibles. "
    "Cuando actualizamos la segunda variable, ya usamos el nuevo valor de la primera. "
    "Cuando actualizamos la tercera, ya tenemos los nuevos valores de la primera y la segunda. "
    "Esto hace que Gauss-Seidel converja generalmente el doble de rápido que Jacobi "
    "usando el mismo número de operaciones por iteración.",

    "La condición suficiente más común para garantizar la convergencia es la dominancia diagonal estricta. "
    "Decimos que una matriz es estrictamente diagonalmente dominante por filas si, "
    "en cada fila, el valor absoluto del elemento diagonal supera a la suma de los valores absolutos "
    "de todos los demás elementos en esa fila. "
    "Bajo esta condición, ambos métodos convergen desde cualquier punto inicial.",

    "La prueba de convergencia para sistemas diagonalmente dominantes se basa en el radio espectral. "
    "La condición de convergencia es que el radio espectral de la matriz de iteración sea estrictamente menor que uno. "
    "Para Jacobi, la matriz de iteración se obtiene tomando el negativo del cociente entre "
    "los elementos no diagonales y el elemento diagonal correspondiente. "
    "La dominancia diagonal garantiza que cada fila de esta matriz tenga norma uno en su suma de valores absolutos, "
    "lo que implica que el radio espectral es menor que uno.",
],

'lu-gaussian': [
    "La factorización L U descompone una matriz A en el producto de una matriz triangular inferior L "
    "y una matriz triangular superior U. "
    "Es el corazón de los métodos directos para resolver sistemas lineales.",

    "La eliminación gaussiana construye implícitamente la factorización L U. "
    "A medida que eliminamos elementos bajo la diagonal, guardamos los multiplicadores usados: "
    "esos multiplicadores forman exactamente la matriz L. "
    "La matriz U es la forma escalonada que obtenemos al final. "
    "Para resolver un sistema con esta factorización, hacemos dos sustituciones triangulares: "
    "primero resolvemos L por y igual a b en dirección hacia adelante, "
    "y luego resolvemos U por x igual a y en dirección hacia atrás.",

    "La gran ventaja de la factorización L U es la reutilización. "
    "Si necesitamos resolver el mismo sistema con múltiples vectores del lado derecho, "
    "calculamos la factorización una sola vez, en el orden del cubo del tamaño de la matriz, "
    "y cada solución adicional cuesta solo del orden del cuadrado. "
    "Esto es crucial en aplicaciones como el método de elementos finitos "
    "donde se resuelve el mismo sistema estructural con muchos tipos de carga.",

    "El pivoteo parcial intercambia filas durante la eliminación para usar siempre "
    "el elemento de mayor valor absoluto como pivote. "
    "Esto evita divisiones por números muy pequeños, que amplificarían los errores de redondeo. "
    "Con pivoteo parcial, la factorización es más estable numéricamente "
    "y se puede demostrar que el factor de crecimiento de los elementos es acotado.",

    "La factorización de Cholesky es una variante especial para matrices simétricas y definidas positivas. "
    "En este caso, la factorización toma la forma A igual a L por L transpuesta, "
    "donde L es triangular inferior. "
    "Es el doble de eficiente que la L U general, porque aprovecha la simetría de la matriz: "
    "solo calculamos la mitad inferior. "
    "Además, es numéricamente muy estable para este tipo de matrices, sin necesidad de pivoteo.",
],

'qr-eigenvalue': [
    "Los valores propios de una matriz describen cómo esa matriz transforma el espacio. "
    "Un vector propio es una dirección que la matriz no cambia de orientación: solo la escala. "
    "El factor de escala es el valor propio correspondiente. "
    "Son fundamentales en análisis modal, vibraciones, análisis de componentes principales "
    "y muchas otras aplicaciones.",

    "El método de la potencia calcula el valor propio de mayor magnitud. "
    "Empezamos con un vector arbitrario y lo multiplicamos repetidamente por la matriz. "
    "Después de cada multiplicación, normalizamos el vector para evitar que crezca sin control. "
    "El cociente de Rayleigh, que es el producto del vector por la matriz por el vector dividido "
    "entre la norma al cuadrado, converge al valor propio dominante. "
    "La convergencia es lineal con razón igual al cociente entre el segundo mayor valor propio "
    "y el mayor.",

    "El algoritmo Q R es el método estándar para calcular todos los valores propios de una matriz. "
    "En cada iteración, factorizamos la matriz actual en Q por R, donde Q es ortogonal y R triangular superior. "
    "Luego formamos la nueva iteración como R por Q, es decir, invertimos el orden de los factores. "
    "Este proceso, que preserva los valores propios porque es una transformación de similitud, "
    "converge a una forma casi triangular cuya diagonal contiene los valores propios.",

    "Para acelerar el algoritmo Q R, usamos deflación y desplazamientos espectrales. "
    "La deflación elimina un valor propio ya convergido y reduce el tamaño del problema. "
    "El desplazamiento espectral selecciona un valor cercano al valor propio buscado, "
    "lo que acelera enormemente la convergencia.",

    "Las matrices de Hessenberg son casi triangulares superiores, con cero solo bajo la primera subdiagonal. "
    "Reducir la matriz original a forma de Hessenberg mediante transformaciones de Householder "
    "es el primer paso del algoritmo Q R moderno. "
    "Una vez en forma de Hessenberg, cada paso Q R cuesta solo del orden del cuadrado del tamaño de la matriz "
    "en lugar del cubo.",
],

'quadrature': [
    "La integración numérica aproxima el área bajo una curva cuando no podemos calcular la integral analíticamente. "
    "Los métodos clásicos como la regla del trapecio y Simpson dividen el intervalo en partes iguales "
    "y aproximan la función por polinomios simples en cada parte.",

    "La regla del trapecio aproxima la función por una línea recta entre dos puntos consecutivos "
    "y calcula el área del trapecio resultante. "
    "El error de la regla compuesta del trapecio en n subintervalos es proporcional "
    "al cuadrado del tamaño del subintervalo, multiplicado por la segunda derivada promedio de la función. "
    "Para funciones suficientemente suaves, el error cae cuadráticamente al reducir el paso.",

    "La regla de Simpson usa tres puntos: los dos extremos y el punto medio. "
    "Ajusta una parábola a través de esos tres puntos y calcula el área exacta bajo la parábola. "
    "La fórmula combina los extremos con peso uno y el punto medio con peso cuatro, "
    "todo dividido entre tres veces el paso. "
    "El error de Simpson es proporcional a la cuarta potencia del paso, "
    "lo que la hace mucho más precisa que el trapecio para funciones suaves.",

    "La cuadratura gaussiana elige los puntos de evaluación de forma óptima, no igualmente espaciados. "
    "Con n puntos cuidadosamente elegidos, integramos exactamente cualquier polinomio de grado hasta dos n menos uno. "
    "Los puntos de Gauss son las raíces de los polinomios de Legendre, "
    "y los pesos se calculan de forma que la integral sea exacta para polinomios de máximo grado posible. "
    "Para calcular los puntos y pesos de Gauss, usamos el sistema de ecuaciones que se obtiene "
    "pidiendo exactitud para los polinomios de grado cero, uno, dos, hasta dos n menos uno.",

    "Un truco importante en la cuadratura gaussiana es el cambio de variable. "
    "Los puntos estándar de Gauss-Legendre están definidos en el intervalo de menos uno a uno. "
    "Para integrar en un intervalo arbitrario de a a b, hacemos una transformación lineal "
    "que mapea el intervalo estándar al deseado. "
    "La integral se multiplica entonces por la semilongitud del intervalo.",
],

'interpolation': [
    "La interpolación polinómica construye un polinomio que pasa exactamente por un conjunto de puntos dados. "
    "El polinomio interpolante de Lagrange usa una combinación de polinomios base especiales, "
    "donde cada polinomio base vale uno en el nodo correspondiente y cero en todos los demás. "
    "Dado n más uno puntos, el polinomio interpolante tiene grado a lo sumo n.",

    "La unicidad del polinomio interpolante es un resultado fundamental. "
    "Dado cualquier conjunto de n más uno puntos con abscisas distintas, "
    "existe exactamente un polinomio de grado a lo sumo n que pasa por todos ellos. "
    "Dos formas distintas del mismo polinomio, como la de Lagrange y la de Newton, "
    "son representaciones diferentes del mismo objeto único.",

    "El error de interpolación mide cuán bien el polinomio aproxima la función entre los nodos. "
    "La fórmula del error involucra la derivada de orden n más uno de la función, "
    "evaluada en algún punto desconocido del intervalo, "
    "multiplicada por el producto de las diferencias entre el punto de evaluación y cada uno de los nodos. "
    "Para acotar el error, acotamos la derivada de la función y el producto de diferencias por separado.",

    "Las splines cúbicas resuelven el problema de Runge: cuando usamos muchos nodos equiespaciados, "
    "el polinomio interpolante de alto grado puede oscilar salvajemente entre los nodos, "
    "especialmente cerca de los extremos del intervalo. "
    "En cambio, las splines usan polinomios cúbicos distintos en cada subintervalo "
    "y los unen imponiendo continuidad de la función y de sus dos primeras derivadas en cada nodo. "
    "El resultado es una curva suave que pasa por todos los puntos sin oscilaciones.",

    "La interpolación de Hermite es una variante que interpola no solo los valores de la función "
    "sino también los valores de la derivada en los nodos. "
    "Esto permite construir polinomios de mayor grado que aproximan mejor la función. "
    "Es la base de la cuadratura gaussiana con derivadas, conocida como cuadratura de Gauss-Hermite.",
],

'bvp': [
    "Los problemas de valor en la frontera aparecen en la física y la ingeniería "
    "cuando la solución debe satisfacer condiciones en ambos extremos de un dominio. "
    "A diferencia de los problemas de valor inicial, donde avanzamos desde un extremo, "
    "aquí necesitamos un enfoque diferente porque las condiciones están en ambos lados.",

    "El método de diferencias finitas discretiza el dominio continuo en una malla de puntos igualmente espaciados. "
    "Luego reemplaza las derivadas de la ecuación diferencial "
    "por cocientes de diferencias entre valores en puntos vecinos de la malla. "
    "El resultado es un sistema de ecuaciones algebraicas que podemos resolver. "
    "Cuanto más fina sea la malla, mayor es la precisión, pero mayor también el costo computacional.",

    "Las condiciones de frontera de Dirichlet imponen el valor de la solución en los extremos. "
    "Las de Neumann imponen el valor de la derivada normal. "
    "En diferencias finitas, las condiciones de Dirichlet se imponen directamente "
    "eliminando las ecuaciones en los nodos de frontera. "
    "Las condiciones de Neumann requieren un poco más de cuidado: "
    "se aproxima la derivada con diferencias unilaterales o simétricas.",

    "El método de disparo, o shooting, transforma el problema de valor en la frontera "
    "en una secuencia de problemas de valor inicial. "
    "Tomamos la condición en el extremo izquierdo y adivinamos el valor de la derivada allí. "
    "Luego integramos la ecuación diferencial hasta el extremo derecho y comparamos "
    "el valor calculado con la condición prescrita allí. "
    "Usamos un método de raíces, como Newton-Raphson, para ajustar la suposición inicial "
    "hasta que la condición de la frontera derecha se satisfaga.",

    "El método de elementos finitos, en su formulación variacional, transforma la ecuación diferencial "
    "en una formulación débil o integral. "
    "Multiplicamos la ecuación por una función de prueba e integramos sobre el dominio. "
    "Usando integración por partes, reducimos el orden de la derivada. "
    "Luego aproximamos la solución en un espacio de funciones de dimensión finita, "
    "típicamente polinomios por partes, y obtenemos un sistema lineal para los coeficientes.",
],

'ode-ivp': [
    "Los problemas de valor inicial para ecuaciones diferenciales ordinarias "
    "buscan la función que satisface la ecuación y parte de un valor dado en el tiempo inicial.",

    "El método de Euler explícito es el más sencillo. "
    "Dado el valor actual de la solución, calculamos su derivada usando la ecuación diferencial, "
    "y avanzamos en esa dirección un paso de tiempo. "
    "El error por paso es proporcional al cuadrado del tamaño del paso, "
    "y el error global acumulado tras integrar hasta un tiempo fijo "
    "es proporcional al tamaño del paso: convergencia de orden uno.",

    "El método de Euler implícito evalúa la derivada en el nuevo punto, no en el actual. "
    "Esto lo hace incondicionalmente estable, pero requiere resolver una ecuación en cada paso "
    "para encontrar el nuevo valor. "
    "Para ecuaciones lineales, esto es simplemente resolver un sistema lineal. "
    "Para ecuaciones no lineales, se usa Newton-Raphson en cada paso.",

    "El método de Heun, o Runge-Kutta de orden dos, mejora la precisión de Euler "
    "usando dos evaluaciones de la derivada por paso: una al inicio del intervalo y otra al final. "
    "Promediando ambas evaluaciones, el error por paso cae al cubo del tamaño del paso "
    "y el error global es del cuadrado: convergencia de orden dos.",

    "El método de Runge-Kutta de orden cuatro es el estándar en la práctica. "
    "Usa cuatro evaluaciones de la derivada por paso, en los extremos y dos puntos intermedios. "
    "Los combina con pesos uno, dos, dos, uno, divididos entre seis. "
    "El error global es de orden cuatro: si reducimos el paso a la mitad, "
    "el error cae a una dieciseisava parte. "
    "Este método tiene un excelente balance entre precisión, estabilidad y costo computacional.",

    "La estabilidad de un método numérico para ecuaciones diferenciales es crucial. "
    "Para la ecuación modelo de prueba, que es la derivada de y igual a lambda por y, "
    "el método de Euler explícito es estable solo si el producto de lambda por el paso "
    "cae dentro de un disco en el plano complejo. "
    "Si lambda tiene parte real negativa grande, el paso debe ser muy pequeño para mantener la estabilidad: "
    "este es el problema de los sistemas rígidos o stiff.",
],

'error-analysis': [
    "El análisis de error en métodos numéricos distingue entre error de redondeo "
    "y error de truncamiento. "
    "El error de redondeo proviene de representar números en punto flotante con precisión finita. "
    "El error de truncamiento proviene de aproximar procesos infinitos, "
    "como derivadas o integrales, por operaciones finitas.",

    "La aritmética de punto flotante en el estándar IEEE usa representación binaria con 64 bits. "
    "Tenemos un bit de signo, once bits para el exponente, y cincuenta y dos bits de mantisa. "
    "La precisión de máquina es aproximadamente dos punto dos por diez a la menos dieciséis. "
    "Cualquier número real se redondea al flotante más cercano, "
    "con error relativo menor que la mitad de esa precisión de máquina.",

    "La cancelación catastrófica ocurre cuando restamos dos números casi iguales. "
    "El resultado exacto es muy pequeño comparado con los operandos, "
    "pero el error de redondeo tiene la misma magnitud que el error de los operandos. "
    "Así, el error relativo del resultado puede ser mucho mayor que la precisión de máquina. "
    "Para evitar la cancelación, se reformulan las expresiones algebraicamente "
    "antes de evaluarlas numéricamente.",

    "El número de condición de un problema mide cuánto puede amplificarse el error de los datos "
    "en el error de la solución. "
    "Para un sistema lineal, el número de condición de la matriz "
    "es el producto de la norma de la matriz por la norma de su inversa. "
    "Si el número de condición es diez a la k, entonces podemos perder hasta k dígitos de precisión. "
    "Una matriz se llama mal condicionada cuando su número de condición es grande.",

    "Las diferencias finitas para aproximar derivadas tienen un error de truncamiento "
    "que depende del orden del método. "
    "La diferencia hacia adelante de primer orden tiene error proporcional al paso. "
    "La diferencia centrada de segundo orden tiene error proporcional al cuadrado del paso. "
    "Para determinar el orden de exactitud de una fórmula de diferencias finitas, "
    "expandimos cada término en serie de Taylor y vemos qué términos se cancelan.",
],

'least-squares': [
    "Los mínimos cuadrados buscan la mejor aproximación a un sistema sobredeterminado: "
    "más ecuaciones que incógnitas. "
    "En lugar de resolver el sistema exactamente, que es imposible cuando está sobredeterminado, "
    "minimizamos la suma de los cuadrados de los residuos.",

    "La solución de mínimos cuadrados satisface las ecuaciones normales: "
    "la transpuesta de la matriz por la matriz, multiplicada por el vector de solución, "
    "es igual a la transpuesta de la matriz por el vector del lado derecho. "
    "Para un sistema bien planteado, esta solución minimiza la norma al cuadrado del residuo.",

    "La factorización Q R es el método estándar para resolver problemas de mínimos cuadrados. "
    "Factorizamos la matriz de coeficientes como Q por R, "
    "donde Q es ortogonal y R es triangular superior. "
    "Multiplicar por Q transpuesta preserva las normas euclídeas, "
    "de forma que el problema de mínimos cuadrados se reduce a resolver "
    "el sistema triangular superior R por x igual a Q transpuesta por b, "
    "que es simplemente una sustitución hacia atrás.",

    "Las transformaciones de Householder construyen la factorización Q R usando reflexiones ortogonales. "
    "Cada reflexión de Householder anula todos los elementos debajo de la diagonal en una columna. "
    "Son numéricamente estables y forman la base del algoritmo Q R estándar "
    "para calcular valores propios.",

    "Las rotaciones de Givens son transformaciones ortogonales que anulan un solo elemento específico. "
    "Son útiles cuando la matriz tiene estructura especial, "
    "como cuando ya es casi triangular y queremos preservar esa estructura. "
    "Se usan también en el algoritmo Q R para matrices de Hessenberg.",
],

'default': [
    "Este problema aplica técnicas avanzadas de análisis numérico. "
    "En el análisis numérico del doctorado, los problemas típicamente involucran "
    "demostrar propiedades matemáticas de los métodos: convergencia, estabilidad, precisión. "
    "No basta con aplicar el algoritmo; hay que entender por qué funciona y cuándo puede fallar.",

    "La estrategia general para estos problemas es la siguiente. "
    "Primero identificamos el tipo de problema y el método relevante. "
    "Luego establecemos las hipótesis necesarias y verificamos que se cumplen. "
    "Después aplicamos el resultado teórico apropiado, como el teorema del punto fijo, "
    "el análisis de error de Taylor, o los criterios de convergencia espectral. "
    "Finalmente, interpretamos el resultado en el contexto del problema original.",

    "Para los problemas de demostración, es crucial citar correctamente los teoremas. "
    "El Teorema del Valor Intermedio, el Teorema del Punto Fijo de Banach, "
    "la expansión de Taylor con residuo de Lagrange, la desigualdad de Cauchy-Schwarz, "
    "son herramientas recurrentes en estos exámenes. "
    "Conocerlos bien, con sus hipótesis exactas, es fundamental.",
],
}

# Brief reminder for repeated topics within the same exam
TOPIC_BRIEF = {
    'newton': "Este problema también usa Newton-Raphson: la tangente define la siguiente aproximación, con convergencia cuadrática.",
    'fixed': "Nuevamente iteración de punto fijo: la derivada de g controla la velocidad de convergencia.",
    'bisection': "Nuevamente bisección: el intervalo se reduce a la mitad en cada paso.",
    'jacobi-gs': "Nuevamente métodos iterativos de Jacobi o Gauss-Seidel para sistemas lineales.",
    'lu-gaussian': "Nuevamente factorización L U o eliminación gaussiana para sistemas lineales.",
    'qr-eigenvalue': "Nuevamente algoritmo Q R o método de valores propios.",
    'quadrature': "Nuevamente cuadratura numérica: trapecio, Simpson, o cuadratura gaussiana.",
    'interpolation': "Nuevamente interpolación polinómica o splines.",
    'bvp': "Nuevamente problema de valor en la frontera.",
    'ode-ivp': "Nuevamente ecuación diferencial con valor inicial: Euler, Runge-Kutta.",
    'error-analysis': "Nuevamente análisis de error numérico y aritmética de punto flotante.",
    'least-squares': "Nuevamente mínimos cuadrados y factorización Q R.",
    'default': "Continuamos con otro problema de análisis numérico.",
}


def detect_topic(topic):
    t = topic.lower()
    if 'newton' in t:
        return 'newton'
    if any(x in t for x in ['mínimos cuadrados', 'minimos cuadrados', 'least squares',
                              'mínimo local', 'minimo local', 'ajuste polinomial', 'ponderados']):
        return 'least-squares'
    if any(x in t for x in ['punto fijo', 'fixed', 'steffensen', 'banach', 'contracción',
                              'contraccion', 'iteración de punto', 'orden de convergencia']):
        return 'fixed'
    if any(x in t for x in ['bisect', 'bisección', 'biseccion']):
        return 'bisection'
    if any(x in t for x in ['jacobi', 'gauss-seidel', 'seidel', 'iterativo lineal',
                              'dominancia diagonal', 'diagonalmente dominante', 'radio espectral',
                              'convergencia de jacobi', 'convergencia de métodos iterativos',
                              'convergencia del esquema']):
        return 'jacobi-gs'
    if any(x in t for x in ['lu', 'cholesky', 'gaussiana', 'eliminación', 'eliminacion',
                              'pivoteo', 'sustitución', 'gauss-jordan', 'número de condición',
                              'inversión', 'inversion']):
        return 'lu-gaussian'
    if any(x in t for x in ['qr', 'valores propios', 'valor propio', 'vectores propios',
                              'hessenberg', 'iteración de potencias', 'givens', 'householder',
                              'svd', 'descenso de gradiente', 'forma cuadrática']):
        return 'qr-eigenvalue'
    if any(x in t for x in ['cuadratura', 'simpson', 'trapecio', 'integr',
                              'gauss-legendre', 'gauss-radau', 'nodo', 'peso', 'riemann']):
        return 'quadrature'
    if any(x in t for x in ['interpol', 'lagrange', 'hermite', 'vandermonde', 'spline',
                              'fenómeno de runge', 'nodos equiespaciados']):
        return 'interpolation'
    if any(x in t for x in ['bvp', 'frontera', 'boundary', 'shooting', 'disparo',
                              'fem', 'elementos finitos', 'neumann', 'dirichlet',
                              'formulación variacional', 'forma variacional', 'elíptic']):
        return 'bvp'
    if any(x in t for x in ['euler', 'heun', 'runge', 'kutta', 'rk', 'pvi', 'ivp',
                              'edo', 'ode', 'valor inicial', 'estabilidad de euler',
                              'método de un paso', 'convergencia de métodos de un paso']):
        return 'ode-ivp'
    if any(x in t for x in ['error de redondeo', 'punto flotante', 'ieee', 'cancelación',
                              'cancelacion', 'error relativo', 'aritmética', 'aritmetica',
                              'precisión', 'diferencia finita', 'orden de exactitud']):
        return 'error-analysis'
    return 'default'


# ─── Clean text for TTS ────────────────────────────────────────────────────

def strip_latex(text):
    text = re.sub(r'\$\$[\s\S]*?\$\$', ' ', text)
    text = re.sub(r'\$[^$\n]{0,4}\$', ' ', text)
    text = re.sub(r'\$[^$\n]+\$', ' la expresión ', text)
    text = re.sub(r'\\[a-zA-Z]+\{[^{}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'[\$\{\}\\]', '', text)
    text = re.sub(r'\^[^\s]', '', text)
    text = re.sub(r'_[^\s]', '', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'\s*,\s*,', ',', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r',\s*\.', '.', text)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_for_tts(text):
    text = text.replace('\x00', '').replace('\r', ' ')
    text = re.sub(r'-{2,}', ', ', text)
    text = re.sub(r"''+", ' primo ', text)
    text = re.sub(r"'", ' ', text)
    text = re.sub(r'[*_`#]', '', text)
    text = re.sub(r'[<>]', ' ', text)
    text = re.sub(r'[|\\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    if text.startswith('-'):
        text = 'Resultado: ' + text.lstrip('-').strip()
    return text


def is_meaningful_sentence(text):
    if len(text) < 20:
        return False
    words = re.findall(r'\b[a-záéíóúñü]{3,}\b', text, re.IGNORECASE)
    if len(words) < 4:
        return False
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha_ratio < 0.4:
        return False
    return True


def read_table_rows(rows):
    if len(rows) < 3:
        return ['La tabla muestra los resultados numéricos del método.']

    header = [strip_latex(c.strip()) for c in rows[0].strip('|').split('|')]
    data_rows = rows[2:]
    spoken = []

    n = len(data_rows)
    if n == 0:
        return ['La tabla no contiene datos.']

    spoken.append(f'La tabla de resultados tiene {n} filas de datos.')

    read_indices = list(range(min(5, n)))
    if n > 5:
        read_indices.append(n - 1)

    for i in read_indices:
        row = data_rows[i]
        cells = [clean_for_tts(strip_latex(c.strip())) for c in row.strip('|').split('|')]
        cells = [c for c in cells if c and c not in ('—', '-', '')]

        parts = []
        for h, v in zip(header, cells):
            h_clean = clean_for_tts(h)
            if h_clean and v:
                parts.append(f'{h_clean} igual a {v}')

        label = f'Iteración {i}' if i < n - 1 or n <= 5 else f'Última iteración, fila {n}'
        if parts:
            spoken.append(f'{label}: {", ".join(parts)}.')

    if n > 6:
        spoken.append(f'El proceso continúa con {n - len(read_indices)} iteraciones adicionales hasta la convergencia.')

    return spoken


# ─── Problem script builder ───────────────────────────────────────────────

TOPIC_RE = re.compile(r'^## Problema\s+(\d+)\s*(?:[—\-]+)?\s*(.*)', re.MULTILINE)


def build_problem_script(prob_md, key, label, prob_num, enunciado_full, seen_topics):
    m = TOPIC_RE.match(prob_md.strip())
    topic = m.group(2).strip() if m else ''
    topic_key = detect_topic(topic)

    script = []

    # ── 1. Header ──
    script.append(
        f"Examen de Análisis Numérico, George Mason University. {label}. "
        f"Problema número {prob_num}."
    )
    if topic:
        clean_topic = clean_for_tts(strip_latex(topic))
        if clean_topic and len(clean_topic) > 3:
            script.append(f"El tema de este problema es: {clean_topic}.")

    # ── 2. Topic explanation (full first time, brief on repeat) ──
    if topic_key not in seen_topics:
        script.append("Antes de ver la solución, repasemos el concepto central.")
        for para in TOPIC_SCRIPTS.get(topic_key, TOPIC_SCRIPTS['default']):
            script.append(para)
        seen_topics.add(topic_key)
    else:
        brief = TOPIC_BRIEF.get(topic_key, "Continuamos con otro problema de análisis numérico.")
        script.append(brief)

    # ── 3. Problem statement ──
    script.append("Ahora el enunciado específico de este problema.")
    enu_text = ''
    if enunciado_full:
        parts = re.split(r'(?=\bProblem\s+\d+\b)', enunciado_full, flags=re.IGNORECASE)
        if len(parts) > prob_num:
            chunk = parts[prob_num]
            chunk = re.sub(r'^\s*Problem\s+\d+[^\n]*\n?', '', chunk, flags=re.IGNORECASE)
            enu_text = strip_latex(chunk.strip())[:600]
    if enu_text:
        enu_text = clean_for_tts(enu_text)
        if is_meaningful_sentence(enu_text):
            script.append("El enunciado dice: " + enu_text)

    script.append("Ahora vamos con la solución, paso a paso.")

    # ── 4. Parse solution for readable content ──
    lines = prob_md.split('\n')
    in_code = False
    in_display_math = False
    in_table = False
    table_rows = []
    text_buffer = []

    def flush_text():
        if text_buffer:
            joined = ' '.join(text_buffer).strip()
            joined = clean_for_tts(joined)
            if is_meaningful_sentence(joined):
                script.append(joined)
            text_buffer.clear()

    def flush_table():
        nonlocal table_rows, in_table
        if table_rows:
            for s in read_table_rows(table_rows):
                script.append(s)
        table_rows.clear()
        in_table = False

    skip_first_heading = True

    for line in lines:
        s = line.strip()

        if s.startswith('```'):
            in_code = not in_code
            flush_text()
            continue
        if in_code:
            continue

        if s.startswith('$$'):
            if not in_display_math:
                flush_text()
                flush_table()
                in_display_math = True
                rest = s[2:]
                if '$$' in rest:
                    in_display_math = False
                continue
            else:
                in_display_math = False
                continue
        if in_display_math:
            continue

        if s.startswith('|'):
            flush_text()
            in_table = True
            table_rows.append(s)
            continue
        elif in_table:
            flush_table()

        if not s:
            flush_text()
            continue

        if s.startswith('#'):
            flush_text()
            flush_table()
            heading = re.sub(r'^#+\s+', '', s)
            if skip_first_heading and re.match(r'^Problema\s+\d+', heading, re.IGNORECASE):
                skip_first_heading = False
                continue
            heading_clean = clean_for_tts(strip_latex(heading))
            if heading_clean and is_meaningful_sentence(heading_clean):
                script.append(heading_clean + '.')
            continue

        m2 = re.match(r'^\*\*\(([a-z])\)\*\*\s*(.*)', s)
        if m2:
            flush_text()
            letter = m2.group(1)
            rest_text = clean_for_tts(strip_latex(m2.group(2).strip()))
            intro = f"Apartado {letter}."
            if rest_text and is_meaningful_sentence(rest_text):
                intro += ' ' + rest_text
            script.append(intro)
            continue

        m2 = re.match(r'^(\d+)\.\s+(.*)', s)
        if m2:
            flush_text()
            item = clean_for_tts(strip_latex(m2.group(2)))
            if is_meaningful_sentence(item):
                script.append(f"Paso {m2.group(1)}: {item}")
            continue

        m2 = re.match(r'^[-*]\s+(.*)', s)
        if m2:
            flush_text()
            item = clean_for_tts(strip_latex(m2.group(1)))
            if is_meaningful_sentence(item):
                script.append(item + '.')
            continue

        cleaned = clean_for_tts(strip_latex(s))
        if cleaned and is_meaningful_sentence(cleaned):
            text_buffer.append(cleaned)

    flush_text()
    flush_table()

    # ── 5. Outro ──
    script.append(
        f"Con esto concluye el problema {prob_num}. "
        "Es importante dominar tanto la teoría como la implementación numérica de estos métodos. "
        "En el examen, demuestran claridad conceptual al explicar por qué el método funciona, "
        "no solo al aplicar la fórmula."
    )

    return script


# ─── Split problems ───────────────────────────────────────────────────────

def split_problems(md):
    parts = re.split(r'(?=^## Problema\s+\d+)', md, flags=re.MULTILINE)
    return [p for p in parts if p.strip().startswith('## Problema')]


# ─── Generate exam MP3 ─────────────────────────────────────────────────────

SILENCE_WAV = None

def get_silence():
    global SILENCE_WAV
    if SILENCE_WAV and os.path.exists(SILENCE_WAV):
        return SILENCE_WAV
    SILENCE_WAV = '/tmp/silence_kokoro.wav'
    make_silence(SILENCE_WAV, 0.55)
    return SILENCE_WAV


def generate_exam_mp3(key, label, problems, enunciado_full, out_path):
    silence = get_silence()
    all_wavs = []
    tmp_wavs = []
    seen_topics = set()

    try:
        for i, prob_md in enumerate(problems):
            n = i + 1
            script = build_problem_script(prob_md, key, label, n, enunciado_full, seen_topics)
            for j, para in enumerate(script):
                pav = f'/tmp/kok_{key}_p{n:02d}_s{j:03d}.wav'
                kokoro_speak(para, pav)
                tmp_wavs.append(pav)
                all_wavs.append(pav)
                all_wavs.append(silence)
            # Extra pause between problems
            all_wavs.append(silence)
            all_wavs.append(silence)

        if all_wavs:
            combined = f'/tmp/kok_{key}_combined.wav'
            enhanced = f'/tmp/kok_{key}_enhanced.wav'
            subprocess.run(['sox'] + all_wavs + [combined], capture_output=True)
            if os.path.exists(combined):
                subprocess.run([
                    'sox', combined, enhanced,
                    'rate', '44100',
                    'norm', '-10',
                    'equalizer', '220', '0.8q', '+2',
                    'equalizer', '700', '1q', '+1',
                    'equalizer', '5500', '1q', '-2',
                    'reverb', '10', '40', '60', '0', '0', '0',
                    'norm', '-3'
                ], capture_output=True)
                src = enhanced if os.path.exists(enhanced) else combined
                subprocess.run(
                    ['lame', '--quiet', '-q', '2', '-b', '128', '--resample', '44.1',
                     src, out_path],
                    check=True, capture_output=True
                )
                for f in (combined, enhanced):
                    if os.path.exists(f):
                        os.unlink(f)
    finally:
        for w in tmp_wavs:
            if os.path.exists(w):
                os.unlink(w)


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    # Pre-load kokoro once
    print('Loading kokoro neural TTS model...', flush=True)
    get_kokoro()
    print('Model loaded.', flush=True)

    md_files = sorted(glob.glob(os.path.join(SOLUTIONS_DIR, '*.md')))
    for md_file in md_files:
        key = os.path.basename(md_file).replace('.md', '')
        label = LABELS.get(key, key)
        out_path = os.path.join(OUT_DIR, f'{key}.mp3')
        md = open(md_file, encoding='utf-8').read()
        problems = split_problems(md)
        enunciado_full = ENUNCIADOS.get(key, '')
        print(f'[{key}] {len(problems)} problemas...', end=' ', flush=True)
        generate_exam_mp3(key, label, problems, enunciado_full, out_path)
        size = os.path.getsize(out_path) // (1024 * 1024)
        print(f'✓ {size}MB')

    print(f'\nDone → {OUT_DIR}')


if __name__ == '__main__':
    main()
