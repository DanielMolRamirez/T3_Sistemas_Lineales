import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def procesar_ecualizador(x, fs, ganancias_db):
    """
    Ecualizador de 6 bandas en paralelo.
    Se separa la señal en bandas, se multiplica por su ganancia lineal y se suma.
    """
    # Lista con las bandas de frecuencia requeridas en la tarea
    bandas = [
        (16, 60),       # Sub-Bass
        (60, 250),      # Bass
        (250, 2000),    # Low Mids
        (2000, 4000),   # High Mids
        (4000, 6000),   # Presence
        (6000, 16000)   # Brilliance
    ]
    
    nyquist = fs / 2.0
    y_eq = np.zeros_like(x, dtype=np.float64)
    
    # Procesamos cada banda por separado
    for i in range(len(bandas)):
        f_low, f_high = bandas[i]
        ganancia_lineal = 10 ** (ganancias_db[i] / 20.0) # conversion de dB a escala lineal
        
        # Ojo aqui: si la f_high se pasa del limite de nyquist (por ejemplo si metemos un audio de 8kHz),
        # el filtro explota. Le ponemos un limite de seguridad.
        if f_low >= nyquist:
            continue # ignoramos esta banda si esta fuera de la capacidad del audio
        f_high = min(f_high, nyquist - 1)
        
        # Diseño del filtro pasabanda Butterworth (orden 4 es mas que suficiente para audio)
        # Dividimos las frecuencias entre nyquist para normalizarlas de 0 a 1 como pide scipy
        Wn = [f_low / nyquist, f_high / nyquist]
        b, a = sig.butter(4, Wn, btype='bandpass')
        
        # Usamos filtfilt en lugar de lfilter para tener fase cero y que no se nos distorsione el audio
        y_filtrado = sig.filtfilt(b, a, x)
        
        # Sumamos la contribucion de esta banda ya con su respectiva ganancia
        y_eq += y_filtrado * ganancia_lineal
        
    return y_eq

def cambiar_tasa(x, L, M):
    """
    Remuestreo fraccional (L/M).
    Aplica expansion, filtro pasabajo combinado y decimacion.
    """
    # 1. EXPANSION (Upsampling por L)
    # Llenamos un vector de ceros y metemos las muestras de x cada L espacios
    x_up = np.zeros(len(x) * L)
    x_up[::L] = x
    
    # 2. FILTRADO (Filtro anti-alias y de interpolacion combinado)
    # La teoria dice que la frecuencia de corte wc debe ser el minimo entre pi/L y pi/M
    # En scipy, la frecuencia normalizada va de 0 a 1 (donde 1 es pi). Por tanto:
    f_corte_norm = 1.0 / max(L, M)
    
    # Diseñamos un filtro FIR de orden 100 usando la ventana de Hamming (default en firwin)
    num_taps = 101
    h = sig.firwin(num_taps, f_corte_norm) 
    
    # Multiplicamos por L para compensar la perdida de energia (amplitud) al insertar tantos ceros
    h = h * L 
    
    # Aplicamos el filtro a la senal expandida
    x_filt = sig.filtfilt(h, [1.0], x_up)
    
    # 3. DECIMACION (Downsampling por M)
    # Simplemente agarramos una de cada M muestras de la senal ya filtrada
    y = x_filt[::M]
    
    return y

def generar_graficas_tiempo(x, y, modo="micro"):
    """
    Genera la grafica en el dominio del tiempo (n).
    Modo macro es la linea continua (para ver envolvente).
    Modo micro es el stem (para ver muestras discretas a detalle).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    if modo == "Vista Micro (Muestras discretas stem)":
        # Agarramos solo las primeras 100 muestras para que se logre ver bien
        n_x = np.arange(min(100, len(x)))
        n_y = np.arange(min(100, len(y)))
        
        # Dibujamos con stem para que el profe vea las secuencias en tiempo discreto x[n]
        ax1.stem(n_x, x[:len(n_x)], basefmt=" ")
        ax2.stem(n_y, y[:len(n_y)], linefmt='orange', markerfmt='D', basefmt=" ")
        
        ax1.set_title("Señal Original x[n] (Primeras 100 muestras)")
        ax2.set_title("Señal Procesada y[n] (Primeras 100 muestras)")
    else:
        # Modo macro: mostramos todo como linea continua
        n_x = np.arange(len(x))
        n_y = np.arange(len(y))
        
        ax1.plot(n_x, x)
        ax2.plot(n_y, y, color='orange')
        
        ax1.set_title("Envolvente de la Señal Original")
        ax2.set_title("Envolvente de la Señal Procesada")

    ax1.set_xlabel("n (Muestras)")
    ax1.set_ylabel("Amplitud")
    ax1.grid(True)
    
    ax2.set_xlabel("n (Muestras)")
    ax2.set_ylabel("Amplitud")
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def generar_graficas_frecuencia(x, y):
    """
    Grafica la transformada de fourier. 
    Se centra en cero y el eje X se normaliza de -pi a pi.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # Numero de puntos para la FFT (potencia de 2 para mayor velocidad)
    N = 4096 
    
    # Calculo de la FFT de x
    X = np.fft.fft(x, N)
    X_mag = np.abs(np.fft.fftshift(X)) # Usamos fftshift para centrar en cero
    
    # Calculo de la FFT de y
    Y = np.fft.fft(y, N)
    Y_mag = np.abs(np.fft.fftshift(Y))
    
    # Creamos el eje de frecuencias omega (w) que va de -pi a pi
    w = np.linspace(-np.pi, np.pi, N)
    
    ax1.plot(w, X_mag)
    ax1.set_title("Espectro de Magnitud de x[n]")
    ax1.set_ylabel("|X(e^{j\Omega})|")
    
    ax2.plot(w, Y_mag, color='orange')
    ax2.set_title("Espectro de Magnitud de y[n]")
    ax2.set_ylabel("|Y(e^{j\Omega})|")
    
    # Ponemos las etiquetas del eje X en terminos de Pi para mayor rigor teorico
    for ax in [ax1, ax2]:
        ax.set_xlim([-np.pi, np.pi])
        # Configuramos los ticks manualmente para que digan -pi, -pi/2, 0, pi/2, pi
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels(['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
        ax.set_xlabel("Frecuencia digital normalizada $\Omega$ (radianes/muestra)")
        ax.grid(True)

    plt.tight_layout()
    return fig