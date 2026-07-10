import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def procesar_ecualizador(x, fs, ganancias_db):
    """
    Ecualizador de 6 bandas en paralelo.
    Se separa la señal en bandas, se multiplica por su ganancia lineal y se suma.
    """
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
    
    for i in range(len(bandas)):
        f_low, f_high = bandas[i]
        ganancia_lineal = 10 ** (ganancias_db[i] / 20.0) 
        
        if f_low >= nyquist:
            continue 
        f_high = min(f_high, nyquist - 1)
        
        Wn = [f_low / nyquist, f_high / nyquist]
        b, a = sig.butter(4, Wn, btype='bandpass')
        
        y_filtrado = sig.filtfilt(b, a, x)
        y_eq += y_filtrado * ganancia_lineal
        
    return y_eq

def cambiar_tasa(x, L, M):
    """
    Remuestreo fraccional (L/M).
    Aplica expansión, filtro pasabajo combinado y decimación.
    """
    # Si no hay cambio de tasa, devolvemos la señal intacta para ahorrar proceso y evitar errores
    if L == 1 and M == 1:
        return x

    # 1. EXPANSION (Upsampling por L)
    x_up = np.zeros(len(x) * L)
    x_up[::L] = x
    
    # 2. FILTRADO (Filtro anti-alias y de interpolacion combinado)
    f_corte_norm = 1.0 / max(L, M)
    
    # Prevención del ValueError: La frecuencia de corte debe ser estrictamente menor a Nyquist (1.0)
    if f_corte_norm >= 1.0:
        f_corte_norm = 0.999
    
    num_taps = 101
    h = sig.firwin(num_taps, f_corte_norm) 
    h = h * L 
    
    x_filt = sig.filtfilt(h, [1.0], x_up)
    
    # 3. DECIMACION (Downsampling por M)
    y = x_filt[::M]
    
    return y

def generar_graficas_tiempo(x, y, modo="micro"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    if modo == "Vista Micro (Muestras discretas stem)":
        n_x = np.arange(min(100, len(x)))
        n_y = np.arange(min(100, len(y)))
        
        ax1.stem(n_x, x[:len(n_x)], basefmt=" ")
        ax2.stem(n_y, y[:len(n_y)], linefmt='orange', markerfmt='D', basefmt=" ")
        
        ax1.set_title("Señal Original x[n] (Primeras 100 muestras)")
        ax2.set_title("Señal Procesada y[n] (Primeras 100 muestras)")
    else:
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    N = 4096 
    
    X = np.fft.fft(x, N)
    X_mag = np.abs(np.fft.fftshift(X)) 
    
    Y = np.fft.fft(y, N)
    Y_mag = np.abs(np.fft.fftshift(Y))
    
    w = np.linspace(-np.pi, np.pi, N)
    
    ax1.plot(w, X_mag)
    ax1.set_title("Espectro de Magnitud de x[n]")
    ax1.set_ylabel("|X(e^{j\Omega})|")
    
    ax2.plot(w, Y_mag, color='orange')
    ax2.set_title("Espectro de Magnitud de y[n]")
    ax2.set_ylabel("|Y(e^{j\Omega})|")
    
    for ax in [ax1, ax2]:
        ax.set_xlim([-np.pi, np.pi])
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels(['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
        ax.set_xlabel("Frecuencia digital normalizada $\Omega$ (radianes/muestra)")
        ax.grid(True)

    plt.tight_layout()
    return fig