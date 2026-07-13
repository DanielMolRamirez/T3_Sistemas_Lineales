import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def procesar_ecualizador(x, fs, ganancias_db):
    """
    Ecualizador IIR de 6 bandas.
    Arquitectura paralela: a 0 dB la señal es perfectamente transparente y z[n] = y[n].
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
    
    # CAMBIO CLAVE: Empezamos con una copia exacta de la señal de entrada, 
    # conservando todo el DC Offset y la forma original.
    y_eq = np.copy(x)
    
    for i in range(len(bandas)):
        f_low, f_high = bandas[i]
        
        # Pasamos la ganancia de decibelios a escala lineal
        ganancia_lineal = 10 ** (ganancias_db[i] / 20.0) 
        
        # Si la ganancia es 0 dB (lineal = 1.0), ignoramos el filtro para ahorrar CPU 
        # y garantizar que no haya ninguna alteracion de fase o amplitud.
        if np.isclose(ganancia_lineal, 1.0):
            continue
            
        # Proteccion contra aliasing: si la banda pide frecuencias por encima de Nyquist, la saltamos
        if f_low >= nyquist:
            continue 
        f_high = min(f_high, nyquist - 1)
        
        # Scipy pide las frecuencias normalizadas de 0 a 1
        Wn = [f_low / nyquist, f_high / nyquist]
        
        sos = sig.butter(4, Wn, btype='bandpass', output='sos')
        y_filtrado = sig.sosfiltfilt(sos, x)
        
        # CAMBIO CLAVE: En lugar de sumar el filtro completo, sumamos solo la DIFERENCIA.
        # Si ganancia_lineal es 1.5, sumamos 0.5 veces el filtro.
        # Si ganancia_lineal es 0.5, restamos 0.5 veces el filtro.
        y_eq += y_filtrado * (ganancia_lineal - 1.0)
        
    return y_eq


def cambiar_tasa(x, L, M):
    """
    Implementacion estricta del remuestreo racional (Expansion L, Decimacion M).
    """
    # Si el profe pone 1 y 1, es la misma senal, evitamos hacer calculo en vano
    if L == 1 and M == 1:
        return x

    # 1. EXPANSION (Upsampling)
    # Segun la teoria, creamos un vector L veces mas grande y ponemos las muestras de x cada L ceros.
    x_up = np.zeros(len(x) * L)
    x_up[::L] = x
    
    # 2. FILTRO DE TASA MULTIPLE (Interpolacion + Anti-alias)
    # La frecuencia de corte wc debe ser estricatamente el minimo entre pi/L y pi/M.
    # Como Scipy normaliza donde 1.0 es pi, hacemos 1 / max(L, M)
    f_corte_norm = 1.0 / max(L, M)
    
    # Limite de seguridad por si L y M son 1 (aunque ya lo atajamos arriba)
    if f_corte_norm >= 1.0:
        f_corte_norm = 0.999
    
    # Disenamos un filtro FIR pasabajo usando firwin (usa ventana de Hamming por defecto)
    num_taps = 101
    h = sig.firwin(num_taps, f_corte_norm) 
    
    # Multiplicamos los coeficientes por L. 
    # Esto compensa la energia perdida al haber metido (L-1) ceros entre cada muestra.
    h = h * L 
    
    # Aplicamos el filtro FIR a la senal expandida
    x_filt = sig.filtfilt(h, [1.0], x_up)
    
    # 3. DECIMACION (Downsampling)
    # Tomamos directamente 1 de cada M muestras de la senal filtrada
    y = x_filt[::M]
    
    return y

def generar_graficas_tiempo(x, y, z, modo="micro"):
    """
    Dibuja 3 subplots para evidenciar los cambios en x[n], y[n] y z[n].
    """
    # 3 filas, 1 columna para ver el flujo completo
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
    
    if modo == "Vista Micro (Muestras discretas stem)":
        # Base de muestras para la señal original
        muestras_base_x = 100
        
        # Calculamos dinámicamente cuántas muestras corresponden a la señal remuestreada.
        # len(y) / len(x) representa exactamente el factor L/M aplicado.
        if len(x) > 0:
            factor_remuestreo = len(y) / len(x)
        else:
            factor_remuestreo = 1.0
            
        muestras_equivalentes_y = int(muestras_base_x * factor_remuestreo)
        
        # Extraemos las muestras dinámicas
        n_x = np.arange(min(muestras_base_x, len(x)))
        n_y = np.arange(min(muestras_equivalentes_y, len(y)))
        n_z = np.arange(min(muestras_equivalentes_y, len(z)))
        
        # Graficamos secuencias discretas
        ax1.stem(n_x, x[:len(n_x)], basefmt=" ")
        ax2.stem(n_y, y[:len(n_y)], linefmt='orange', markerfmt='D', basefmt=" ")
        ax3.stem(n_z, z[:len(n_z)], linefmt='green', markerfmt='s', basefmt=" ")
        
        # Títulos dinámicos que muestran el número real de muestras impresas
        ax1.set_title(f"x[n]: Entrada Original (Primeras {len(n_x)} muestras)")
        ax2.set_title(f"y[n]: Tras Cambio de Tasa L/M (Primeras {len(n_y)} muestras)")
        ax3.set_title(f"z[n]: Tras Ecualizacion (Primeras {len(n_z)} muestras)")
    else:
        # Envolventes continuas para ver la amplitud general
        n_x = np.arange(len(x))
        n_y = np.arange(len(y))
        n_z = np.arange(len(z))
        
        ax1.plot(n_x, x)
        ax2.plot(n_y, y, color='orange')
        ax3.plot(n_z, z, color='green')
        
        ax1.set_title("x[n]: Envolvente Original")
        ax2.set_title("y[n]: Envolvente Tras Remuestreo")
        ax3.set_title("z[n]: Envolvente Ecualizada")

    # Etiquetado formal de los ejes en el dominio discreto
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("n (Muestras)")
        ax.set_ylabel("Amplitud")
        ax.grid(True)
    
    plt.tight_layout()
    return fig

def generar_graficas_frecuencia(x, y, z):
    """
    Calcula la FFT y grafica los espectros de magnitud de las 3 senales.
    Usa raw strings de Python (r"") para renderizar las formulas en LaTeX correctamente.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    # N = 4096 es potencia de 2. Esto fuerza a numpy a usar Radix-2 FFT, 
    # haciendo el calculo muchisimo mas rapido para no colgar el servidor.
    N = 4096 
    
    # Transformadas de Fourier. Usamos fftshift para centrar el espectro en Omega = 0
    X_mag = np.abs(np.fft.fftshift(np.fft.fft(x, N)))
    Y_mag = np.abs(np.fft.fftshift(np.fft.fft(y, N)))
    Z_mag = np.abs(np.fft.fftshift(np.fft.fft(z, N)))
    
    # Creamos el vector de frecuencia digital Omega desde -pi hasta pi
    w = np.linspace(-np.pi, np.pi, N)
    
    ax1.plot(w, X_mag)
    ax1.set_title("Espectro de x[n]")
    # Usamos r"" para que Matplotlib entienda la barra invertida y dibuje Omega mayuscula
    ax1.set_ylabel(r"$|X(e^{j\Omega})|$")
    
    ax2.plot(w, Y_mag, color='orange')
    ax2.set_title("Espectro de y[n] (Efecto del antialiasing y remuestreo)")
    ax2.set_ylabel(r"$|Y(e^{j\Omega})|$")
    
    ax3.plot(w, Z_mag, color='green')
    ax3.set_title("Espectro de z[n] (Efecto del ecualizador)")
    ax3.set_ylabel(r"$|Z(e^{j\Omega})|$")
    
    for ax in [ax1, ax2, ax3]:
        # Forzamos los limites estrictos de -pi a pi
        ax.set_xlim([-np.pi, np.pi])
        # Reemplazamos los numeros decimales por marcas exactas de pi
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        ax.set_xlabel(r"Frecuencia Digital Normalizada $\Omega$ (rad/muestra)")
        ax.grid(True)

    plt.tight_layout()
    return fig