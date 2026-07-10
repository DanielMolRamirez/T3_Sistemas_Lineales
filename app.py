import streamlit as st
import numpy as np
from scipy.io import wavfile
import os
import dsp_core as dsp 

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="DSP Audio - Tarea Oppenheim", layout="wide")

st.title("Conversor de Frecuencia de Muestreo y Ecualizador")
st.markdown("**Desarrollado por:** Daniel Molina, Santiago Zumba y Juan Pacheco")
st.markdown("### Universidad de Cuenca - Sistemas Lineales y Señales")

# --- PANEL DE CONTROL LATERAL ---
with st.sidebar:
    st.header("1. Carga de Audio")
    
    # Buscar archivos dentro de la carpeta local 'audio_files'
    carpeta_audios = "audio_files"
    archivos_repo = []
    if os.path.exists(carpeta_audios):
        archivos_repo = [f for f in os.listdir(carpeta_audios) if f.endswith(".wav")]
    
    # Desplegable para seleccionar audios pre-cargados
    opcion_archivo = st.selectbox(
        "Selecciona un audio del repositorio:", 
        ["-- Subir un archivo nuevo --"] + archivos_repo
    )
    
    # Si elige subir un archivo nuevo, mostramos el uploader
    if opcion_archivo == "-- Subir un archivo nuevo --":
        archivo_a_procesar = st.file_uploader("O sube tu propio archivo .wav", type=["wav"])
    else:
        # Si elige uno del repo, armamos la ruta local
        archivo_a_procesar = os.path.join(carpeta_audios, opcion_archivo)
    
    st.divider()
    
    st.header("2. Análisis por Ventanas")
    st.caption("Selecciona el segmento de la canción a analizar (en segundos)")
    inicio_seg, fin_seg = st.slider("Ventana de tiempo:", 0.0, 100.0, (0.0, 10.0), step=0.5)

    st.divider()
    
    st.header("3. Conversión de Tasa (L/M)")
    st.latex(r"y[n] = x_{(\uparrow L \downarrow M)}[n]")
    col1, col2 = st.columns(2)
    with col1:
        factor_L = st.number_input("Expansión (L):", min_value=1, value=1, step=1)
    with col2:
        factor_M = st.number_input("Decimación (M):", min_value=1, value=1, step=1)
        
    st.divider()
    
    st.header("4. Ecualizador (Ganancia dB)")
    sub_bass = st.slider("Sub-Bass (16 - 60 Hz)", -12.0, 12.0, 0.0)
    bass = st.slider("Bass (60 - 250 Hz)", -12.0, 12.0, 0.0)
    low_mids = st.slider("Low Mids (250 - 2k Hz)", -12.0, 12.0, 0.0)
    high_mids = st.slider("High Mids (2k - 4k Hz)", -12.0, 12.0, 0.0)
    presence = st.slider("Presence (4k - 6k Hz)", -12.0, 12.0, 0.0)
    brilliance = st.slider("Brilliance (6k - 16k Hz)", -12.0, 12.0, 0.0)

# --- PROCESAMIENTO Y VISTA PRINCIPAL ---
if archivo_a_procesar is not None:
    try:
        # scipy.io.wavfile.read acepta tanto rutas de archivos (strings) como archivos subidos en streamlit
        fs_original, audio_data = wavfile.read(archivo_a_procesar)
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
            
        duracion_total = len(audio_data) / fs_original
        
        muestra_inicio = int(inicio_seg * fs_original)
        muestra_fin = int(min(fin_seg, duracion_total) * fs_original) 
        x_n = audio_data[muestra_inicio:muestra_fin]
        
        st.subheader("🎵 Audio Original (Ventana Seleccionada)")
        st.write(f"**Frecuencia de muestreo original ($F_s$):** {fs_original} Hz")
        st.audio(x_n, sample_rate=fs_original)
        
        st.divider()

        ganancias = [sub_bass, bass, low_mids, high_mids, presence, brilliance]
        
        x_ecualizada = dsp.procesar_ecualizador(x_n, fs_original, ganancias)
        y_n = dsp.cambiar_tasa(x_ecualizada, factor_L, factor_M)
        
        fs_nueva = int(fs_original * factor_L / factor_M)
        
        if np.max(np.abs(y_n)) > 0:
            y_n = y_n / np.max(np.abs(y_n)) * 32767
        y_n = y_n.astype(np.int16)
        
        st.subheader("🎧 Audio Procesado (Ecualizado y Remuestreado)")
        st.write(f"**Nueva Frecuencia de muestreo absoluta:** {fs_nueva} Hz")
        st.audio(y_n, sample_rate=fs_nueva)
        
        st.divider()

        tab_tiempo, tab_frecuencia = st.tabs(["⏱️ Dominio del Tiempo", "📊 Dominio de la Frecuencia"])
        
        with tab_tiempo:
            st.header("Análisis en el Dominio del Tiempo")
            vista_tiempo = st.radio("Tipo de visualización:", ["Vista Macro (Envolvente continua)", "Vista Micro (Muestras discretas stem)"], horizontal=True)
            
            fig_tiempo = dsp.generar_graficas_tiempo(x_n, y_n, modo=vista_tiempo)
            st.pyplot(fig_tiempo)
            
            with st.expander("📚 Ver Teoría: Muestreo y Reconstrucción en el Tiempo"):
                st.latex(r"x_{(\downarrow M)}[n] = x[nM]")
                st.latex(r"x_{(\uparrow L)}[n] = \begin{cases} x[n/L], & \text{si } n/L \text{ es entero} \\ 0, & \text{si } n/L \text{ no es entero} \end{cases}")
                st.write("Al realizar conversión fraccional L/M, la señal pasa por un único filtro pasabajo de tasa múltiple que actúa simultáneamente como filtro de interpolación (para evitar imágenes espectrales al expandir) y filtro anti-aliasing (para evitar solapamiento al diezmar).")

        with tab_frecuencia:
            st.header("Análisis en el Dominio de la Frecuencia (Espectro)")
            
            fig_frec = dsp.generar_graficas_frecuencia(x_n, y_n)
            st.pyplot(fig_frec)
            
            with st.expander("📚 Ver Teoría: Efecto de L y M en el Espectro"):
                st.write("Frecuencia de corte del filtro combinado:")
                st.latex(r"\omega_c = \min\left(\frac{\pi}{L}, \frac{\pi}{M}\right)")
                st.write("Al diezmar, el espectro se expande por un factor M. Al interpolar (expandir), el espectro se comprime por un factor L. Las bandas del ecualizador alteran la magnitud en regiones específicas de la transformada, lo cual se evidencia en la densidad espectral de la señal procesada.")
                
    except Exception as e:
        st.error(f"Error al leer el archivo de audio. Asegúrate de que sea un WAV válido y sin compresión. Detalles: {e}")

else:
    st.info("👈 Por favor, selecciona o sube un archivo .wav en el panel izquierdo para comenzar el análisis.")