import streamlit as st
import numpy as np
from scipy.io import wavfile
import os
import dsp_core as dsp 

# --- CONFIGURACION INICIAL ---
st.set_page_config(page_title="DSP Audio - Proyecto Sistemas Lineales", layout="wide")

st.title("Conversor de Frecuencia de Muestreo y Ecualizador")
st.markdown("**Desarrollado por:** Daniel Molina, Santiago Zumba y Juan Pacheco")
st.markdown("### Universidad de Cuenca - Sistemas Lineales y Señales")

# --- PANEL LATERAL (CONTROLES) ---
with st.sidebar:
    st.header("1. Seleccion de la Señal x[n]")
    
    # Leemos la carpeta local para que el profesor pueda elegir rapidamente audios ya probados
    carpeta_audios = "audio_files"
    archivos_repo = []
    if os.path.exists(carpeta_audios):
        archivos_repo = [f for f in os.listdir(carpeta_audios) if f.endswith(".wav")]
    
    opcion_archivo = st.selectbox(
        "Escoge un audio del repositorio:", 
        ["-- Subir un archivo nuevo --"] + archivos_repo
    )
    
    # Damos la opcion de subir uno en vivo por si el profe quiere probar con su propio audio
    if opcion_archivo == "-- Subir un archivo nuevo --":
        archivo_a_procesar = st.file_uploader("O sube tu propio archivo .wav", type=["wav"])
    else:
        archivo_a_procesar = os.path.join(carpeta_audios, opcion_archivo)
    
    st.divider()
    
    st.header("2. Analisis por Ventanas")
    st.caption("Recorta un segmento para no saturar la memoria al calcular la FFT")
    inicio_seg, fin_seg = st.slider("Ventana (segundos):", 0.0, 100.0, (0.0, 10.0), step=0.5)

    st.divider()
    
    st.header("3. Bloque SRC (Conversor L/M)")
    # Mostramos la teoria del conversor de tasa multiple
    st.latex(r"y[n] = x_{(\uparrow L \downarrow M)}[n]")
    col1, col2 = st.columns(2)
    with col1:
        factor_L = st.number_input("Expansion (L):", min_value=1, value=1, step=1)
    with col2:
        factor_M = st.number_input("Decimacion (M):", min_value=1, value=1, step=1)
        
    st.divider()
    
    st.header("4. Bloque EQ (Ganancias en dB)")
    # Sliders para las 6 bandas requeridas. Se ajustan en tiempo real.
    sub_bass = st.slider("Sub-Bass (16 - 60 Hz)", -12.0, 12.0, 0.0)
    bass = st.slider("Bass (60 - 250 Hz)", -12.0, 12.0, 0.0)
    low_mids = st.slider("Low Mids (250 - 2k Hz)", -12.0, 12.0, 0.0)
    high_mids = st.slider("High Mids (2k - 4k Hz)", -12.0, 12.0, 0.0)
    presence = st.slider("Presence (4k - 6k Hz)", -12.0, 12.0, 0.0)
    brilliance = st.slider("Brilliance (6k - 16k Hz)", -12.0, 12.0, 0.0)

# --- PROCESAMIENTO CENTRAL ---
if archivo_a_procesar is not None:
    try:
        # Extraemos la frecuencia original y los datos de amplitud del wav
        fs_original, audio_data = wavfile.read(archivo_a_procesar)
        
        # Si la pista es estereo (2 canales), promediamos para volverla mono y tener una sola secuencia x[n]
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
            
        duracion_total = len(audio_data) / fs_original
        
        # Recortamos exactamente las muestras de la ventana seleccionada
        muestra_inicio = int(inicio_seg * fs_original)
        muestra_fin = int(min(fin_seg, duracion_total) * fs_original) 
        
        # SEÑAL 1: x[n] (Entrada original)
        x_n = audio_data[muestra_inicio:muestra_fin]
        
        st.subheader("🎵 Señal de Entrada: x[n] (Original)")
        st.write(f"**Fs original:** {fs_original} Hz")
        st.audio(x_n, sample_rate=fs_original)
        st.divider()

        # SEÑAL 2: y[n] (Salida del conversor, ANTES de ecualizar)
        # Aplicamos la expansion y decimacion primero, como manda el diagrama de bloques
        y_n = dsp.cambiar_tasa(x_n, factor_L, factor_M)
        fs_intermedia = int(fs_original * factor_L / factor_M)
        
        st.subheader("⚙️ Señal Intermedia: y[n] (Remuestreada, sin EQ)")
        st.write(f"**Fs modificada:** {fs_intermedia} Hz")
        # Reproductor opcional para escuchar como el anti-alias corta los agudos al decimar
        st.audio(y_n.astype(np.int16), sample_rate=fs_intermedia)
        st.divider()

        # SEÑAL 3: z[n] (Salida final del ecualizador)
        ganancias = [sub_bass, bass, low_mids, high_mids, presence, brilliance]
        
        # El ecualizador ahora trabaja sobre la fs_intermedia, lo cual es matematicamente correcto
        z_n = dsp.procesar_ecualizador(y_n, fs_intermedia, ganancias)
        
        # Normalizamos la señal final para evitar saturacion (clipping) en la tarjeta de sonido
        if np.max(np.abs(z_n)) > 0:
            z_n = z_n / np.max(np.abs(z_n)) * 32767
        z_n = z_n.astype(np.int16)
        
        st.subheader("🎧 Señal de Salida: z[n] (Final Ecualizada)")
        st.write("Esta es la señal que llega al altavoz, con cambio de tasa y realce frecuencial.")
        st.audio(z_n, sample_rate=fs_intermedia)
        st.divider()

        # --- ZONA DE GRAFICAS ---
        tab_tiempo, tab_frecuencia = st.tabs(["⏱️ Dominio del Tiempo", "📊 Dominio de la Frecuencia"])
        
        with tab_tiempo:
            st.header("Analisis en el Dominio del Tiempo")
            vista_tiempo = st.radio(
                "Modo de visualizacion:", 
                ["Vista Macro (Envolvente continua)", "Vista Micro (Muestras discretas stem)"], 
                horizontal=True
            )
            
            # Pasamos las tres señales a la funcion de graficas para visualizar cada etapa
            fig_tiempo = dsp.generar_graficas_tiempo(x_n, y_n, z_n, modo=vista_tiempo)
            st.pyplot(fig_tiempo)
            
            with st.expander("📚 Ver Teoria: Remuestreo L/M"):
                st.write("Al pasar de x[n] a y[n], la señal atraviesa un filtro pasabajo con frecuencia de corte $\omega_c = \min(\pi/L, \pi/M)$.")

        with tab_frecuencia:
            st.header("Analisis Espectral")
            
            # Pasamos las tres señales para ver como el espectro muta en cada bloque
            fig_frec = dsp.generar_graficas_frecuencia(x_n, y_n, z_n)
            st.pyplot(fig_frec)
            
            with st.expander("📚 Ver Teoria: Efecto de las etapas"):
                st.write("El espectro de y[n] muestra el encogimiento/expansion del eje de frecuencias por el remuestreo. El espectro de z[n] evidencia como los filtros IIR modificaron la magnitud (ganancia) en las bandas seleccionadas.")
                
    except Exception as e:
        st.error(f"Error al procesar el audio. Asegurate de que el archivo no este corrupto. Detalles: {e}")

else:
    st.info(" Sube o selecciona un archivo en el panel izquierdo para arrancar el sistema.")