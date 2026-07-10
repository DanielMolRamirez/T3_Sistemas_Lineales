import streamlit as st

# Configuración inicial de la página
st.set_page_config(page_title="DSP Audio - Proyecto Final", layout="wide")

st.title("Conversor de Tasa de Muestreo y Ecualizador")
st.markdown("### Sistemas Lineales y Señales")

# Crear el layout de dos columnas: una para controles, otra para gráficas
col_controles, col_graficas = st.columns([1, 2])

with col_controles:
    st.header("Panel de Control")
    st.subheader("1. Frecuencia de Muestreo")
    modo = st.radio("Operación:", ["Decimación (Disminuir)", "Expansión (Incrementar)"])
    factor = st.number_input("Factor (M o L):", min_value=1, value=2, step=1)
    
    st.subheader("2. Ecualizador (Ganancia en dB)")
    sub_bass = st.slider("Sub-Bass (16-60 Hz)", -12.0, 12.0, 0.0)
    bass = st.slider("Bass (60-250 Hz)", -12.0, 12.0, 0.0)
    low_mids = st.slider("Low Mids (250-2k Hz)", -12.0, 12.0, 0.0)
    high_mids = st.slider("High Mids (2k-4k Hz)", -12.0, 12.0, 0.0)
    presence = st.slider("Presence (4k-6k Hz)", -12.0, 12.0, 0.0)
    brilliance = st.slider("Brilliance (6k-16k Hz)", -12.0, 12.0, 0.0)

with col_graficas:
    st.header("Visualización de Señales")
    st.info("Sube un archivo .wav y ajusta los parámetros para ver los resultados aquí.")