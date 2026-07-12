from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from scipy.io import wavfile

import dsp_core as dsp


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="DSP Audio - Proyecto Sistemas Lineales",
    page_icon="🎵",
    layout="wide",
)

st.title("Conversor de Frecuencia de Muestreo y Ecualizador")
st.markdown("**Desarrollado por:** Daniel Molina, Santiago Zumba y Juan Pacheco")
st.markdown("### Universidad de Cuenca — Sistemas Lineales y Señales")

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio_files"


def leer_wav(origen) -> tuple[int, np.ndarray]:
    """Lee un WAV local o cargado y lo convierte a mono float64."""

    if hasattr(origen, "getvalue"):
        fuente = BytesIO(origen.getvalue())
    else:
        fuente = str(origen)

    fs, datos = wavfile.read(fuente)
    datos = dsp.pcm_a_float(datos)

    if datos.ndim == 2:
        datos = datos.mean(axis=1)
    elif datos.ndim != 1:
        raise ValueError("El WAV debe ser mono o estéreo.")

    return int(fs), np.asarray(datos, dtype=np.float64)


def convertir_a_wav_bytes(senal: np.ndarray, fs: float) -> bytes:
    """Convierte una señal flotante a WAV PCM de 16 bits."""

    x = np.asarray(senal, dtype=np.float64)
    x = np.clip(x, -1.0, 1.0)
    pcm = np.round(x * 32767.0).astype(np.int16)

    memoria = BytesIO()
    wavfile.write(memoria, int(round(fs)), pcm)
    return memoria.getvalue()


# ============================================================
# 1. SELECCIÓN DE LA SEÑAL
# ============================================================
with st.sidebar:
    st.header("1. Selección de la señal x[n]")

    AUDIO_DIR.mkdir(exist_ok=True)

    archivos_repo = sorted(
        [
            archivo.name
            for archivo in AUDIO_DIR.iterdir()
            if archivo.is_file() and archivo.suffix.lower() == ".wav"
        ],
        key=str.lower,
    )

    opcion_archivo = st.selectbox(
        "Escoge un audio del repositorio:",
        ["— Subir un archivo nuevo —"] + archivos_repo,
    )

    if opcion_archivo == "— Subir un archivo nuevo —":
        origen_audio = st.file_uploader(
            "O sube tu propio archivo WAV",
            type=["wav"],
        )
    else:
        origen_audio = AUDIO_DIR / opcion_archivo


if origen_audio is None:
    st.info(
        "Selecciona un archivo de audio del repositorio o sube un WAV "
        "desde el panel lateral."
    )
    st.stop()


try:
    fs_original, audio_completo = leer_wav(origen_audio)
except (ValueError, TypeError, OSError) as error:
    st.error(f"No fue posible leer el archivo WAV: {error}")
    st.stop()


duracion_total = len(audio_completo) / fs_original

if duracion_total <= 0:
    st.error("El archivo no contiene una señal válida.")
    st.stop()


# ============================================================
# 2. VENTANA Y PARÁMETROS DSP
# ============================================================
with st.sidebar:
    st.divider()
    st.header("2. Ventana de análisis")
    st.caption(
        "Selecciona directamente el instante inicial y final del segmento."
    )

    fin_predeterminado = min(10.0, duracion_total)

    if duracion_total <= 1.0:
        paso_ventana = 0.01
    elif duracion_total <= 20.0:
        paso_ventana = 0.1
    else:
        paso_ventana = 0.5

    # Un único control con dos extremos, igual que en la versión anterior.
    inicio_seg, fin_seg = st.slider(
        "Ventana (segundos):",
        min_value=0.0,
        max_value=float(duracion_total),
        value=(0.0, float(fin_predeterminado)),
        step=float(paso_ventana),
    )

    st.caption(
        f"Segmento seleccionado: {inicio_seg:.2f} s → {fin_seg:.2f} s "
        f"({fin_seg - inicio_seg:.2f} s)"
    )

    st.divider()
    st.header("3. Bloque SRC")

    st.latex(
        r"x[n]\longrightarrow \uparrow L"
        r"\longrightarrow h[n]\longrightarrow \downarrow M"
        r"\longrightarrow y[n]"
    )

    col_l, col_m = st.columns(2)

    with col_l:
        factor_L = int(
            st.number_input(
                "Expansión L",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
            )
        )

    with col_m:
        factor_M = int(
            st.number_input(
                "Decimación M",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
            )
        )

    st.divider()
    st.header("4. Ecualizador [dB]")

    sub_bass = st.slider(
        "Sub-Bass (16–60 Hz)", -12.0, 12.0, 0.0, 0.25
    )
    bass = st.slider(
        "Bass (60–250 Hz)", -12.0, 12.0, 0.0, 0.25
    )
    low_mids = st.slider(
        "Low Mids (250–2000 Hz)", -12.0, 12.0, 0.0, 0.25
    )
    high_mids = st.slider(
        "High Mids (2000–4000 Hz)", -12.0, 12.0, 0.0, 0.25
    )
    presence = st.slider(
        "Presence (4000–6000 Hz)", -12.0, 12.0, 0.0, 0.25
    )
    brilliance = st.slider(
        "Brilliance (6000–16000 Hz)", -12.0, 12.0, 0.0, 0.25
    )


# ============================================================
# 3. RECORTE Y PROCESAMIENTO
# ============================================================
if fin_seg <= inicio_seg:
    st.warning("Selecciona un intervalo con una duración mayor que cero.")
    st.stop()

muestra_inicio = int(round(inicio_seg * fs_original))
muestra_fin = int(round(fin_seg * fs_original))
muestra_fin = min(muestra_fin, len(audio_completo))

x_n = audio_completo[muestra_inicio:muestra_fin].copy()

ganancias = [
    sub_bass,
    bass,
    low_mids,
    high_mids,
    presence,
    brilliance,
]

try:
    y_n, info_src = dsp.cambiar_tasa(
        x_n,
        factor_L,
        factor_M,
        devolver_info=True,
    )

    fs_salida = (
        fs_original
        * info_src["L_reducido"]
        / info_src["M_reducido"]
    )

    z_n, estados_bandas = dsp.procesar_ecualizador(
        y_n,
        fs_salida,
        ganancias,
        devolver_info=True,
    )

    # La normalización se usa solo para escuchar o descargar.
    x_reproduccion, _ = dsp.normalizar_audio(x_n)
    y_reproduccion, _ = dsp.normalizar_audio(y_n)
    z_reproduccion, factor_normalizacion = dsp.normalizar_audio(z_n)

except (ValueError, TypeError, FloatingPointError) as error:
    st.error(f"Error durante el procesamiento DSP: {error}")
    st.stop()


# ============================================================
# 4. RESUMEN
# ============================================================
st.subheader("Resumen del procesamiento")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Fs original", f"{fs_original:.0f} Hz")
col2.metric("Fs de salida", f"{fs_salida:.2f} Hz")
col3.metric("Nyquist de salida", f"{fs_salida / 2.0:.2f} Hz")
col4.metric(
    "Razón reducida",
    f"{info_src['L_reducido']}/{info_src['M_reducido']}",
)

st.caption(
    f"Ventana analizada: {inicio_seg:.2f}–{fin_seg:.2f} s · "
    f"Duración: {len(x_n) / fs_original:.6f} s · "
    f"Muestras de entrada: {len(x_n)} · "
    f"Muestras de salida: {len(y_n)} · "
    f"Factor de normalización de z[n]: {factor_normalizacion:.6g}"
)


# ============================================================
# 5. REPRODUCCIÓN
# ============================================================
st.subheader("Señales del sistema")

col_x, col_y, col_z = st.columns(3)

with col_x:
    st.markdown("#### x[n] — Original")
    st.audio(
        convertir_a_wav_bytes(x_reproduccion, fs_original),
        format="audio/wav",
    )

with col_y:
    st.markdown("#### y[n] — Remuestreada")
    st.audio(
        convertir_a_wav_bytes(y_reproduccion, fs_salida),
        format="audio/wav",
    )

with col_z:
    st.markdown("#### z[n] — Ecualizada")
    st.audio(
        convertir_a_wav_bytes(z_reproduccion, fs_salida),
        format="audio/wav",
    )

st.download_button(
    "Descargar señal procesada z[n]",
    data=convertir_a_wav_bytes(z_reproduccion, fs_salida),
    file_name="senal_procesada_z.wav",
    mime="audio/wav",
)


# ============================================================
# 6. BANDAS
# ============================================================
st.subheader("Disponibilidad real de las bandas")

st.dataframe(
    estados_bandas,
    use_container_width=True,
    hide_index=True,
)

for estado in estados_bandas:
    if estado["Estado"] != "Completa":
        st.warning(f"{estado['Banda']}: {estado['Observación']}")


# ============================================================
# 7. GRÁFICAS
# ============================================================
tab_tiempo, tab_frecuencia, tab_respuesta = st.tabs(
    [
        "⏱️ Dominio del tiempo",
        "📊 Dominio de la frecuencia",
        "🎚️ Respuesta del ecualizador",
    ]
)

with tab_tiempo:
    modo_tiempo = st.radio(
        "Modo de visualización:",
        ["Vista completa", "Detalle de muestras"],
        horizontal=True,
    )

    figura_tiempo = dsp.generar_graficas_tiempo(
        x_n,
        y_n,
        z_n,
        fs_original,
        fs_salida,
        fs_salida,
        modo=modo_tiempo,
        inicio_detalle_s=min(1.0, (fin_seg - inicio_seg) / 2.0),
        muestras_detalle=100,
    )

    # Cada señal aparece en un subplot independiente.
    st.pyplot(figura_tiempo, clear_figure=True)

with tab_frecuencia:
    modo_frecuencia = st.radio(
        "Eje de frecuencia:",
        [
            "Frecuencia digital normalizada",
            "Frecuencia física [Hz]",
        ],
        horizontal=True,
    )

    figura_frecuencia = dsp.generar_graficas_frecuencia(
        x_n,
        y_n,
        z_n,
        fs_original,
        fs_salida,
        fs_salida,
        modo=modo_frecuencia,
        minimo_db=-120.0,
        frecuencia_maxima_hz=min(
            fs_original / 2.0,
            fs_salida / 2.0,
        ),
    )

    # Cada espectro aparece en un subplot independiente.
    st.pyplot(figura_frecuencia, clear_figure=True)

with tab_respuesta:
    figura_respuesta = dsp.generar_respuesta_ecualizador(
        fs_salida,
        ganancias,
    )
    st.pyplot(figura_respuesta, clear_figure=True)

    if all(abs(valor) < 1e-12 for valor in ganancias):
        metricas = dsp.metricas_transparencia(y_n, z_n)
        st.success(
            "Todas las ganancias están en 0 dB: "
            "el ecualizador es transparente."
        )
        st.json(metricas)
