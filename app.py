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
    page_title="DSP Audio - Sistemas Lineales",
    page_icon="🎵",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio_files"
MAX_MUESTRAS_SALIDA = 5_000_000

st.title("Conversor de Frecuencia de Muestreo y Ecualizador")
st.markdown("**Desarrollado por:** Daniel Molina, Santiago Zumba y Juan Pacheco")
st.markdown("### Universidad de Cuenca — Sistemas Lineales y Señales")


@st.cache_data(show_spinner=False)
def decodificar_wav(contenido: bytes) -> tuple[int, np.ndarray]:
    """Lee bytes WAV, convierte a mono y devuelve float64."""

    fs, datos = wavfile.read(BytesIO(contenido))
    datos = dsp.pcm_a_float(datos)

    if datos.ndim == 2:
        datos = np.mean(datos, axis=1)
    elif datos.ndim != 1:
        raise ValueError("El archivo WAV debe ser mono o estéreo.")

    audio = np.asarray(datos, dtype=np.float64)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if fs <= 0:
        raise ValueError("La frecuencia de muestreo del WAV no es válida.")
    if audio.size < 2:
        raise ValueError("El WAV no contiene suficientes muestras.")

    return int(fs), audio


def convertir_a_wav_bytes(senal: np.ndarray, fs: float) -> bytes:
    """Convierte una señal flotante en un WAV PCM de 16 bits."""

    x = np.asarray(senal, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -1.0, 1.0)
    pcm = np.round(x * 32767.0).astype(np.int16)

    memoria = BytesIO()
    wavfile.write(memoria, int(round(fs)), pcm)
    return memoria.getvalue()


def detener_con_error(mensaje: str, error: Exception | None = None) -> None:
    """Muestra un error claro en la interfaz y detiene la ejecución."""

    st.error(mensaje)
    if error is not None:
        with st.expander("Detalles técnicos"):
            st.exception(error)
    st.stop()


# ============================================================
# 1. SELECCIÓN DE LA SEÑAL
# ============================================================
with st.sidebar:
    st.header("1. Selección de la señal x[n]")

    if AUDIO_DIR.exists():
        archivos_repo = sorted(
            [
                archivo.name
                for archivo in AUDIO_DIR.iterdir()
                if archivo.is_file() and archivo.suffix.lower() == ".wav"
            ],
            key=str.lower,
        )
    else:
        archivos_repo = []

    opcion_archivo = st.selectbox(
        "Escoge un audio del repositorio:",
        ["— Subir un archivo nuevo —"] + archivos_repo,
        key="selector_audio_repo",
    )

    if opcion_archivo == "— Subir un archivo nuevo —":
        archivo_subido = st.file_uploader(
            "O sube tu propio archivo WAV",
            type=["wav"],
            key="cargador_wav",
        )
        if archivo_subido is None:
            contenido_audio = None
            identificador_audio = "sin_audio"
        else:
            contenido_audio = archivo_subido.getvalue()
            identificador_audio = (
                f"subido_{archivo_subido.name}_{len(contenido_audio)}"
            )
    else:
        ruta_audio = AUDIO_DIR / opcion_archivo
        try:
            contenido_audio = ruta_audio.read_bytes()
        except OSError as error:
            detener_con_error(
                f"No se pudo abrir {opcion_archivo}.",
                error,
            )
        identificador_audio = (
            f"repo_{opcion_archivo}_{len(contenido_audio)}"
        )


if contenido_audio is None:
    st.info(
        "Selecciona un WAV de la carpeta audio_files o sube un archivo "
        "desde el panel lateral."
    )
    st.stop()


try:
    fs_original, audio_completo = decodificar_wav(contenido_audio)
except (ValueError, TypeError, OSError) as error:
    detener_con_error("No fue posible leer el archivo WAV.", error)


duracion_total = audio_completo.size / fs_original

if not np.isfinite(duracion_total) or duracion_total <= 0:
    detener_con_error("La duración calculada del audio no es válida.")


# ============================================================
# 2. VENTANA Y PARÁMETROS DSP
# ============================================================
with st.sidebar:
    st.divider()
    st.header("2. Ventana de análisis")
    st.caption(
        "Mueve los dos extremos para escoger el inicio y el final."
    )

    fin_predeterminado = min(10.0, duracion_total)

    if duracion_total <= 1.0:
        paso_ventana = 0.01
    elif duracion_total <= 20.0:
        paso_ventana = 0.10
    else:
        paso_ventana = 0.50

    inicio_seg, fin_seg = st.slider(
        "Ventana (segundos)",
        min_value=0.0,
        max_value=float(duracion_total),
        value=(0.0, float(fin_predeterminado)),
        step=float(paso_ventana),
        key=f"ventana_{identificador_audio}",
    )

    st.caption(
        f"Selección: {inicio_seg:.2f} s → {fin_seg:.2f} s "
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
                key="factor_L",
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
                key="factor_M",
            )
        )

    with st.expander("Configuración avanzada del FIR"):
        num_taps = int(
            st.slider(
                "Número de coeficientes",
                min_value=31,
                max_value=401,
                value=101,
                step=2,
                key="num_taps",
            )
        )
        margen_corte = float(
            st.slider(
                "Margen de corte",
                min_value=0.75,
                max_value=0.99,
                value=0.90,
                step=0.01,
                key="margen_corte",
            )
        )

    st.divider()
    st.header("4. Ecualizador [dB]")

    ganancias = [
        st.slider(
            "Sub-Bass (16–60 Hz)",
            -12.0, 12.0, 0.0, 0.25,
            key="eq_sub_bass",
        ),
        st.slider(
            "Bass (60–250 Hz)",
            -12.0, 12.0, 0.0, 0.25,
            key="eq_bass",
        ),
        st.slider(
            "Low Mids (250–2000 Hz)",
            -12.0, 12.0, 0.0, 0.25,
            key="eq_low_mids",
        ),
        st.slider(
            "High Mids (2000–4000 Hz)",
            -12.0, 12.0, 0.0, 0.25,
            key="eq_high_mids",
        ),
        st.slider(
            "Presence (4000–6000 Hz)",
            -12.0, 12.0, 0.0, 0.25,
            key="eq_presence",
        ),
        st.slider(
            "Brilliance (6000–16000 Hz)",
            -12.0, 12.0, 0.0, 0.25,
            key="eq_brilliance",
        ),
    ]


# ============================================================
# 3. VALIDACIÓN Y RECORTE
# ============================================================
if fin_seg <= inicio_seg:
    detener_con_error(
        "La ventana seleccionada debe tener una duración mayor que cero."
    )

muestra_inicio = int(round(inicio_seg * fs_original))
muestra_fin = int(round(fin_seg * fs_original))
muestra_inicio = max(0, min(muestra_inicio, audio_completo.size - 1))
muestra_fin = max(muestra_inicio + 1, min(muestra_fin, audio_completo.size))

x_n = audio_completo[muestra_inicio:muestra_fin].copy()

if x_n.size < 32:
    detener_con_error(
        "La ventana es demasiado corta. Selecciona al menos 32 muestras."
    )

try:
    L_reducido, M_reducido = dsp.reducir_factores(
        factor_L,
        factor_M,
    )
except (TypeError, ValueError) as error:
    detener_con_error("Los factores L y M no son válidos.", error)

muestras_salida_estimadas = int(
    np.ceil(x_n.size * L_reducido / M_reducido)
)

if muestras_salida_estimadas > MAX_MUESTRAS_SALIDA:
    detener_con_error(
        "La configuración seleccionada generaría demasiadas muestras "
        f"({muestras_salida_estimadas:,}). Reduce la ventana o el factor L."
    )


# ============================================================
# 4. PROCESAMIENTO
# ============================================================
try:
    with st.spinner("Procesando señal..."):
        y_n, info_src = dsp.cambiar_tasa(
            x_n,
            factor_L,
            factor_M,
            num_taps=num_taps,
            margen_corte=margen_corte,
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

        x_reproduccion, _ = dsp.normalizar_audio(x_n)
        y_reproduccion, _ = dsp.normalizar_audio(y_n)
        z_reproduccion, factor_normalizacion = dsp.normalizar_audio(z_n)

except (
    ValueError,
    TypeError,
    FloatingPointError,
    MemoryError,
) as error:
    detener_con_error("Error durante el procesamiento DSP.", error)


# ============================================================
# 5. RESUMEN
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
    f"Ventana: {inicio_seg:.2f}–{fin_seg:.2f} s · "
    f"Duración: {x_n.size / fs_original:.6f} s · "
    f"Muestras de entrada: {x_n.size:,} · "
    f"Muestras de salida: {y_n.size:,} · "
    f"Factor de normalización de z[n]: {factor_normalizacion:.6g}"
)


# ============================================================
# 6. REPRODUCCIÓN
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

if fs_salida > 192000:
    st.warning(
        "La frecuencia de salida es superior a 192 kHz. Algunos navegadores "
        "podrían no reproducirla, aunque el procesamiento y la descarga "
        "siguen siendo válidos."
    )

st.download_button(
    "Descargar señal procesada z[n]",
    data=convertir_a_wav_bytes(z_reproduccion, fs_salida),
    file_name="senal_procesada_z.wav",
    mime="audio/wav",
)


# ============================================================
# 7. DISPONIBILIDAD DE BANDAS
# ============================================================
st.subheader("Disponibilidad real de las bandas")

st.dataframe(
    estados_bandas,
    width="stretch",
    hide_index=True,
)

for estado in estados_bandas:
    if estado["Estado"] != "Completa":
        st.warning(f"{estado['Banda']}: {estado['Observación']}")


# ============================================================
# 8. GRÁFICAS
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
        "Modo de visualización",
        ["Vista completa", "Detalle de muestras"],
        horizontal=True,
        key="modo_tiempo",
    )

    figura_tiempo = dsp.generar_graficas_tiempo(
        x_n,
        y_n,
        z_n,
        fs_original,
        fs_salida,
        fs_salida,
        modo=modo_tiempo,
        inicio_detalle_s=min(
            1.0,
            max(0.0, (fin_seg - inicio_seg) / 2.0),
        ),
        muestras_detalle=100,
    )
    st.pyplot(figura_tiempo, clear_figure=True)

with tab_frecuencia:
    modo_frecuencia = st.radio(
        "Eje de frecuencia",
        [
            "Frecuencia digital normalizada",
            "Frecuencia física [Hz]",
        ],
        horizontal=True,
        key="modo_frecuencia",
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
