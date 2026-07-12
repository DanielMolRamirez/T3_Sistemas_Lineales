from __future__ import annotations

from math import gcd
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig


EPS = 1e-12

BANDAS_EQ = [
    ("Sub-Bass", 16.0, 60.0),
    ("Bass", 60.0, 250.0),
    ("Low Mids", 250.0, 2000.0),
    ("High Mids", 2000.0, 4000.0),
    ("Presence", 4000.0, 6000.0),
    ("Brilliance", 6000.0, 16000.0),
]


def _como_vector_float(x: np.ndarray) -> np.ndarray:
    """Convierte la entrada en un vector mono float64 finito."""

    senal = np.asarray(x)

    if senal.ndim == 2:
        senal = np.mean(senal, axis=1)
    elif senal.ndim != 1:
        raise ValueError("La señal debe ser mono o estéreo.")

    senal = senal.astype(np.float64, copy=False)
    senal = np.nan_to_num(
        senal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if senal.size < 2:
        raise ValueError("La señal debe contener al menos dos muestras.")

    return senal


def pcm_a_float(datos: np.ndarray) -> np.ndarray:
    """Convierte PCM o flotantes WAV a float64."""

    x = np.asarray(datos)

    if np.issubdtype(x.dtype, np.floating):
        return np.nan_to_num(
            x.astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    if x.dtype == np.uint8:
        return (x.astype(np.float64) - 128.0) / 128.0

    if np.issubdtype(x.dtype, np.signedinteger):
        info = np.iinfo(x.dtype)
        escala = float(max(abs(info.min), info.max))
        return x.astype(np.float64) / escala

    if np.issubdtype(x.dtype, np.unsignedinteger):
        info = np.iinfo(x.dtype)
        centro = (float(info.max) + 1.0) / 2.0
        return (x.astype(np.float64) - centro) / centro

    raise TypeError(f"Tipo WAV no compatible: {x.dtype}")


def reducir_factores(L: int, M: int) -> tuple[int, int]:
    """Reduce L/M mediante el máximo común divisor."""

    if not isinstance(L, (int, np.integer)):
        raise TypeError("L debe ser entero.")
    if not isinstance(M, (int, np.integer)):
        raise TypeError("M debe ser entero.")
    if L <= 0 or M <= 0:
        raise ValueError("L y M deben ser mayores que cero.")

    divisor = gcd(int(L), int(M))
    return int(L) // divisor, int(M) // divisor


def cambiar_tasa(
    x: np.ndarray,
    L: int,
    M: int,
    *,
    num_taps: int = 101,
    margen_corte: float = 0.90,
    devolver_info: bool = False,
):
    """Conversión racional mediante filtrado polifásico."""

    senal = _como_vector_float(x)
    Lr, Mr = reducir_factores(L, M)

    if num_taps < 3 or num_taps % 2 == 0:
        raise ValueError("num_taps debe ser impar y al menos 3.")
    if not 0.0 < margen_corte < 1.0:
        raise ValueError("margen_corte debe pertenecer a (0, 1).")

    if Lr == 1 and Mr == 1:
        info = {
            "L_reducido": 1,
            "M_reducido": 1,
            "corte_normalizado": 1.0,
            "num_taps": 1,
            "muestras_esperadas": senal.size,
        }
        return (senal.copy(), info) if devolver_info else senal.copy()

    corte_normalizado = margen_corte / max(Lr, Mr)

    h0 = sig.firwin(
        numtaps=num_taps,
        cutoff=corte_normalizado,
        window="hamming",
        pass_zero="lowpass",
        scale=True,
    )

    y = sig.resample_poly(
        senal,
        up=Lr,
        down=Mr,
        window=h0,
        padtype="line",
    )
    y = np.asarray(y, dtype=np.float64)

    muestras_esperadas = int(np.ceil(senal.size * Lr / Mr))

    if y.size > muestras_esperadas:
        y = y[:muestras_esperadas]
    elif y.size < muestras_esperadas:
        y = np.pad(y, (0, muestras_esperadas - y.size))

    info = {
        "L_reducido": Lr,
        "M_reducido": Mr,
        "corte_normalizado": float(corte_normalizado),
        "num_taps": int(num_taps),
        "muestras_esperadas": muestras_esperadas,
    }

    return (y, info) if devolver_info else y


def _ganancias_a_lista(ganancias_db) -> list[float]:
    """Acepta una lista de seis ganancias o un diccionario."""

    if isinstance(ganancias_db, Mapping):
        valores = [
            float(ganancias_db.get(nombre, 0.0))
            for nombre, _, _ in BANDAS_EQ
        ]
    else:
        valores = [float(valor) for valor in ganancias_db]

    if len(valores) != len(BANDAS_EQ):
        raise ValueError("Se requieren exactamente seis ganancias.")
    if not all(np.isfinite(valor) for valor in valores):
        raise ValueError("Las ganancias deben ser finitas.")

    return valores


def procesar_ecualizador(
    x: np.ndarray,
    fs: float,
    ganancias_db,
    *,
    orden: int = 4,
    margen_nyquist: float = 0.98,
    devolver_info: bool = False,
):
    """Ecualizador transparente a 0 dB."""

    senal = _como_vector_float(x)
    ganancias = _ganancias_a_lista(ganancias_db)

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs debe ser positiva y finita.")
    if orden < 1:
        raise ValueError("El orden debe ser mayor o igual que uno.")
    if not 0.0 < margen_nyquist < 1.0:
        raise ValueError("margen_nyquist debe pertenecer a (0, 1).")

    limite_seguro = margen_nyquist * fs / 2.0
    z = senal.copy()
    estados = []

    for (nombre, f_low, f_high), ganancia_db in zip(
        BANDAS_EQ,
        ganancias,
    ):
        intervalo_solicitado = f"{f_low:.0f}–{f_high:.0f}"

        if f_low >= limite_seguro:
            estados.append(
                {
                    "Banda": nombre,
                    "Ganancia [dB]": ganancia_db,
                    "Intervalo solicitado [Hz]": intervalo_solicitado,
                    "Intervalo utilizado [Hz]": "—",
                    "Estado": "Desactivada",
                    "Observación": (
                        "La banda comienza por encima de Nyquist."
                    ),
                }
            )
            continue

        f_high_usada = min(f_high, limite_seguro)

        if f_high_usada <= f_low:
            estados.append(
                {
                    "Banda": nombre,
                    "Ganancia [dB]": ganancia_db,
                    "Intervalo solicitado [Hz]": intervalo_solicitado,
                    "Intervalo utilizado [Hz]": "—",
                    "Estado": "Desactivada",
                    "Observación": "No existe un intervalo realizable.",
                }
            )
            continue

        if f_high_usada < f_high:
            estado = "Parcial"
            observacion = (
                f"El límite superior se redujo a {f_high_usada:.1f} Hz."
            )
        else:
            estado = "Completa"
            observacion = "Banda nominal disponible completamente."

        estados.append(
            {
                "Banda": nombre,
                "Ganancia [dB]": ganancia_db,
                "Intervalo solicitado [Hz]": intervalo_solicitado,
                "Intervalo utilizado [Hz]": (
                    f"{f_low:.1f}–{f_high_usada:.1f}"
                ),
                "Estado": estado,
                "Observación": observacion,
            }
        )

        if np.isclose(ganancia_db, 0.0, atol=1e-12):
            continue

        sos = sig.butter(
            orden,
            [f_low, f_high_usada],
            btype="bandpass",
            fs=fs,
            output="sos",
        )

        try:
            banda_filtrada = sig.sosfiltfilt(sos, senal)
        except ValueError:
            zi = sig.sosfilt_zi(sos) * senal[0]
            banda_filtrada, _ = sig.sosfilt(
                sos,
                senal,
                zi=zi,
            )

        ganancia_lineal = 10.0 ** (ganancia_db / 20.0)
        z += (ganancia_lineal - 1.0) * banda_filtrada

    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return (z, estados) if devolver_info else z


def normalizar_audio(
    x: np.ndarray,
    pico_objetivo: float = 0.98,
) -> tuple[np.ndarray, float]:
    """Normaliza una copia para reproducción."""

    senal = _como_vector_float(x)

    if not 0.0 < pico_objetivo <= 1.0:
        raise ValueError("pico_objetivo debe pertenecer a (0, 1].")

    pico = float(np.max(np.abs(senal)))

    if pico < EPS:
        return senal.copy(), 1.0

    factor = pico_objetivo / pico
    return senal * factor, factor


def metricas_transparencia(
    referencia: np.ndarray,
    estimada: np.ndarray,
) -> dict[str, float]:
    """Mide el error del ecualizador plano."""

    a = _como_vector_float(referencia)
    b = _como_vector_float(estimada)
    n = min(a.size, b.size)

    error = b[:n] - a[:n]
    mse = float(np.mean(error**2))
    energia = float(np.mean(a[:n] ** 2))

    return {
        "RMSE": float(np.sqrt(mse)),
        "Error máximo": float(np.max(np.abs(error))),
        "SNR [dB]": (
            float("inf")
            if mse < EPS
            else float(
                10.0
                * np.log10((energia + EPS) / (mse + EPS))
            )
        ),
    }


def _reducir_puntos(
    tiempo: np.ndarray,
    senal: np.ndarray,
    max_puntos: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce solo la cantidad de puntos dibujados."""

    paso = max(1, int(np.ceil(senal.size / max_puntos)))
    return tiempo[::paso], senal[::paso]


def generar_graficas_tiempo(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fs_x: float,
    fs_y: float,
    fs_z: float,
    *,
    modo: str = "Vista completa",
    inicio_detalle_s: float = 1.0,
    muestras_detalle: int = 100,
):
    """Tres señales temporales en subgráficas separadas."""

    senales = [
        (_como_vector_float(x), fs_x, "x[n] — Señal original"),
        (_como_vector_float(y), fs_y, "y[n] — Señal remuestreada"),
        (_como_vector_float(z), fs_z, "z[n] — Señal ecualizada"),
    ]

    fig, ejes = plt.subplots(3, 1, figsize=(11, 9))
    fig.subplots_adjust(hspace=0.48)

    if modo == "Detalle de muestras":
        for eje, (senal, fs, titulo) in zip(ejes, senales):
            i0 = int(round(inicio_detalle_s * fs))
            i0 = min(max(i0, 0), max(senal.size - 1, 0))
            i1 = min(senal.size, i0 + muestras_detalle)

            indices = np.arange(i0, i1)
            tiempo_ms = 1000.0 * indices / fs

            eje.stem(
                tiempo_ms,
                senal[i0:i1],
                basefmt=" ",
            )
            eje.set_title(f"{titulo} — {i1 - i0} muestras")
            eje.set_xlabel("Tiempo [ms]")
            eje.set_ylabel("Amplitud")
            eje.grid(True)
    else:
        for eje, (senal, fs, titulo) in zip(ejes, senales):
            tiempo = np.arange(senal.size, dtype=np.float64) / fs
            tg, sg = _reducir_puntos(tiempo, senal)

            eje.plot(tg, sg)
            eje.set_title(titulo)
            eje.set_xlabel("Tiempo [s]")
            eje.set_ylabel("Amplitud")
            eje.grid(True)

    return fig


def _nfft_adecuado(longitud: int) -> int:
    """Escoge una potencia de dos razonable."""

    objetivo = max(4096, min(int(longitud), 65536))
    return 1 << int(np.ceil(np.log2(objetivo)))


def _espectro_db(
    x: np.ndarray,
    fs: float,
    *,
    bilateral: bool,
    minimo_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    """FFT normalizada mediante ventana Hann."""

    senal = _como_vector_float(x)
    senal = senal - np.mean(senal)

    nfft = _nfft_adecuado(senal.size)

    if senal.size < 4:
        ventana = np.ones(senal.size)
    else:
        ventana = np.hanning(senal.size)

    normalizador = max(float(np.sum(ventana)), EPS)

    if bilateral:
        espectro = np.fft.fftshift(
            np.fft.fft(senal * ventana, n=nfft)
        )
        magnitud = np.abs(espectro) / normalizador
        eje = (
            2.0
            * np.pi
            * np.fft.fftshift(np.fft.fftfreq(nfft))
        )
    else:
        espectro = np.fft.rfft(senal * ventana, n=nfft)
        magnitud = np.abs(espectro) / normalizador

        if magnitud.size > 2:
            magnitud[1:-1] *= 2.0

        eje = np.fft.rfftfreq(nfft, d=1.0 / fs)

    piso = 10.0 ** (minimo_db / 20.0)
    magnitud_db = 20.0 * np.log10(
        np.maximum(magnitud, piso)
    )

    return eje, magnitud_db


def generar_graficas_frecuencia(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fs_x: float,
    fs_y: float,
    fs_z: float,
    *,
    modo: str = "Frecuencia digital normalizada",
    minimo_db: float = -120.0,
    frecuencia_maxima_hz: float | None = None,
):
    """Tres espectros en subgráficas separadas."""

    senales = [
        (_como_vector_float(x), fs_x, "Espectro de x[n]"),
        (_como_vector_float(y), fs_y, "Espectro de y[n]"),
        (_como_vector_float(z), fs_z, "Espectro de z[n]"),
    ]

    fig, ejes = plt.subplots(3, 1, figsize=(11, 10))
    fig.subplots_adjust(hspace=0.52)

    if modo == "Frecuencia física [Hz]":
        if frecuencia_maxima_hz is None:
            frecuencia_maxima_hz = min(
                fs_x,
                fs_y,
                fs_z,
            ) / 2.0

        for eje, (senal, fs, titulo) in zip(ejes, senales):
            frecuencia, db = _espectro_db(
                senal,
                fs,
                bilateral=False,
                minimo_db=minimo_db,
            )
            mascara = frecuencia <= frecuencia_maxima_hz

            eje.plot(frecuencia[mascara], db[mascara])
            eje.set_title(titulo)
            eje.set_xlabel("Frecuencia [Hz]")
            eje.set_ylabel("Magnitud [dB]")
            eje.set_xlim(0.0, frecuencia_maxima_hz)
            eje.set_ylim(bottom=minimo_db)
            eje.grid(True)
    else:
        for eje, (senal, fs, titulo) in zip(ejes, senales):
            omega, db = _espectro_db(
                senal,
                fs,
                bilateral=True,
                minimo_db=minimo_db,
            )

            eje.plot(omega, db)
            eje.set_title(titulo)
            eje.set_xlabel(
                r"Frecuencia digital $\Omega$ [rad/muestra]"
            )
            eje.set_ylabel("Magnitud [dB]")
            eje.set_xlim(-np.pi, np.pi)
            eje.set_ylim(bottom=minimo_db)
            eje.set_xticks(
                [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
            )
            eje.set_xticklabels(
                [
                    r"$-\pi$",
                    r"$-\pi/2$",
                    "0",
                    r"$\pi/2$",
                    r"$\pi$",
                ]
            )
            eje.grid(True)

    return fig


def generar_respuesta_ecualizador(
    fs: float,
    ganancias_db,
    *,
    orden: int = 4,
    margen_nyquist: float = 0.98,
    puntos: int = 8192,
):
    """Respuesta total aproximada del ecualizador."""

    ganancias = _ganancias_a_lista(ganancias_db)

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs debe ser positiva y finita.")

    frecuencias = np.linspace(
        0.0,
        fs / 2.0,
        puntos,
        endpoint=False,
    )
    respuesta = np.ones(puntos, dtype=np.complex128)
    limite_seguro = margen_nyquist * fs / 2.0

    for (_, f_low, f_high), ganancia_db in zip(
        BANDAS_EQ,
        ganancias,
    ):
        if f_low >= limite_seguro:
            continue

        f_high_usada = min(f_high, limite_seguro)

        if f_high_usada <= f_low:
            continue
        if np.isclose(ganancia_db, 0.0, atol=1e-12):
            continue

        sos = sig.butter(
            orden,
            [f_low, f_high_usada],
            btype="bandpass",
            fs=fs,
            output="sos",
        )

        _, H = sig.sosfreqz(
            sos,
            worN=frecuencias,
            fs=fs,
        )

        G = 10.0 ** (ganancia_db / 20.0)
        respuesta += (G - 1.0) * (np.abs(H) ** 2)

    respuesta_db = 20.0 * np.log10(
        np.maximum(np.abs(respuesta), EPS)
    )

    fig, eje = plt.subplots(figsize=(10, 4.5))
    eje.plot(frecuencias, respuesta_db)
    eje.set_title("Respuesta total aproximada del ecualizador")
    eje.set_xlabel("Frecuencia [Hz]")
    eje.set_ylabel("Ganancia [dB]")
    eje.set_xlim(0.0, fs / 2.0)
    eje.grid(True)
    fig.tight_layout()

    return fig
