# Conversor de Frecuencia de Muestreo y Ecualizador de Audio en Tiempo Discreto

**Universidad de Cuenca - Carrera de Ingeniería en Telecomunicaciones** **Materia:** Sistemas Lineales y Señales  
**Autores:** Daniel Molina, Santiago Zumba, Juan Pacheco  

---

## 📌 Descripción del Proyecto
Este proyecto es una aplicación web interactiva desarrollada en Python orientada al Procesamiento Digital de Señales (DSP). Su objetivo principal es aplicar la teoría del análisis de Fourier y el muestreo en tiempo discreto basándose en la literatura de Oppenheim. 

El sistema recibe una señal de audio original $x[n]$, aplica un cambio de tasa de muestreo fraccional (diezmado e interpolación) para generar una señal intermedia $y[n]$, y finalmente procesa la señal mediante un ecualizador IIR de 6 bandas para obtener la salida $z[n]$.

## 🚀 Características Principales

* **Remuestreo Fraccional ($L/M$):** Implementa expansión (upsampling por $L$) y decimación (downsampling por $M$) de forma simultánea utilizando un único filtro FIR de tasa múltiple. La frecuencia de corte del filtro se ajusta dinámicamente como $\omega_c = \min(\pi/L, \pi/M)$ para evitar imágenes espectrales y aliasing.
* **Ecualizador IIR de 6 Bandas:** Filtros Butterworth diseñados utilizando Secciones de Segundo Orden (SOS) para garantizar una estabilidad numérica absoluta, evitando la cancelación catastrófica en bajas frecuencias (ej. banda Sub-Bass de 16 Hz).
* **Análisis por Ventanas:** Permite recortar segmentos específicos del audio (en segundos) para aislar transientes y evitar sobrecargas de memoria al calcular la Transformada Rápida de Fourier (FFT).
* **Visualización en el Dominio del Tiempo:** * *Vista Macro:* Muestra la envolvente continua de la señal.
  * *Vista Micro:* Utiliza diagramas de tallos (`stem`) para evidenciar el comportamiento discreto de las muestras $x[n]$.
* **Visualización en el Dominio de la Frecuencia:** Calcula y grafica el espectro de magnitud normalizado de las señales, con el eje de frecuencias digitales $\Omega$ rigurosamente etiquetado en el intervalo $[-\pi, \pi]$.

## 📁 Estructura del Repositorio

```text
/
├── app.py               # Interfaz gráfica (UI) y servidor web en Streamlit
├── dsp_core.py          # Motor matemático de DSP (Filtros, FFT, remuestreo)
├── requirements.txt     # Dependencias del proyecto
├── README.md            # Documentación
└── /audio_files         # Carpeta con archivos .wav locales para pruebas rápidas