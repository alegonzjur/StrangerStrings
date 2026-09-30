"""Detección de frecuencia fundamental (pitch) sobre una trama de audio.

Dos algoritmos:
- autocorrelation_pitch: línea base, sencilla, propensa a errores de octava.
- yin_pitch: YIN (de Cheveigné & Kawahara, 2002), el estándar en afinadores.

Ambas funciones reciben una trama mono en float y devuelven la frecuencia
en Hz, o None si no detectan una nota fiable.

Rango por defecto 70-1000 Hz: cubre Mi2 (82,41 Hz) con margen para
afinaciones más graves y los trastes altos de la 1ª cuerda.
"""
import numpy as np


def rms(frame: np.ndarray) -> float:
    """Energía media de la trama; se usa como puerta de silencio."""
    return float(np.sqrt(np.mean(frame ** 2)))


def _parabolic(y: np.ndarray, i: int) -> float:
    """Refina la posición de un extremo en i ajustando una parábola a
    y[i-1], y[i], y[i+1]. Devuelve una posición con decimales."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = a - 2 * b + c
    if denom == 0:
        return float(i)
    return i + 0.5 * (a - c) / denom


# ---------------------------------------------------------------------------
# Autocorrelación
# ---------------------------------------------------------------------------
def autocorrelation(frame: np.ndarray) -> np.ndarray:
    """Autocorrelación normalizada (r[0] = 1), calculada vía FFT.

    Se rellena con ceros hasta 2N para evitar la correlación circular."""
    x = frame - np.mean(frame)
    n = len(x)
    spec = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(spec * np.conj(spec))[:n]
    if r[0] <= 0:
        return np.zeros(n)
    return r / r[0]


def autocorrelation_pitch(frame, sr, fmin=70.0, fmax=1000.0):
    """Elige el pico más alto de la autocorrelación dentro del rango de
    períodos posibles. Deliberadamente ingenuo: es la línea base."""
    r = autocorrelation(frame)
    tau_min = int(sr / fmax)
    tau_max = min(int(sr / fmin), len(r) - 2)
    if tau_max <= tau_min:
        return None
    i = int(np.argmax(r[tau_min:tau_max + 1])) + tau_min
    if r[i] <= 0:
        return None
    return sr / _parabolic(r, i)


# ---------------------------------------------------------------------------
# YIN
# ---------------------------------------------------------------------------
def difference_function(x: np.ndarray, tau_max: int) -> np.ndarray:
    """Paso 1: d(tau) = sum_j (x[j] - x[j+tau])^2 sobre una ventana fija W.

    Implementación directa (bucle sobre tau) por claridad. Para tiempo
    real se sustituirá por una versión vía FFT."""
    w = len(x) - tau_max
    d = np.zeros(tau_max + 1)
    for tau in range(1, tau_max + 1):
        diff = x[:w] - x[tau:tau + w]
        d[tau] = np.dot(diff, diff)
    return d


def cmndf(d: np.ndarray) -> np.ndarray:
    """Paso 2: diferencia normalizada por la media acumulada.

    d'(0) = 1;  d'(tau) = d(tau) / ((1/tau) * sum_{j=1..tau} d(j))
    """
    out = np.ones_like(d)
    cumsum = np.cumsum(d[1:])
    taus = np.arange(1, len(d))
    safe = np.where(cumsum > 0, cumsum, 1.0)
    out[1:] = np.where(cumsum > 0, d[1:] * taus / safe, 1.0)
    return out


def yin_curve(frame, sr, fmin=70.0):
    """Devuelve la curva CMNDF completa (útil para graficar y depurar)."""
    x = frame - np.mean(frame)
    tau_max = int(sr / fmin)
    return cmndf(difference_function(x, tau_max))


def yin_pitch(frame, sr, fmin=70.0, fmax=1000.0, threshold=0.15):
    """YIN completo: diferencia, CMNDF, umbral absoluto e interpolación."""
    x = frame - np.mean(frame)
    tau_min = int(sr / fmax)
    tau_max = int(sr / fmin)
    if len(x) <= tau_max + tau_min:
        raise ValueError(
            f"Trama demasiado corta ({len(x)}) para fmin={fmin} Hz; "
            f"necesita más de {tau_max + tau_min} muestras."
        )
    dp = cmndf(difference_function(x, tau_max))

    # Paso 3: primer tau bajo el umbral, y de ahí bajar hasta el mínimo local
    tau = None
    for t in range(tau_min, tau_max):
        if dp[t] < threshold:
            while t + 1 < tau_max and dp[t + 1] < dp[t]:
                t += 1
            tau = t
            break
    if tau is None:
        return None  # señal no periódica (ruido, silencio, ataque)

    # Paso 4: precisión por debajo de la muestra
    return sr / _parabolic(dp, tau)
