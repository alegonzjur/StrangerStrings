"""Utilidades musicales: conversión frecuencia <-> nota y desviación en cents.

Convenciones:
- Notación MIDI: La4 (A4) = 69. Cada semitono suma 1.
- Un semitono = 100 cents. Una octava = 1200 cents.
"""
import math

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
A4_MIDI = 69

# Afinación estándar de guitarra (de la 6ª cuerda a la 1ª) en notación MIDI
STANDARD_TUNING_MIDI = {
    "E2": 40,  # 6ª cuerda
    "A2": 45,  # 5ª
    "D3": 50,  # 4ª
    "G3": 55,  # 3ª
    "B3": 59,  # 2ª
    "E4": 64,  # 1ª
}


def midi_to_freq(midi: float, a4: float = 440.0) -> float:
    """Frecuencia de una nota MIDI con temperamento igual."""
    return a4 * 2 ** ((midi - A4_MIDI) / 12)


def freq_to_midi(freq: float, a4: float = 440.0) -> float:
    """Nota MIDI continua (con decimales) de una frecuencia."""
    return A4_MIDI + 12 * math.log2(freq / a4)


def midi_to_name(midi: int) -> str:
    """Nombre de la nota en notación anglosajona, p. ej. 40 -> 'E2'."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def cents(freq: float, reference: float) -> float:
    """Desviación en cents de freq respecto a reference (positivo = alta)."""
    return 1200 * math.log2(freq / reference)


def nearest_note(freq: float, a4: float = 440.0):
    """Modo cromático: devuelve (nombre, frecuencia objetivo, cents)."""
    midi = round(freq_to_midi(freq, a4))
    target = midi_to_freq(midi, a4)
    return midi_to_name(midi), target, cents(freq, target)


def nearest_string(freq: float, tuning=STANDARD_TUNING_MIDI, a4: float = 440.0):
    """Modo guitarra: devuelve (cuerda, frecuencia objetivo, cents).

    Elige la cuerda con menor desviación absoluta, aunque esté a varios
    semitonos. Útil cuando la cuerda está muy desafinada.
    """
    best = None
    for name, midi in tuning.items():
        target = midi_to_freq(midi, a4)
        c = cents(freq, target)
        if best is None or abs(c) < abs(best[2]):
            best = (name, target, c)
    return best
