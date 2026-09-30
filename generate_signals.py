"""Genera WAVs sintéticos de cuerdas de guitarra con frecuencia conocida.

Por qué sintéticos: para medir el error de un algoritmo hace falta saber la
frecuencia real. Con grabaciones solo sabes lo que "debería" sonar.

El modelo imita los rasgos que complican la detección en una guitarra:
- Fundamental más débil que el 2º armónico (típico en cuerdas graves y
  pastillas de guitarra eléctrica): provoca errores de octava.
- Armónicos agudos que se apagan antes que los graves.
- Inharmonicidad: los armónicos reales de una cuerda rígida están un poco
  por encima de k*f0 (f_k = k*f0*sqrt(1 + B*k^2)).
- Transitorio de ataque (ráfaga de ruido de la púa) y ruido de fondo.

Uso:
    python generate_signals.py            # genera en ./test_audio
    python generate_signals.py --out otra_carpeta
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from notes import STANDARD_TUNING_MIDI, midi_to_freq

SR = 44100
DURATION = 3.0
# Desafinaciones de prueba en cents: muy baja, afinada, algo alta
DETUNINGS = [-30, 0, 15]
# Amplitud relativa de los armónicos 1..10 (el 2º domina)
HARMONIC_AMPS = [0.4, 1.0, 0.8, 0.6, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08]


def pluck(f0, sr=SR, duration=DURATION, inharmonicity=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * duration)) / sr
    signal = np.zeros_like(t)
    for k, amp in enumerate(HARMONIC_AMPS, start=1):
        fk = k * f0 * np.sqrt(1 + inharmonicity * k ** 2)
        if fk >= sr / 2:
            break
        decay = np.exp(-t * (1.2 + 0.5 * k))
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * decay * np.sin(2 * np.pi * fk * t + phase)

    # Ataque: 20 ms de ruido que decae rápido
    attack = int(0.02 * sr)
    signal[:attack] += rng.normal(0, 0.5, attack) * np.linspace(1, 0, attack)
    # Ruido de fondo constante
    signal += rng.normal(0, 0.005, len(t))

    return signal / np.max(np.abs(signal)) * 0.9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="test_audio")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    manifest = {}
    seed = 0
    for name, midi in STANDARD_TUNING_MIDI.items():
        for det in DETUNINGS:
            f0 = midi_to_freq(midi) * 2 ** (det / 1200)
            filename = f"{name}_{det:+d}c.wav"
            audio = pluck(f0, seed=seed)
            wavfile.write(out / filename, SR, (audio * 32767).astype(np.int16))
            manifest[filename] = {"string": name, "detune_cents": det, "f0": f0}
            seed += 1

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generados {len(manifest)} ficheros en {out.resolve()}")


if __name__ == "__main__":
    main()
