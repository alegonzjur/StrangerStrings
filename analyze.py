"""Analiza WAVs trama a trama con autocorrelación y YIN y compara resultados.

- Si la carpeta tiene manifest.json (señales sintéticas), calcula el error
  real en cents y la tasa de errores gruesos (> 50 cents, casi siempre
  errores de octava o de quinta).
- Si no (grabaciones propias), muestra la nota detectada por cada algoritmo.

Uso:
    python analyze.py                     # analiza ./test_audio
    python analyze.py mis_grabaciones     # otra carpeta
    python analyze.py test_audio --plot   # además guarda gráficas en ./plots
    python analyze.py --threshold 0.05    # umbral de YIN (por defecto 0.15)
"""
import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from notes import cents, nearest_string
from pitch import autocorrelation_pitch, rms, yin_curve, yin_pitch

FRAME = 2048      # ~46 ms a 44,1 kHz: cabe más de 2 períodos de Mi2
HOP = 512
SILENCE_RMS = 0.01
GROSS_ERROR_CENTS = 50

def load_mono(path):
    sr, data = wavfile.read(path)
    if data.dtype.kind == "i":
        data = data / np.iinfo(data.dtype).max
    data = data.astype(np.float64)
    if data.ndim == 2:  # estéreo (p. ej. interfaz de audio) -> mono
        data = data.mean(axis=1)
    peak = np.max(np.abs(data))
    return sr, data / peak if peak > 0 else data


def track(signal, sr, algorithm):
    """Estimación trama a trama. Devuelve (tiempos, f0s) con NaN si no hay
    nota (silencio o señal no periódica)."""
    times, f0s = [], []
    for start in range(0, len(signal) - FRAME, HOP):
        frame = signal[start:start + FRAME]
        f0 = algorithm(frame, sr) if rms(frame) >= SILENCE_RMS else None
        times.append(start / sr)
        f0s.append(np.nan if f0 is None else f0)
    return np.array(times), np.array(f0s)


def summarize(f0s, true_f0=None):
    valid = f0s[~np.isnan(f0s)]
    result = {"voiced": len(valid) / len(f0s)}
    if len(valid) == 0:
        return result
    result["median_f0"] = float(np.median(valid))
    if true_f0 is not None:
        errors = np.array([cents(f, true_f0) for f in valid])
        good = np.abs(errors) <= GROSS_ERROR_CENTS
        result["gross_rate"] = float(np.mean(~good))
        result["median_err"] = float(np.median(errors[good])) if good.any() else np.nan
        result["max_err"] = float(np.max(np.abs(errors[good]))) if good.any() else np.nan
    return result


def plot_file(name, sr, signal, tracks, true_f0, out_dir, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    for alg, (times, f0s) in tracks.items():
        ax1.plot(times, f0s, ".", markersize=3, label=alg)
    if true_f0:
        ax1.axhline(true_f0, color="k", lw=0.8, ls="--", label="real")
    ax1.set_yscale("log")
    ax1.set_xlabel("tiempo (s)")
    ax1.set_ylabel("f0 (Hz)")
    ax1.set_title(name)
    ax1.legend()

    mid = len(signal) // 3
    curve = yin_curve(signal[mid:mid + FRAME], sr)
    ax2.plot(curve)
    ax2.axhline(threshold, color="r", lw=0.8, ls="--", label="umbral")
    if true_f0:
        ax2.axvline(sr / true_f0, color="k", lw=0.8, ls=":", label="período real")
    ax2.set_xlabel("tau (muestras)")
    ax2.set_ylabel("CMNDF")
    ax2.set_title("Curva YIN de una trama")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"{Path(name).stem}.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default="test_audio")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="umbral absoluto de YIN")
    args = parser.parse_args()
    algorithms = {
        "autocorr": autocorrelation_pitch,
        "yin": partial(yin_pitch, threshold=args.threshold),
    }

    folder = Path(args.folder)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    plots_dir = Path("plots")
    if args.plot:
        plots_dir.mkdir(exist_ok=True)

    totals = {alg: [] for alg in algorithms}
    for path in sorted(folder.glob("*.wav")):
        sr, signal = load_mono(path)
        true_f0 = manifest.get(path.name, {}).get("f0")
        tracks = {alg: track(signal, sr, fn) for alg, fn in algorithms.items()}

        print(f"\n{path.name}" + (f"  (real: {true_f0:.2f} Hz)" if true_f0 else ""))
        for alg, (_, f0s) in tracks.items():
            s = summarize(f0s, true_f0)
            totals[alg].append(s)
            if "median_f0" not in s:
                print(f"  {alg:9s} sin detección")
                continue
            string, target, c = nearest_string(s["median_f0"])
            line = (f"  {alg:9s} {s['median_f0']:8.2f} Hz -> {string} {c:+6.1f} c"
                    f" | con nota {s['voiced']:.0%}")
            if true_f0:
                line += (f" | errores gruesos {s['gross_rate']:.0%}"
                         f" | error mediano {s['median_err']:+.2f} c")
            print(line)
        if args.plot:
            plot_file(path.name, sr, signal, tracks, true_f0, plots_dir,
                      args.threshold)

    if manifest:
        print("\n=== Resumen ===")
        for alg, results in totals.items():
            gross = np.mean([r.get("gross_rate", 1.0) for r in results])
            errs = [abs(r["median_err"]) for r in results
                    if not np.isnan(r.get("median_err", np.nan))]
            worst = max(errs) if errs else float("nan")
            print(f"  {alg:9s} errores gruesos medios {gross:6.1%}"
                  f" | peor error mediano {worst:.2f} c")


if __name__ == "__main__":
    main()
