from __future__ import annotations

import argparse
import time

import numpy as np

from .engine import HybridFireDetectionEngine


def run_simulation(steps: int = 40, sleep_seconds: float = 0.8, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    engine = HybridFireDetectionEngine()
    state = "normal"
    print("Hybrid AI fire simulation started.")

    for index in range(steps):
        if (index + 1) % 8 == 0:
            state = "fire"
        elif (index + 1) % 5 == 0:
            state = "warning"
        elif state == "normal" and rng.random() < 0.18:
            state = "warning"
        elif state == "warning" and rng.random() < 0.28:
            state = "fire"
        elif state == "fire" and rng.random() < 0.22:
            state = "normal"
            engine.reset()

        phase = state

        if phase == "normal":
            temperature = rng.normal(29.0, 1.8)
            humidity = rng.normal(58.0, 5.0)
            flame_signal = int(rng.random() < 0.02)
        elif phase == "warning":
            temperature = rng.normal(38.0, 3.0)
            humidity = rng.normal(40.0, 6.0)
            flame_signal = int(rng.random() < 0.35)
        else:
            temperature = rng.normal(52.0, 6.0)
            humidity = rng.normal(22.0, 5.0)
            flame_signal = 1

        result = engine.predict(float(temperature), float(humidity), flame_signal)
        print("=" * 64)
        print(
            f"Step {index + 1:02d} | Phase={phase.upper():>7} | Temp={temperature:5.2f} C | "
            f"Humidity={humidity:5.2f}% | Flame={flame_signal}"
        )
        print(
            f"Prediction={result['prediction']:>7} | Confidence={result['confidence']:.2f} | "
            f"Anomaly={result['anomaly_flag']} | Buzzer={result['buzzer']}"
        )
        if result["reasons"]:
            print("Reasons:", ", ".join(result["reasons"]))
        time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--sleep", type=float, default=0.8)
    args = parser.parse_args()
    run_simulation(steps=args.steps, sleep_seconds=args.sleep)


if __name__ == "__main__":
    main()
