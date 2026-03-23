from __future__ import annotations

from flask import Flask, jsonify, request

from .engine import HybridFireDetectionEngine

app = Flask(__name__)
engine = HybridFireDetectionEngine()


@app.get("/health")
def health() -> tuple:
    return jsonify({"status": "ok", "model": "hybrid_fire_model"}), 200


@app.post("/reset")
def reset() -> tuple:
    engine.reset()
    return jsonify({"status": "reset"}), 200


@app.post("/predict")
def predict() -> tuple:
    payload = request.get_json(force=True)

    temperature = float(payload["temperature"])
    humidity = float(payload["humidity"])
    flame_signal = int(payload.get("flame", payload.get("flame_signal", 0)))

    result = engine.predict(
        temperature=temperature,
        humidity=humidity,
        flame_signal=flame_signal,
    )
    return jsonify(result), 200


def main() -> None:
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
