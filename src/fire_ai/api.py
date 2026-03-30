from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from .analysis import build_project_analysis
from .engine import HybridFireDetectionEngine

app = Flask(__name__, template_folder="templates", static_folder="static")
engine = HybridFireDetectionEngine()


@app.get("/health")
def health() -> tuple:
    return jsonify({"status": "ok", "model": "hybrid_fire_model"}), 200


@app.get("/")
def dashboard() -> str:
    return render_template("dashboard.html", analysis=build_project_analysis())


@app.get("/analysis")
def analysis() -> tuple:
    return jsonify(build_project_analysis()), 200


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
    print("Hybrid AI Fire Detection dashboard: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
