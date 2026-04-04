from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from .analysis import build_project_analysis
from .dataset_player import DatasetPlaybackService
from .engine import HybridFireDetectionEngine
from .hardware import ArduinoHardwareMonitor

app = Flask(__name__, template_folder="templates", static_folder="static")
engine = HybridFireDetectionEngine()
hardware_monitor = ArduinoHardwareMonitor()
dataset_player = DatasetPlaybackService()


@app.get("/health")
def health() -> tuple:
    return jsonify({"status": "ok", "model": "hybrid_fire_model"}), 200


@app.get("/")
def dashboard() -> str:
    return render_template("dashboard.html", analysis=build_project_analysis())


@app.get("/analysis")
def analysis() -> tuple:
    return jsonify(build_project_analysis()), 200


@app.get("/hardware/status")
def hardware_status() -> tuple:
    return jsonify(hardware_monitor.status()), 200


@app.post("/hardware/connect")
def hardware_connect() -> tuple:
    payload = request.get_json(silent=True) or {}
    hardware_monitor.start(port=payload.get("port"))
    return jsonify(hardware_monitor.status()), 200


@app.post("/hardware/disconnect")
def hardware_disconnect() -> tuple:
    hardware_monitor.stop()
    return jsonify(hardware_monitor.status()), 200


@app.get("/dataset/status")
def dataset_status() -> tuple:
    return jsonify(dataset_player.status()), 200


@app.post("/dataset/reset")
def dataset_reset() -> tuple:
    dataset_player.reset()
    return jsonify(dataset_player.status()), 200


@app.post("/dataset/next")
def dataset_next() -> tuple:
    packet = dataset_player.next_packet()
    if packet["reading"]["flame"] == 1 or packet["result"]["buzzer"]:
        hardware_monitor.pulse_buzzer(reason="dataset event")
    return jsonify(packet), 200


@app.post("/reset")
def reset() -> tuple:
    engine.reset()
    hardware_monitor.reset()
    dataset_player.reset()
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
    hardware_monitor.start()
    print("Hybrid AI Fire Detection dashboard: http://127.0.0.1:5000")
    print("Arduino auto-detect is enabled. Use the dashboard to switch between dataset playback and live hardware capture.")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
