from __future__ import annotations

import os
import re

from flask import Flask, jsonify, render_template, request

from .analysis import build_project_analysis
from .dataset_player import DatasetPlaybackService
from .engine import HybridFireDetectionEngine
from .hardware import ArduinoHardwareMonitor

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))

engine = HybridFireDetectionEngine()
hardware_monitor = ArduinoHardwareMonitor()
dataset_player = DatasetPlaybackService()

SERIAL_PORT_PATTERN = re.compile(
    r"^(/dev/tty[A-Za-z0-9._-]+|COM\d{1,3})$"
)


@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


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
    port = payload.get("port")
    if port is not None:
        if not isinstance(port, str) or not SERIAL_PORT_PATTERN.match(port):
            return jsonify({"error": "Invalid serial port format."}), 400
    hardware_monitor.start(port=port)
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
    hardware_monitor.stop()
    hardware_monitor.reset()
    dataset_player.reset()
    return jsonify({"status": "reset"}), 200


@app.post("/predict")
def predict() -> tuple:
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    try:
        temperature = float(payload["temperature"])
        humidity = float(payload["humidity"])
        flame_signal = int(payload.get("flame", payload.get("flame_signal", 0)))
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    if not (-50 <= temperature <= 150):
        return jsonify({"error": "temperature must be between -50 and 150."}), 400
    if not (0 <= humidity <= 100):
        return jsonify({"error": "humidity must be between 0 and 100."}), 400
    if flame_signal not in (0, 1):
        return jsonify({"error": "flame must be 0 or 1."}), 400

    result = engine.predict(
        temperature=temperature,
        humidity=humidity,
        flame_signal=flame_signal,
    )
    return jsonify(result), 200


def main() -> None:
    print("Hybrid AI Fire Detection dashboard: http://127.0.0.1:5000")
    print("Start Live Feed in the dashboard when you want to connect to the Arduino.")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
