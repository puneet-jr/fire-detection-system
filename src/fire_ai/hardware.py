from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import os
import re
import threading
import time
from typing import Any

from .engine import HybridFireDetectionEngine

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover - depends on local environment
    serial = None
    list_ports = None


PAIR_PATTERN = re.compile(r"([A-Za-z_]+)\s*[:=]\s*([-+]?[A-Za-z0-9_.]+)")

KEY_ALIASES = {
    "temperature": "temperature",
    "temp": "temperature",
    "t": "temperature",
    "humidity": "humidity",
    "hum": "humidity",
    "h": "humidity",
    "flame": "flame",
    "flame_signal": "flame",
    "f": "flame",
    "alarm": "local_alarm",
    "local": "local_alarm",
    "local_alarm": "local_alarm",
    "zt": "temp_zscore",
    "zh": "humidity_zscore",
}


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse_json_line(line: str) -> dict[str, Any] | None:
    if not line.startswith("{"):
        return None

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None

    if "temperature" not in payload or "humidity" not in payload:
        return None

    return {
        "temperature": float(payload["temperature"]),
        "humidity": float(payload["humidity"]),
        "flame": int(payload.get("flame", payload.get("flame_signal", 0))),
        "local_alarm": int(payload.get("local_alarm", payload.get("alarm", 0))),
        "temp_zscore": float(payload.get("temp_zscore", 0.0)),
        "humidity_zscore": float(payload.get("humidity_zscore", 0.0)),
        "raw_line": line,
        "timestamp": _timestamp(),
    }


def parse_sensor_line(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None

    json_reading = _parse_json_line(stripped)
    if json_reading is not None:
        return json_reading

    values: dict[str, str] = {}
    for raw_key, raw_value in PAIR_PATTERN.findall(stripped):
        canonical_key = KEY_ALIASES.get(raw_key.strip().lower())
        if canonical_key is None:
            continue
        values[canonical_key] = raw_value

    if "temperature" not in values or "humidity" not in values:
        return None

    return {
        "temperature": float(values["temperature"]),
        "humidity": float(values["humidity"]),
        "flame": int(float(values.get("flame", "0"))),
        "local_alarm": int(float(values.get("local_alarm", "0"))),
        "temp_zscore": float(values.get("temp_zscore", "0.0")),
        "humidity_zscore": float(values.get("humidity_zscore", "0.0")),
        "raw_line": stripped,
        "timestamp": _timestamp(),
    }


@dataclass
class ArduinoHardwareMonitor:
    preferred_port: str | None = None
    baudrate: int = 9600
    reconnect_delay: float = 3.0
    read_timeout: float = 1.0
    history_size: int = 40

    def __post_init__(self) -> None:
        env_port = os.getenv("FIRE_HARDWARE_PORT")
        env_baud = os.getenv("FIRE_HARDWARE_BAUD")
        if not self.preferred_port and env_port:
            self.preferred_port = env_port
        if env_baud:
            self.baudrate = int(env_baud)

        self.engine = HybridFireDetectionEngine()
        self.history: deque[dict[str, Any]] = deque(maxlen=self.history_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial_connection = None
        self._connected = False
        self._port: str | None = None
        self._last_reading: dict[str, Any] | None = None
        self._last_result: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._last_message: str = "Hardware monitor idle."
        self._sample_count = 0
        self._last_buzzer_command: bool | None = None
        self._pulse_generation = 0

    @property
    def available(self) -> bool:
        return serial is not None and list_ports is not None

    def start(self, port: str | None = None) -> None:
        if port:
            self.preferred_port = port

        if not self.available:
            with self._lock:
                self._last_error = "pyserial is not installed. Install requirements to enable Arduino support."
                self._last_message = "Hardware mode unavailable until pyserial is installed."
            return

        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="arduino-hardware-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self._close_connection("Hardware monitor stopped.")

    def reset(self) -> None:
        self.engine.reset()
        with self._lock:
            self.history.clear()
            self._sample_count = 0
            self._last_reading = None
            self._last_result = None
            self._last_buzzer_command = None
            self._last_message = "Hardware monitor reset. Waiting for fresh readings."
        self.push_buzzer_state(False, reason="reset")

    def push_buzzer_state(self, enabled: bool, reason: str = "prediction") -> bool:
        command = f"BUZZER:{1 if enabled else 0}\n".encode("ascii")
        with self._lock:
            connection = self._serial_connection
            is_connected = self._connected

        if not is_connected or connection is None:
            return False

        try:
            connection.write(command)
            with self._lock:
                self._last_buzzer_command = enabled
                self._last_message = f"Sent buzzer command from {reason}: {'ON' if enabled else 'OFF'}."
            return True
        except Exception as exc:  # pragma: no cover - depends on OS/device state
            self._close_connection(f"Lost Arduino connection while writing buzzer command: {exc}")
            return False

    def pulse_buzzer(self, duration_seconds: float = 0.35, reason: str = "dataset pulse") -> bool:
        with self._lock:
            self._pulse_generation += 1
            generation = self._pulse_generation

        if not self.push_buzzer_state(True, reason=reason):
            return False

        def _reset_after_pulse() -> None:
            time.sleep(max(duration_seconds, 0.05))
            with self._lock:
                if generation != self._pulse_generation:
                    return
            self.push_buzzer_state(False, reason=f"{reason} reset")

        threading.Thread(target=_reset_after_pulse, name="arduino-buzzer-pulse", daemon=True).start()
        return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": self.available,
                "running": bool(self._thread and self._thread.is_alive()),
                "connected": self._connected,
                "port": self._port,
                "baudrate": self.baudrate,
                "preferred_port": self.preferred_port,
                "last_error": self._last_error,
                "message": self._last_message,
                "sample_count": self._sample_count,
                "last_reading": self._last_reading,
                "last_result": self._last_result,
                "history": list(self.history),
            }

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._connected:
                if not self._connect():
                    time.sleep(self.reconnect_delay)
                    continue

            try:
                raw_line = self._serial_connection.readline().decode("utf-8", errors="ignore").strip()
            except Exception as exc:  # pragma: no cover - depends on OS/device state
                self._close_connection(f"Lost Arduino connection while reading serial data: {exc}")
                time.sleep(self.reconnect_delay)
                continue

            if not raw_line:
                continue

            reading = parse_sensor_line(raw_line)
            if reading is None:
                with self._lock:
                    self._last_message = f"Ignored non-sensor serial line: {raw_line}"
                continue

            result = self.engine.predict(
                temperature=reading["temperature"],
                humidity=reading["humidity"],
                flame_signal=reading["flame"],
            )
            history_entry = self._build_history_entry(reading, result)

            with self._lock:
                self._sample_count += 1
                history_entry["index"] = self._sample_count
                self.history.appendleft(history_entry)
                self._last_reading = reading
                self._last_result = result
                self._last_error = None
                self._last_message = (
                    f"Live hardware feed active on {self._port}. "
                    f"Latest prediction: {result['prediction']}."
                )

            if self._last_buzzer_command is None or bool(result["buzzer"]) != self._last_buzzer_command:
                self.push_buzzer_state(bool(result["buzzer"]), reason="hardware stream")

    def _connect(self) -> bool:
        if not self.available:
            return False

        for candidate in self._candidate_ports():
            try:
                connection = serial.Serial(candidate, self.baudrate, timeout=self.read_timeout, write_timeout=1.0)
                time.sleep(2.0)
                connection.reset_input_buffer()
                with self._lock:
                    self._serial_connection = connection
                    self._connected = True
                    self._port = candidate
                    self._last_error = None
                    self._last_message = f"Connected to Arduino on {candidate}."
                self.push_buzzer_state(False, reason="startup")
                return True
            except Exception as exc:  # pragma: no cover - depends on local devices
                with self._lock:
                    self._last_error = f"Unable to open {candidate}: {exc}"
                    self._last_message = "Waiting for Arduino Uno serial feed."

        return False

    def _candidate_ports(self) -> list[str]:
        if self.preferred_port:
            return [self.preferred_port]

        ports = list(list_ports.comports())
        ranked = sorted(ports, key=self._port_rank)
        return [item.device for item in ranked]

    @staticmethod
    def _port_rank(port_info: Any) -> tuple[int, str]:
        description = f"{port_info.device} {getattr(port_info, 'description', '')} {getattr(port_info, 'manufacturer', '')}".lower()
        preferred_terms = ("arduino", "uno", "ch340", "usb serial", "silicon labs", "wch")
        score = 0 if any(term in description for term in preferred_terms) else 1
        return score, port_info.device

    def _close_connection(self, message: str) -> None:
        with self._lock:
            connection = self._serial_connection
            self._serial_connection = None
            self._connected = False
            self._port = None
            self._last_error = message
            self._last_message = "Waiting for Arduino Uno serial feed."

        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    @staticmethod
    def _build_history_entry(reading: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": 0,
            "generatedState": "HARDWARE",
            "temperature": reading["temperature"],
            "humidity": reading["humidity"],
            "flame": reading["flame"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "timestamp": reading["timestamp"],
            "local_alarm": reading["local_alarm"],
            "buzzer": result["buzzer"],
        }
