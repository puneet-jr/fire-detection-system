from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import LABEL_MAP, RAW_DATASET_PATH
from .engine import HybridFireDetectionEngine


PLAYBACK_PATTERN = (0, 0, 0, 1, 0, 1, 2, 0, 1, 2)


@dataclass
class DatasetPlaybackService:
    dataset_path: str | None = None
    history_size: int = 40

    def __post_init__(self) -> None:
        self.dataset = pd.read_csv(self.dataset_path or RAW_DATASET_PATH)
        self.engine = HybridFireDetectionEngine()
        self.history: deque[dict[str, Any]] = deque(maxlen=self.history_size)
        self.event_log: deque[dict[str, Any]] = deque(maxlen=self.history_size)
        self.group_indices = {
            int(label): self.dataset[self.dataset["label"] == int(label)].reset_index(drop=True)
            for label in sorted(self.dataset["label"].dropna().unique().tolist())
        }
        self.group_pointers = {label: 0 for label in self.group_indices}
        self.pattern_index = 0
        self.sample_count = 0
        self.last_packet: dict[str, Any] | None = None

    def reset(self) -> None:
        self.engine.reset()
        self.history.clear()
        self.event_log.clear()
        self.pattern_index = 0
        self.sample_count = 0
        self.last_packet = None
        self.group_pointers = {label: 0 for label in self.group_indices}

    def status(self) -> dict[str, Any]:
        return {
            "available": True,
            "sample_count": self.sample_count,
            "history": list(self.history),
            "event_log": list(self.event_log),
            "last_packet": self.last_packet,
        }

    def next_packet(self) -> dict[str, Any]:
        row = self._next_row()
        result = self.engine.predict(
            temperature=float(row.temperature),
            humidity=float(row.humidity),
            flame_signal=int(row.flame_signal),
        )

        self.sample_count += 1
        packet = {
            "index": self.sample_count,
            "reading": {
                "temperature": round(float(row.temperature), 3),
                "humidity": round(float(row.humidity), 3),
                "flame": int(row.flame_signal),
            },
            "dataset": {
                "label_id": int(row.label),
                "label": LABEL_MAP.get(int(row.label), str(row.label)),
                "scenario": str(getattr(row, "scenario", "unknown")),
                "source": str(getattr(row, "source", "dataset")),
                "sequence_id": int(getattr(row, "sequence_id", self.sample_count)),
                "step": int(getattr(row, "step", 0)),
            },
            "result": result,
        }

        history_row = {
            "index": self.sample_count,
            "generatedState": packet["dataset"]["label"],
            "temperature": packet["reading"]["temperature"],
            "humidity": packet["reading"]["humidity"],
            "flame": packet["reading"]["flame"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "scenario": packet["dataset"]["scenario"],
            "source": packet["dataset"]["source"],
        }
        self.history.appendleft(history_row)
        self.last_packet = packet

        if packet["reading"]["flame"] == 1 or result["buzzer"]:
            self.event_log.appendleft(
                {
                    "index": self.sample_count,
                    "scenario": packet["dataset"]["scenario"],
                    "truth": packet["dataset"]["label"],
                    "prediction": result["prediction"],
                    "flame": packet["reading"]["flame"],
                    "buzzer": result["buzzer"],
                }
            )

        return packet

    def _next_row(self) -> Any:
        target_label = PLAYBACK_PATTERN[self.pattern_index % len(PLAYBACK_PATTERN)]
        self.pattern_index += 1
        if target_label not in self.group_indices:
            target_label = sorted(self.group_indices)[0]

        group = self.group_indices[target_label]
        pointer = self.group_pointers[target_label]
        row = group.iloc[pointer]
        self.group_pointers[target_label] = (pointer + 1) % len(group)
        return row
