from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import ensure_dir

RUN_PATTERN = re.compile(r"run_(\d{3})$")


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def save_json(data: Any, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(_to_jsonable(data), indent=2), encoding="utf-8")
    return output_path


def create_new_run_dir(results_root: str | Path) -> dict[str, Path | str]:
    results_path = Path(results_root)
    runs_dir = ensure_dir(results_path / "runs")
    ensure_dir(results_path / "aggregated" / "figures")
    ensure_dir(results_path / "aggregated" / "tables")
    ensure_dir(results_path / "aggregated" / "logs")

    existing_numbers: list[int] = []
    for path in runs_dir.iterdir():
        if path.is_dir():
            match = RUN_PATTERN.fullmatch(path.name)
            if match:
                existing_numbers.append(int(match.group(1)))

    next_number = (max(existing_numbers) + 1) if existing_numbers else 1
    run_id = f"run_{next_number:03d}"
    run_dir = ensure_dir(runs_dir / run_id)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "figures_dir": ensure_dir(run_dir / "figures"),
        "tables_dir": ensure_dir(run_dir / "tables"),
        "logs_dir": ensure_dir(run_dir / "logs"),
        "aggregated_dir": ensure_dir(results_path / "aggregated"),
        "aggregated_figures_dir": ensure_dir(results_path / "aggregated" / "figures"),
        "aggregated_tables_dir": ensure_dir(results_path / "aggregated" / "tables"),
        "aggregated_logs_dir": ensure_dir(results_path / "aggregated" / "logs"),
    }


@dataclass
class ExportManager:
    results_root: Path
    run_id: str
    timestamp: str
    run_dir: Path
    figures_dir: Path
    tables_dir: Path
    logs_dir: Path
    aggregated_dir: Path
    aggregated_figures_dir: Path
    aggregated_tables_dir: Path
    aggregated_logs_dir: Path
    manifest: dict[str, Any] = field(
        default_factory=lambda: {
            "run_id": "",
            "timestamp": "",
            "figures": [],
            "tables": [],
            "logs": [],
            "config": None,
            "summary": None,
        }
    )

    def __post_init__(self) -> None:
        self.manifest["run_id"] = self.run_id
        self.manifest["timestamp"] = self.timestamp

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.results_root))

    def _register_file(self, category: str, run_path: Path, aggregated_path: Path | None = None) -> None:
        entry = {"path": self._relative(run_path)}
        if aggregated_path is not None:
            entry["aggregated_path"] = self._relative(aggregated_path)
        self.manifest[category].append(entry)

    def _copy_to_aggregated(self, source_path: Path, aggregated_path: Path) -> Path:
        ensure_dir(aggregated_path.parent)
        shutil.copy2(source_path, aggregated_path)
        return aggregated_path

    def save_figure(self, fig: Any, filename: str, aggregate: bool = True) -> Path:
        run_path = self.figures_dir / filename
        fig.savefig(run_path, dpi=300, bbox_inches="tight")
        aggregated_path = None
        if aggregate:
            aggregated_path = self._copy_to_aggregated(run_path, self.aggregated_figures_dir / filename)
        self._register_file("figures", run_path, aggregated_path)
        return run_path

    def save_table(self, frame: pd.DataFrame, stem: str, aggregate: bool = True) -> dict[str, Path]:
        csv_path = self.tables_dir / f"{stem}.csv"
        json_path = self.tables_dir / f"{stem}.json"
        ensure_dir(csv_path.parent)
        frame.to_csv(csv_path, index=False)
        save_json(frame.to_dict(orient="records"), json_path)

        aggregated_csv = None
        aggregated_json = None
        if aggregate:
            aggregated_csv = self._copy_to_aggregated(csv_path, self.aggregated_tables_dir / csv_path.name)
            aggregated_json = self._copy_to_aggregated(json_path, self.aggregated_tables_dir / json_path.name)

        self._register_file("tables", csv_path, aggregated_csv)
        self._register_file("tables", json_path, aggregated_json)
        return {"csv": csv_path, "json": json_path}

    def save_log_dataframe(self, frame: pd.DataFrame, stem: str, aggregate: bool = True) -> dict[str, Path]:
        csv_path = self.logs_dir / f"{stem}.csv"
        json_path = self.logs_dir / f"{stem}.json"
        ensure_dir(csv_path.parent)
        frame.to_csv(csv_path, index=False)
        save_json(frame.to_dict(orient="records"), json_path)

        aggregated_csv = None
        aggregated_json = None
        if aggregate:
            aggregated_csv = self._copy_to_aggregated(csv_path, self.aggregated_logs_dir / csv_path.name)
            aggregated_json = self._copy_to_aggregated(json_path, self.aggregated_logs_dir / json_path.name)

        self._register_file("logs", csv_path, aggregated_csv)
        self._register_file("logs", json_path, aggregated_json)
        return {"csv": csv_path, "json": json_path}

    def save_log_json(self, data: Any, filename: str, aggregate: bool = True) -> Path:
        run_path = save_json(data, self.logs_dir / filename)
        aggregated_path = None
        if aggregate:
            aggregated_path = self._copy_to_aggregated(run_path, self.aggregated_logs_dir / filename)
        self._register_file("logs", run_path, aggregated_path)
        return run_path

    def write_config(self, config: dict[str, Any]) -> Path:
        config_path = save_json(config, self.run_dir / "config.json")
        self.manifest["config"] = {"path": self._relative(config_path)}
        return config_path

    def write_run_summary(self, summary: dict[str, Any]) -> Path:
        summary_path = save_json(summary, self.run_dir / "run_summary.json")
        aggregated_summary_path = save_json(summary, self.aggregated_dir / "summary.json")
        self.manifest["summary"] = {
            "path": self._relative(summary_path),
            "aggregated_path": self._relative(aggregated_summary_path),
        }
        return summary_path

    def write_manifest(self) -> Path:
        manifest_path = save_json(self.manifest, self.run_dir / "manifest.json")
        return manifest_path


def create_export_manager(results_root: str | Path) -> ExportManager:
    results_path = Path(results_root)
    run_paths = create_new_run_dir(results_path)
    return ExportManager(
        results_root=results_path,
        run_id=str(run_paths["run_id"]),
        timestamp=_timestamp_now(),
        run_dir=Path(run_paths["run_dir"]),
        figures_dir=Path(run_paths["figures_dir"]),
        tables_dir=Path(run_paths["tables_dir"]),
        logs_dir=Path(run_paths["logs_dir"]),
        aggregated_dir=Path(run_paths["aggregated_dir"]),
        aggregated_figures_dir=Path(run_paths["aggregated_figures_dir"]),
        aggregated_tables_dir=Path(run_paths["aggregated_tables_dir"]),
        aggregated_logs_dir=Path(run_paths["aggregated_logs_dir"]),
    )
