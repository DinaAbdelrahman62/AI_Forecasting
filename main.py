import asyncio
import json
import math
import os
import pickle
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


BASE_DIR = Path(__file__).resolve().parent
BEST_MODELS_DIR = BASE_DIR / "best_models"
DATA_DIR = BASE_DIR / "data"
SMALL_PREDICT_CACHE = DATA_DIR / "predict_30_buildings.parquet"
FULL_PREDICT_CACHE = DATA_DIR / "predict_all.parquet"
DEFAULT_PREDICTION_CACHE = os.getenv("PREDICTION_CACHE_FILE", "small")
STREAM_DELAY_SECONDS = 30
PREDICTION_CACHE_ROW_LIMIT = 1_000_000
FULL_CACHE_MIN_ROWS = PREDICTION_CACHE_ROW_LIMIT * 2
REMOVED_CACHE_COLUMNS = {"absolute_error", "is_anomaly"}
PREDICTION_CACHE_CHOICES = {
    "small": SMALL_PREDICT_CACHE,
    "limited": SMALL_PREDICT_CACHE,
    "predict_30_buildings.parquet": SMALL_PREDICT_CACHE,
    "full": FULL_PREDICT_CACHE,
    "all": FULL_PREDICT_CACHE,
    "predict_11m_all.parquet": FULL_PREDICT_CACHE,
}

ENERGIES = ("electricity", "chilledwater", "steam", "hotwater", "solar", "gas")
BUILDING_METADATA_COLUMNS = [
    "building_id",
    "site_id",
    "primaryspaceusage",
    "sub_primaryspaceusage",
    "sqm",
    "sqft",
    "yearbuilt",
    "timezone",
    "real_lat",
    "real_lng",
    "lat",
    "lng",
    "local_x_m",
    "local_y_m",
]
LOCATION_COLUMNS = ("real_lat", "real_lng", "lat", "lng", "local_x_m", "local_y_m")


def _looks_like_parquet(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 8:
        return False
    with path.open("rb") as f:
        header = f.read(4)
        f.seek(-4, 2)
        footer = f.read(4)
    return header == b"PAR1" and footer == b"PAR1"


def _resolve_prediction_cache(cache: str | None = None) -> Path:
    cache_key = (cache or DEFAULT_PREDICTION_CACHE).strip().lower()
    if cache_key not in PREDICTION_CACHE_CHOICES:
        choices = ", ".join(sorted(PREDICTION_CACHE_CHOICES))
        raise HTTPException(status_code=400, detail=f"cache must be one of: {choices}")
    return PREDICTION_CACHE_CHOICES[cache_key]


def _is_limited_cache(cache_path: Path) -> bool:
    return cache_path == SMALL_PREDICT_CACHE


app = FastAPI(title="Digital Twin Forecasting API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PREDICT_CACHE = _resolve_prediction_cache()
prediction_cache_ready = _looks_like_parquet(PREDICT_CACHE)
prediction_status = {
    "state": "ready" if prediction_cache_ready else "not_started",
    "message": "Prediction cache is ready."
    if prediction_cache_ready
    else "Prediction cache has not been generated or is invalid.",
    "rows": None,
    "timestamps": None,
    "path": str(PREDICT_CACHE),
}
prediction_lock = Lock()
prediction_task: asyncio.Task | None = None
predictions_df: pd.DataFrame | None = None
prediction_timestamps: list[pd.Timestamp] | None = None
loaded_prediction_cache: Path | None = None


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return _json_default(value)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(_clean(data), default=_json_default)}\n\n"


def _load_prediction_cache(cache_path: Path = PREDICT_CACHE) -> pd.DataFrame:
    frame = pd.read_parquet(cache_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def _limit_to_complete_timestamps(frame: pd.DataFrame, row_limit: int = PREDICTION_CACHE_ROW_LIMIT) -> pd.DataFrame:
    if len(frame) <= row_limit:
        return frame

    counts = frame.groupby("timestamp", sort=True).size()
    cumulative = counts.cumsum()
    cutoff_position = cumulative.searchsorted(row_limit, side="left")
    cutoff_timestamp = counts.index[min(cutoff_position, len(counts) - 1)]
    return frame[frame["timestamp"] <= cutoff_timestamp].reset_index(drop=True)


def _cache_matches_row_limit(
    frame: pd.DataFrame,
    cache_path: Path = PREDICT_CACHE,
    row_limit: int = PREDICTION_CACHE_ROW_LIMIT,
) -> bool:
    if not _is_limited_cache(cache_path):
        return len(frame) >= FULL_CACHE_MIN_ROWS
    if len(frame) <= row_limit:
        return True

    counts = frame.groupby("timestamp", sort=True).size()
    if counts.empty:
        return True

    rows_before_last_timestamp = int(counts.iloc[:-1].sum())
    return rows_before_last_timestamp < row_limit <= len(frame)


def _cache_matches_schema(frame: pd.DataFrame) -> bool:
    return not REMOVED_CACHE_COLUMNS.intersection(frame.columns)


def _cache_file_needs_regeneration(cache_path: Path) -> bool:
    if not _looks_like_parquet(cache_path):
        return True

    parquet_file = pq.ParquetFile(cache_path)
    if REMOVED_CACHE_COLUMNS.intersection(parquet_file.schema.names):
        return True

    rows = parquet_file.metadata.num_rows
    return not _is_limited_cache(cache_path) and rows < FULL_CACHE_MIN_ROWS


def _load_manifest(energy: str) -> dict[str, Any]:
    with open(BEST_MODELS_DIR / energy / "manifest.pkl", "rb") as f:
        return pickle.load(f)


def _prediction_column(frame: pd.DataFrame, manifest: dict[str, Any]) -> pd.Series:
    special = frame["pred_special"] if "pred_special" in frame else pd.Series(np.nan, index=frame.index)
    if manifest["winner"] == "stack" and "pred_stack" in frame:
        base = frame["pred_stack"]
    elif manifest["winner"] == "blend":
        base = pd.Series(0.0, index=frame.index)
        for name, weight in manifest.get("weights", {}).items():
            if weight == 0:
                continue
            column = "naive" if name == "naive" else f"pred_{name}"
            if column in frame:
                base = base + (frame[column].fillna(0.0) * weight)
    else:
        column = f"pred_{manifest['winner']}"
        base = frame[column] if column in frame else frame["naive"]
    return special.combine_first(base)


def _read_locations() -> pd.DataFrame:
    path = DATA_DIR / "building_locations.parquet"
    if not path.exists():
        return pd.DataFrame(columns=BUILDING_METADATA_COLUMNS)
    locations = pd.read_parquet(path)
    keep = [c for c in BUILDING_METADATA_COLUMNS if c in locations.columns]
    return locations[keep].drop_duplicates("building_id")


def _coalesce_location_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in BUILDING_METADATA_COLUMNS:
        location_column = f"{column}_location"
        if location_column not in frame.columns:
            continue
        if column in frame.columns:
            frame[column] = frame[column].combine_first(frame[location_column])
        else:
            frame[column] = frame[location_column]
        frame = frame.drop(columns=[location_column])
    return frame


def _build_prediction_cache(cache_path: Path = PREDICT_CACHE) -> pd.DataFrame:
    parts = []
    locations = _read_locations()
    metadata_columns = [
        "building_id",
        "timestamp",
        "meter_type",
        "meter_reading",
        "site_id",
        "primaryspaceusage",
        "sub_primaryspaceusage",
        "sqm",
        "sqft",
        "timezone",
        "yearbuilt",
        "lat",
        "lng",
    ]

    for energy in ENERGIES:
        predictions_path = BEST_MODELS_DIR / energy / "test_predictions.parquet"
        features_path = BEST_MODELS_DIR / energy / "features.parquet"
        if not predictions_path.exists():
            continue

        manifest = _load_manifest(energy)
        preds = pd.read_parquet(predictions_path)
        preds["building_id"] = preds["building_id"].astype(str)
        preds["timestamp"] = pd.to_datetime(preds["timestamp"])
        preds["energy"] = energy
        preds["prediction"] = _prediction_column(preds, manifest)
        special_mask = preds["pred_special"].notna() if "pred_special" in preds.columns else pd.Series(False, index=preds.index)
        preds["model_used"] = np.where(special_mask, "special", manifest["winner"])
        preds["actual"] = preds["y"] if "y" in preds else np.nan

        pred_columns = [
            "building_id",
            "timestamp",
            "energy",
            "actual",
            "naive",
            "prediction",
            "model_used",
        ]
        preds = preds[pred_columns]

        if features_path.exists():
            available = set(pq.read_schema(features_path).names)
            feature_columns = [c for c in metadata_columns if c in available]
            features = pd.read_parquet(features_path, columns=feature_columns)
            features["building_id"] = features["building_id"].astype(str)
            features["timestamp"] = pd.to_datetime(features["timestamp"])
            features = features.drop_duplicates(["building_id", "timestamp"])
            preds = preds.merge(features, on=["building_id", "timestamp"], how="left")

        if not locations.empty:
            preds = preds.merge(locations, on="building_id", how="left", suffixes=("", "_location"))
            preds = _coalesce_location_columns(preds)

        if "meter_reading" not in preds.columns:
            preds["meter_reading"] = preds["actual"]
        else:
            preds["meter_reading"] = preds["meter_reading"].combine_first(preds["actual"])

        parts.append(preds)

    if not parts:
        raise RuntimeError("No prediction parquet files were found under best_models.")

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["timestamp", "building_id", "energy"]).reset_index(drop=True)
    if _is_limited_cache(cache_path):
        out = _limit_to_complete_timestamps(out)
    DATA_DIR.mkdir(exist_ok=True)
    out.to_parquet(cache_path, index=False)
    return out


def _update_prediction_status(
    state: str,
    message: str,
    frame: pd.DataFrame | None = None,
    cache_path: Path = PREDICT_CACHE,
) -> None:
    prediction_status.update(
        {
            "state": state,
            "message": message,
            "rows": None if frame is None else int(len(frame)),
            "timestamps": None if frame is None else int(frame["timestamp"].nunique()),
            "path": str(cache_path),
        }
    )


def generate_prediction_cache(cache_path: Path = PREDICT_CACHE) -> None:
    global loaded_prediction_cache, predictions_df, prediction_timestamps
    with prediction_lock:
        if cache_path.exists():
            try:
                frame = _load_prediction_cache(cache_path)
                if _cache_matches_row_limit(frame, cache_path) and _cache_matches_schema(frame):
                    predictions_df = frame
                    prediction_timestamps = sorted(frame["timestamp"].dropna().unique())
                    loaded_prediction_cache = cache_path
                    _update_prediction_status("ready", "Prediction cache is ready.", frame, cache_path)
                    return
                _update_prediction_status("running", "Prediction cache needs to be regenerated.", cache_path=cache_path)
            except Exception:
                _update_prediction_status(
                    "running",
                    "Existing prediction cache is invalid; regenerating from best_models.",
                    cache_path=cache_path,
                )

        try:
            _update_prediction_status("running", "Generating prediction cache from best_models.", cache_path=cache_path)
            frame = _build_prediction_cache(cache_path)
            predictions_df = frame
            prediction_timestamps = sorted(frame["timestamp"].dropna().unique())
            loaded_prediction_cache = cache_path
            _update_prediction_status("ready", "Prediction cache generated successfully.", frame, cache_path)
        except Exception as exc:
            _update_prediction_status("error", str(exc), cache_path=cache_path)
            raise


async def ensure_prediction_task(cache_path: Path = PREDICT_CACHE) -> asyncio.Task | None:
    global prediction_task
    if prediction_status["state"] == "ready" and cache_path == PREDICT_CACHE:
        return None
    if prediction_task is None or prediction_task.done():
        prediction_task = asyncio.create_task(asyncio.to_thread(generate_prediction_cache, cache_path))
    return prediction_task


def get_predictions(cache_path: Path = PREDICT_CACHE) -> pd.DataFrame:
    global loaded_prediction_cache, predictions_df, prediction_timestamps
    if predictions_df is None or loaded_prediction_cache != cache_path:
        if cache_path.exists():
            try:
                predictions_df = _load_prediction_cache(cache_path)
                if not _cache_matches_row_limit(predictions_df, cache_path) or not _cache_matches_schema(predictions_df):
                    _update_prediction_status("running", "Prediction cache needs to be regenerated.", cache_path=cache_path)
                    predictions_df = _build_prediction_cache(cache_path)
            except Exception:
                _update_prediction_status(
                    "running",
                    "Existing prediction cache is invalid; regenerating from best_models.",
                    cache_path=cache_path,
                )
                predictions_df = _build_prediction_cache(cache_path)
        else:
            raise HTTPException(status_code=202, detail=prediction_status)
        prediction_timestamps = sorted(predictions_df["timestamp"].dropna().unique())
        loaded_prediction_cache = cache_path
        _update_prediction_status("ready", "Prediction cache is ready.", predictions_df, cache_path)
    return predictions_df


def _metadata(row: pd.Series) -> dict[str, Any]:
    fields = [
        "primaryspaceusage",
        "sub_primaryspaceusage",
        "sqm",
        "sqft",
        "yearbuilt",
    ]
    return {field: row.get(field) for field in fields if field in row.index}


def _location(row: pd.Series) -> dict[str, Any]:
    return {field: row.get(field) for field in LOCATION_COLUMNS if field in row.index}


def _reading_batch(frame: pd.DataFrame, index: int) -> dict[str, Any]:
    timestamps = prediction_timestamps or sorted(frame["timestamp"].dropna().unique())
    if index < 0 or index >= len(timestamps):
        raise HTTPException(status_code=404, detail=f"reading_index must be between 0 and {len(timestamps) - 1}")

    timestamp = pd.Timestamp(timestamps[index])
    batch = frame[frame["timestamp"] == timestamp]
    buildings = []
    for building_id, group in batch.groupby("building_id", sort=True):
        first = group.iloc[0]
        readings = []
        for _, row in group.iterrows():
            readings.append(
                {
                    "energy": row.get("energy") or row.get("meter_type"),
                    "meter_reading": row.get("meter_reading"),
                }
            )
        buildings.append(
            {
                "building_id": building_id,
                "metadata": _metadata(first),
                "location": _location(first),
                "readings": readings,
            }
        )
    return {"reading_index": index, "timestamp": timestamp, "buildings": buildings}


def _prediction_batch(frame: pd.DataFrame, index: int) -> dict[str, Any]:
    timestamps = prediction_timestamps or sorted(frame["timestamp"].dropna().unique())
    if index < 0 or index >= len(timestamps):
        raise HTTPException(status_code=404, detail=f"prediction_index must be between 0 and {len(timestamps) - 1}")

    timestamp = pd.Timestamp(timestamps[index])
    batch = frame[frame["timestamp"] == timestamp]

    buildings = []
    for building_id, group in batch.groupby("building_id", sort=True):
        first = group.iloc[0]
        predictions = []
        for _, row in group.iterrows():
            predictions.append(
                {
                    "energy": row.get("energy") or row.get("meter_type"),
                    "prediction": row.get("prediction"),
                }
            )
        buildings.append(
            {
                "building_id": building_id,
                "metadata": _metadata(first),
                "location": _location(first),
                "predictions": predictions,
            }
        )
    return {"prediction_index": index, "timestamp": timestamp, "buildings": buildings}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "prediction_cache": prediction_status,
        "available_prediction_caches": {
            "small": str(SMALL_PREDICT_CACHE),
            "full": str(FULL_PREDICT_CACHE),
        },
    }


@app.get("/api/readings/stream")
async def stream_readings(cache: str | None = Query(None)) -> StreamingResponse:
    cache_path = _resolve_prediction_cache(cache)

    async def events():
        task = await ensure_prediction_task(cache_path)
        while task is not None and not task.done():
            yield _sse("status", prediction_status)
            await asyncio.sleep(1)
        if task is not None:
            await task

        frame = await asyncio.to_thread(get_predictions, cache_path)
        total = len(prediction_timestamps or [])
        for index in range(total):
            yield _sse("reading", _reading_batch(frame, index))
            await asyncio.sleep(STREAM_DELAY_SECONDS)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/readings/{reading_index}")
def read_reading(reading_index: int, cache: str | None = Query(None)) -> dict[str, Any]:
    frame = get_predictions(_resolve_prediction_cache(cache))
    return _clean(_reading_batch(frame, reading_index))


@app.get("/api/forecasting/predict")
async def predict_status(background_tasks: BackgroundTasks, cache: str | None = Query(None)) -> dict[str, Any]:
    cache_path = _resolve_prediction_cache(cache)
    needs_regeneration = _cache_file_needs_regeneration(cache_path)
    if needs_regeneration:
        _update_prediction_status("not_started", "Prediction cache has not been generated or is invalid.", cache_path=cache_path)
        background_tasks.add_task(generate_prediction_cache, cache_path)
    elif prediction_status["path"] != str(cache_path):
        _update_prediction_status("ready", "Prediction cache is ready.", cache_path=cache_path)
    return _clean({"all_predicted": not needs_regeneration, **prediction_status})


@app.get("/api/forecasting/predictions/{prediction_index}")
def forecasting_predictions(prediction_index: int, cache: str | None = Query(None)) -> dict[str, Any]:
    frame = get_predictions(_resolve_prediction_cache(cache))
    return _clean(_prediction_batch(frame, prediction_index))


@app.get("/api/predictions/stream")
async def stream_predictions(cache: str | None = Query(None)) -> StreamingResponse:
    cache_path = _resolve_prediction_cache(cache)

    async def events():
        task = await ensure_prediction_task(cache_path)
        while task is not None and not task.done():
            yield _sse("status", prediction_status)
            await asyncio.sleep(1)
        if task is not None:
            await task

        frame = await asyncio.to_thread(get_predictions, cache_path)
        total = len(prediction_timestamps or [])
        for index in range(total):
            yield _sse("prediction", _prediction_batch(frame, index))
            await asyncio.sleep(STREAM_DELAY_SECONDS)

    return StreamingResponse(events(), media_type="text/event-stream")
