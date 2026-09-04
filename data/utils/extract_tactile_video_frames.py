#!/usr/bin/env python3
"""Build an ImageFolder-compatible dataset from tactile MP4 videos.

The first decoded frame of each video is treated as the neutral reference.
Frames whose downsampled grayscale difference from that reference exceeds the
configured threshold are retained as deformed samples. Neutral samples are
then selected separately so static frames cannot dominate the dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import av
except ImportError as error:  # pragma: no cover - depends on the local environment
    raise SystemExit(
        "PyAV is required. Activate the tacvae environment before running this script."
    ) from error

import numpy as np
from PIL import Image, ImageOps
from tqdm.auto import tqdm


VIDEO_SUFFIXES = {".avi", ".m4v", ".mov", ".mp4", ".webm", ".mkv"}


@dataclass(frozen=True)
class Candidate:
    episode_index: int
    reference_frame_index: int
    frame_index: int
    difference: float
    category: str


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class EpisodeBoundary:
    episode_index: int
    from_frame: int
    to_frame: int
    from_timestamp: float
    to_timestamp: float


@dataclass
class EpisodeScan:
    boundary: EpisodeBoundary
    neutral_candidates: list[Candidate]
    deformed_candidates: list[Candidate]
    scores: list[float]


@dataclass
class VideoScan:
    path: Path
    video_id: str
    split: str
    width: int
    height: int
    fps: float
    decoded_frames: int
    evaluated_frames: int
    episodes: list[EpisodeScan]
    score_min: float
    score_max: float
    score_mean: float
    score_quantiles: dict[str, float]


@dataclass(frozen=True)
class SelectedFrame:
    scan: VideoScan
    candidate: Candidate


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract reference-balanced tactile images from MP4 videos."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/home/xhz/OpenNeoData/aloha/videos/"
            "observation.images.left_wrist_left_tactile"
        ),
        help="Video file or directory containing video files. Defaults to the ALOHA tactile stream.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "data" / "tactile_ae_dataset",
        help="Output directory. It is created unless --dry-run is used.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.015,
        help="Normalized difference threshold in [0, 1]. Default: 0.015.",
    )
    parser.add_argument(
        "--difference-metric",
        choices=["mean_abs", "changed_fraction"],
        default="mean_abs",
        help="Reference difference metric. Default: mean_abs.",
    )
    parser.add_argument(
        "--pixel-threshold",
        type=float,
        default=0.08,
        help=(
            "Per-pixel threshold for changed_fraction, normalized to [0, 1]. "
            "Ignored by mean_abs. Default: 0.08."
        ),
    )
    parser.add_argument(
        "--metric-width",
        type=int,
        default=160,
        help="Width used for the difference calculation; height preserves aspect ratio. Default: 160.",
    )
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=(0.0, 0.0, 1.0, 1.0),
        help="Optional normalized ROI for difference calculation. Default: the full frame.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Evaluate every Nth source frame, starting at frame 0. Default: 5.",
    )
    parser.add_argument(
        "--min-deformed-gap",
        type=int,
        default=15,
        help=(
            "Minimum source-frame distance between retained deformed frames in one video. "
            "Default: 15, about 0.5 seconds at 30 FPS."
        ),
    )
    parser.add_argument(
        "--neutral-ratio",
        type=float,
        default=0.25,
        help=(
            "Approximate fraction of neutral samples per video. The first frame is always kept. "
            "Default: 0.25."
        ),
    )
    parser.add_argument(
        "--max-images",
        "--sample-count",
        dest="max_images",
        type=int,
        default=0,
        help="Global output image cap; 0 means no cap. --sample-count is an alias.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Use at most this many videos after --video-start; 0 means all videos.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=0,
        help=(
            "Scan at most this many source frames in each video; 0 means the complete video. "
            "Useful for fast smoke tests."
        ),
    )
    parser.add_argument(
        "--video-start",
        type=int,
        default=0,
        help="Skip this many videos after sorting source paths. Default: 0.",
    )
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing LeRobot episode parquet files. It is discovered automatically for a standard "
            "dataset layout; set this when the video directory is mounted separately."
        ),
    )
    parser.add_argument(
        "--max-deformed-per-video",
        type=int,
        default=0,
        help="Cap deformed samples per video; 0 means no cap.",
    )
    parser.add_argument(
        "--max-neutral-per-video",
        type=int,
        default=0,
        help="Cap neutral samples per video; 0 means no cap apart from neutral-ratio.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of videos assigned to validation. Set to 0 to disable validation. Default: 0.1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the video-level train/validation split. Default: 42.",
    )
    parser.add_argument(
        "--output-resolution",
        type=int,
        default=512,
        help="Square output image resolution. Default: 512.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=["letterbox", "stretch", "center_crop"],
        default="letterbox",
        help="Output geometry transform. Letterbox preserves the full tactile view.",
    )
    parser.add_argument(
        "--image-format",
        choices=["png", "jpeg"],
        default="png",
        help="Output image format. PNG avoids additional tactile compression. Default: png.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality when --image-format jpeg is selected. Default: 95.",
    )
    parser.add_argument(
        "--validation-image-count",
        type=int,
        default=8,
        help="Number of validation paths written to validation_images.txt. Default: 8.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and print selection statistics without writing images.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if not 0.0 <= args.pixel_threshold <= 1.0:
        raise ValueError("--pixel-threshold must be between 0 and 1")
    if args.metric_width <= 0:
        raise ValueError("--metric-width must be positive")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.min_deformed_gap < 0:
        raise ValueError("--min-deformed-gap must be non-negative")
    if not 0.0 <= args.neutral_ratio < 1.0:
        raise ValueError("--neutral-ratio must be at least 0 and less than 1")
    if args.max_images < 0:
        raise ValueError("--max-images must be non-negative")
    if args.max_videos < 0 or args.video_start < 0:
        raise ValueError("--max-videos and --video-start must be non-negative")
    if args.max_frames_per_video < 0:
        raise ValueError("--max-frames-per-video must be non-negative")
    if args.max_deformed_per_video < 0 or args.max_neutral_per_video < 0:
        raise ValueError("Per-video sample caps must be non-negative")
    if not 0.0 <= args.validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between 0 and 1")
    if args.output_resolution <= 0 or args.output_resolution % 8 != 0:
        raise ValueError("--output-resolution must be positive and divisible by 8")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.validation_image_count < 0:
        raise ValueError("--validation-image-count must be non-negative")

    x0, y0, x1, y1 = args.roi
    if not all(0.0 <= value <= 1.0 for value in args.roi):
        raise ValueError("All --roi values must be between 0 and 1")
    if not x0 < x1 or not y0 < y1:
        raise ValueError("--roi must satisfy X0 < X1 and Y0 < Y1")


def discover_videos(source: Path) -> list[Path]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported video suffix: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Video source does not exist: {source}")

    videos = sorted(
        (path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda path: path.as_posix(),
    )
    if not videos:
        raise FileNotFoundError(f"No video files found under: {source}")
    return videos


def find_dataset_root(source: Path) -> Path | None:
    start = source if source.is_dir() else source.parent
    for parent in (start, *start.parents):
        if (parent / "videos").is_dir() and (parent / "meta" / "episodes").is_dir():
            return parent
    return None


def video_key_for_path(path: Path, dataset_root: Path) -> str:
    videos_root = dataset_root / "videos"
    relative = path.relative_to(videos_root)
    if len(relative.parts) < 3:
        raise ValueError(
            f"Video path must follow <dataset>/videos/<video_key>/chunk-XXX/file-XXX.mp4: {path}"
        )
    return "/".join(relative.parts[:-2])


def episode_metadata_columns(video_key: str) -> tuple[str, ...]:
    prefix = f"videos/{video_key}"
    return (
        "episode_index",
        f"{prefix}/chunk_index",
        f"{prefix}/file_index",
        f"{prefix}/from_timestamp",
        f"{prefix}/to_timestamp",
    )


def load_episode_records(
    videos: list[Path], source: Path, episodes_dir: Path | None
) -> tuple[
    dict[tuple[str, int, int], list[EpisodeRecord]] | None,
    Path | None,
    Path | None,
]:
    dataset_root = find_dataset_root(source)
    if episodes_dir is None:
        if dataset_root is None:
            return None, None, None
        episodes_dir = dataset_root / "meta" / "episodes"
    else:
        episodes_dir = episodes_dir.expanduser().resolve()

    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"Episode metadata directory does not exist: {episodes_dir}")
    metadata_files = sorted(episodes_dir.rglob("*.parquet"), key=lambda path: path.as_posix())
    if not metadata_files:
        raise FileNotFoundError(f"No episode parquet files found under: {episodes_dir}")

    if dataset_root is None:
        dataset_root = episodes_dir.parent.parent
    video_keys = {video_key_for_path(path, dataset_root) for path in videos}

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - depends on the local environment
        raise SystemExit(
            "PyArrow is required to reset the reference frame for each episode. "
            "Install it with `pip install pyarrow` or pass a video source without episode metadata."
        ) from error

    records: dict[tuple[str, int, int], list[EpisodeRecord]] = {}
    for video_key in sorted(video_keys):
        columns = episode_metadata_columns(video_key)
        for metadata_file in metadata_files:
            available_columns = set(parquet.read_schema(metadata_file).names)
            if not set(columns).issubset(available_columns):
                continue
            for row in parquet.read_table(metadata_file, columns=list(columns)).to_pylist():
                video_id = (
                    video_key,
                    int(row[f"videos/{video_key}/chunk_index"]),
                    int(row[f"videos/{video_key}/file_index"]),
                )
                records.setdefault(video_id, []).append(
                    EpisodeRecord(
                        episode_index=int(row["episode_index"]),
                        from_timestamp=float(row[f"videos/{video_key}/from_timestamp"]),
                        to_timestamp=float(row[f"videos/{video_key}/to_timestamp"]),
                    )
                )

    for video_key, chunk_index, file_index in (
        (video_key_for_path(path, dataset_root), *video_file_indices(path)) for path in videos
    ):
        video_id = (video_key, chunk_index, file_index)
        if video_id not in records:
            raise ValueError(
                f"No episode metadata found for video {video_key} chunk={chunk_index} file={file_index}"
            )
        records[video_id].sort(key=lambda record: record.from_timestamp)

    return records, episodes_dir, dataset_root


def video_file_indices(path: Path) -> tuple[int, int]:
    try:
        chunk_name, file_name = path.parent.name, path.stem
        return int(chunk_name.removeprefix("chunk-")), int(file_name.removeprefix("file-"))
    except ValueError as error:
        raise ValueError(f"Video path must use chunk-XXX/file-XXX naming: {path}") from error


def make_episode_boundaries(records: list[EpisodeRecord], fps: float) -> list[EpisodeBoundary]:
    boundaries = [
        EpisodeBoundary(
            episode_index=record.episode_index,
            from_frame=round(record.from_timestamp * fps),
            to_frame=round(record.to_timestamp * fps),
            from_timestamp=record.from_timestamp,
            to_timestamp=record.to_timestamp,
        )
        for record in records
    ]
    if not boundaries or boundaries[0].from_frame != 0:
        raise ValueError("Episode metadata must start at video frame 0")
    for previous, current in zip(boundaries, boundaries[1:]):
        if previous.to_frame != current.from_frame:
            raise ValueError(
                f"Episode frame ranges are not contiguous: {previous.episode_index} ends at "
                f"{previous.to_frame}, {current.episode_index} starts at {current.from_frame}"
            )
    if any(boundary.to_frame <= boundary.from_frame for boundary in boundaries):
        raise ValueError("Episode metadata contains an empty or negative frame range")
    return boundaries


def make_video_id(path: Path, source: Path) -> str:
    if source.is_dir():
        relative = path.relative_to(source).with_suffix("").as_posix()
    else:
        relative = path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", relative.replace("/", "__"))


def split_videos(videos: list[Path], args: argparse.Namespace, source: Path) -> list[tuple[Path, str, str]]:
    selected = videos[args.video_start :]
    if args.max_videos:
        selected = selected[: args.max_videos]
    if not selected:
        raise ValueError("No videos remain after --video-start/--max-videos")

    shuffled = list(selected)
    random.Random(args.seed).shuffle(shuffled)
    if args.validation_ratio == 0.0:
        validation_count = 0
    else:
        if len(shuffled) < 2:
            raise ValueError("At least two videos are required when --validation-ratio is non-zero")
        validation_count = max(1, min(len(shuffled) - 1, math.ceil(len(shuffled) * args.validation_ratio)))

    validation = set(shuffled[:validation_count])
    result = []
    for path in selected:
        split = "validation" if path in validation else "train"
        result.append((path, split, make_video_id(path, source)))
    return result


def metric_gray(frame: av.VideoFrame, metric_width: int, roi: tuple[float, float, float, float]) -> np.ndarray:
    metric_height = max(1, round(metric_width * frame.height / frame.width))
    gray = frame.reformat(width=metric_width, height=metric_height, format="gray").to_ndarray()
    x0, y0, x1, y1 = roi
    left = min(metric_width - 1, max(0, math.floor(x0 * metric_width)))
    top = min(metric_height - 1, max(0, math.floor(y0 * metric_height)))
    right = min(metric_width, max(left + 1, math.ceil(x1 * metric_width)))
    bottom = min(metric_height, max(top + 1, math.ceil(y1 * metric_height)))
    return np.asarray(gray[top:bottom, left:right], dtype=np.float32)


def difference_score(
    reference: np.ndarray,
    current: np.ndarray,
    metric: str,
    pixel_threshold: float,
) -> float:
    delta = np.abs(current - reference) / 255.0
    if metric == "mean_abs":
        return float(delta.mean())
    return float(np.mean(delta > pixel_threshold))


def summarize_scores(scores: list[float]) -> tuple[float, float, float, dict[str, float]]:
    values = np.asarray(scores, dtype=np.float32)
    quantiles = {
        f"p{percentile}": float(np.quantile(values, percentile / 100.0))
        for percentile in (50, 90, 95, 99)
    }
    return float(values.min()), float(values.max()), float(values.mean()), quantiles


def scan_video(
    path: Path,
    split: str,
    video_id: str,
    args: argparse.Namespace,
    episode_records: dict[tuple[str, int, int], list[EpisodeRecord]] | None,
    dataset_root: Path | None,
) -> VideoScan:
    scores: list[float] = []
    decoded_frames = 0
    width = 0
    height = 0
    fps = 30.0

    with av.open(str(path)) as container:
        video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if video_stream is None:
            raise RuntimeError(f"No video stream found in {path}")
        video_stream.thread_type = "AUTO"
        width = int(video_stream.width or 0)
        height = int(video_stream.height or 0)
        if video_stream.average_rate is not None:
            fps = float(video_stream.average_rate)

        if episode_records is not None:
            if dataset_root is None:
                raise ValueError("A dataset root is required when episode metadata is provided")
            video_key = video_key_for_path(path, dataset_root)
            chunk_index, file_index = video_file_indices(path)
            records = episode_records[(video_key, chunk_index, file_index)]
            boundaries = make_episode_boundaries(records, fps)
        else:
            boundaries = [
                EpisodeBoundary(
                    episode_index=0,
                    from_frame=0,
                    to_frame=0,
                    from_timestamp=0.0,
                    to_timestamp=0.0,
                )
            ]

        episode_scans = [EpisodeScan(boundary, [], [], []) for boundary in boundaries]
        episode_cursor = 0
        reference: np.ndarray | None = None
        last_deformed_frame = -args.min_deformed_gap - 1

        for frame_index, frame in enumerate(container.decode(video=video_stream.index)):
            if args.max_frames_per_video and frame_index >= args.max_frames_per_video:
                break
            decoded_frames = frame_index + 1

            while (
                episode_cursor + 1 < len(boundaries)
                and frame_index >= boundaries[episode_cursor].to_frame
            ):
                episode_cursor += 1
                reference = None
                last_deformed_frame = -args.min_deformed_gap - 1

            boundary = boundaries[episode_cursor]
            if frame_index < boundary.from_frame or (
                boundary.to_frame and frame_index >= boundary.to_frame
            ):
                continue
            is_episode_start = frame_index == boundary.from_frame
            if frame_index % args.frame_stride != 0 and not is_episode_start:
                continue

            current = metric_gray(frame, args.metric_width, tuple(args.roi))
            episode_scan = episode_scans[episode_cursor]
            if is_episode_start:
                reference = current.copy()
                score = 0.0
            else:
                if reference is None:
                    raise RuntimeError(f"Missing reference frame for episode {boundary.episode_index}")
                score = difference_score(
                    reference,
                    current,
                    args.difference_metric,
                    args.pixel_threshold,
                )
            scores.append(score)
            episode_scan.scores.append(score)

            if is_episode_start or score <= args.threshold:
                episode_scan.neutral_candidates.append(
                    Candidate(
                        boundary.episode_index,
                        boundary.from_frame,
                        frame_index,
                        score,
                        "neutral",
                    )
                )
            elif frame_index - last_deformed_frame >= args.min_deformed_gap:
                if (
                    args.max_deformed_per_video == 0
                    or len(episode_scan.deformed_candidates) < args.max_deformed_per_video
                ):
                    episode_scan.deformed_candidates.append(
                        Candidate(
                            boundary.episode_index,
                            boundary.from_frame,
                            frame_index,
                            score,
                            "deformed",
                        )
                    )
                    last_deformed_frame = frame_index

    if reference is None or not scores:
        raise RuntimeError(f"Could not decode an evaluable frame from {path}")
    if episode_records is not None and decoded_frames >= boundaries[-1].to_frame:
        if decoded_frames != boundaries[-1].to_frame:
            raise ValueError(
                f"Decoded frame count mismatch for {path}: decoded {decoded_frames}, "
                f"episode metadata ends at {boundaries[-1].to_frame}"
            )
    score_min, score_max, score_mean, score_quantiles = summarize_scores(scores)
    return VideoScan(
        path=path,
        video_id=video_id,
        split=split,
        width=width,
        height=height,
        fps=fps,
        decoded_frames=decoded_frames,
        evaluated_frames=len(scores),
        episodes=episode_scans,
        score_min=score_min,
        score_max=score_max,
        score_mean=score_mean,
        score_quantiles=score_quantiles,
    )


def select_evenly(candidates: list[Candidate], count: int) -> list[Candidate]:
    if count <= 0 or not candidates:
        return []
    if count >= len(candidates):
        return list(candidates)

    indices = np.rint(np.linspace(0, len(candidates) - 1, count)).astype(int).tolist()
    selected: list[Candidate] = []
    seen: set[int] = set()
    for index in indices:
        if index not in seen:
            selected.append(candidates[index])
            seen.add(index)
    if len(selected) < count:
        for index, candidate in enumerate(candidates):
            if index not in seen:
                selected.append(candidate)
                seen.add(index)
                if len(selected) == count:
                    break
    return selected


def select_per_video(scan: VideoScan, args: argparse.Namespace) -> list[SelectedFrame]:
    all_deformed = [
        candidate
        for episode in scan.episodes
        for candidate in episode.deformed_candidates
    ]
    if args.max_deformed_per_video:
        deformed_keys = {
            (candidate.episode_index, candidate.frame_index)
            for candidate in select_evenly(all_deformed, args.max_deformed_per_video)
        }
    else:
        deformed_keys = {
            (candidate.episode_index, candidate.frame_index)
            for candidate in all_deformed
        }

    neutral: list[Candidate] = []
    deformed: list[Candidate] = []
    for episode in scan.episodes:
        episode_deformed = [
            candidate
            for candidate in episode.deformed_candidates
            if (candidate.episode_index, candidate.frame_index) in deformed_keys
        ]
        neutral_count = (
            math.ceil(len(episode_deformed) * args.neutral_ratio / (1.0 - args.neutral_ratio))
            if episode_deformed
            else 1
        )
        neutral.extend(select_evenly(episode.neutral_candidates, max(1, neutral_count)))
        deformed.extend(episode_deformed)

    if args.max_neutral_per_video and len(neutral) > args.max_neutral_per_video:
        references = [
            candidate for candidate in neutral if candidate.frame_index == candidate.reference_frame_index
        ]
        if args.max_neutral_per_video >= len(references):
            remaining = select_evenly(
                [candidate for candidate in neutral if candidate not in references],
                args.max_neutral_per_video - len(references),
            )
            neutral = [*references, *remaining]
        else:
            neutral = select_evenly(neutral, args.max_neutral_per_video)

    selected = [SelectedFrame(scan, candidate) for candidate in [*neutral, *deformed]]
    selected.sort(key=lambda item: item.candidate.frame_index)
    return selected


def select_global_cap(
    selected: list[SelectedFrame], args: argparse.Namespace
) -> list[SelectedFrame]:
    if args.max_images == 0 or len(selected) <= args.max_images:
        return selected

    neutral = [item for item in selected if item.candidate.category == "neutral"]
    deformed = [item for item in selected if item.candidate.category == "deformed"]
    if args.max_images == 1:
        return select_evenly(neutral or selected, 1)
    if not deformed:
        references = {
            (item.scan.video_id, item.candidate.episode_index): item
            for item in neutral
            if item.candidate.frame_index == item.candidate.reference_frame_index
        }
        if args.max_images >= len(references):
            other_neutral = [
                item
                for item in neutral
                if item.candidate.frame_index != item.candidate.reference_frame_index
            ]
            result = [
                *references.values(),
                *select_evenly(other_neutral, args.max_images - len(references)),
            ]
            result.sort(key=lambda item: (item.scan.split, item.scan.video_id, item.candidate.frame_index))
            return result
        indices = np.rint(np.linspace(0, len(selected) - 1, args.max_images)).astype(int)
        return [selected[index] for index in indices]

    references = {
        (item.scan.video_id, item.candidate.episode_index): item
        for item in neutral
        if item.candidate.frame_index == item.candidate.reference_frame_index
    }
    desired_neutral = max(1, round(args.max_images * args.neutral_ratio))
    desired_neutral = max(desired_neutral, len(references))
    desired_neutral = min(desired_neutral, len(neutral), args.max_images)
    desired_deformed = args.max_images - desired_neutral
    chosen_neutral = select_evenly(neutral, desired_neutral)
    chosen_deformed = select_evenly(deformed, desired_deformed)
    result = [*chosen_neutral, *chosen_deformed]

    # Keep at least one reference image per episode whenever the cap permits it.
    if args.max_images >= len(references):
        present = {
            (item.scan.video_id, item.candidate.episode_index)
            for item in result
        }
        missing = [
            item for key, item in references.items() if key not in present
        ]
        if missing:
            replaceable = [
                item
                for item in result
                if item.candidate.category == "neutral"
                and item.candidate.frame_index != item.candidate.reference_frame_index
            ]
            for replacement, required in zip(replaceable, missing):
                result[result.index(replacement)] = required

    result.sort(key=lambda item: (item.scan.split, item.scan.video_id, item.candidate.frame_index))
    return result


def transform_image(image: Image.Image, resolution: int, mode: str) -> Image.Image:
    image = image.convert("RGB")
    if mode == "letterbox":
        scale = min(resolution / image.width, resolution / image.height)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        resized = image.resize(size, Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (resolution, resolution), (0, 0, 0))
        offset = ((resolution - resized.width) // 2, (resolution - resized.height) // 2)
        canvas.paste(resized, offset)
        return canvas
    if mode == "stretch":
        return image.resize((resolution, resolution), Image.Resampling.BICUBIC)
    return ImageOps.fit(
        image,
        (resolution, resolution),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )


def output_path(output_dir: Path, item: SelectedFrame, image_format: str) -> Path:
    extension = "jpg" if image_format == "jpeg" else "png"
    filename = (
        f"{item.scan.video_id}__episode-{item.candidate.episode_index:06d}"
        f"__frame-{item.candidate.frame_index:08d}"
        f"__{item.candidate.category}.{extension}"
    )
    return output_dir / item.scan.split / filename


def write_selected_images(
    selected: list[SelectedFrame], output_dir: Path, args: argparse.Namespace
) -> dict[tuple[str, int, int], Path]:
    by_video: dict[Path, list[SelectedFrame]] = {}
    for item in selected:
        by_video.setdefault(item.scan.path, []).append(item)
    written: dict[tuple[str, int, int], Path] = {}

    for source, items in tqdm(sorted(by_video.items(), key=lambda pair: pair[0].as_posix()), desc="Writing images"):
        targets = {item.candidate.frame_index: item for item in items}
        with av.open(str(source)) as container:
            video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
            if video_stream is None:
                raise RuntimeError(f"No video stream found in {source}")
            video_stream.thread_type = "AUTO"
            for frame_index, frame in enumerate(container.decode(video=video_stream.index)):
                if args.max_frames_per_video and frame_index >= args.max_frames_per_video:
                    break
                item = targets.get(frame_index)
                if item is None:
                    continue
                image = Image.fromarray(frame.to_rgb().to_ndarray())
                image = transform_image(image, args.output_resolution, args.resize_mode)
                path = output_path(output_dir, item, args.image_format)
                path.parent.mkdir(parents=True, exist_ok=True)
                if args.image_format == "jpeg":
                    image.save(path, format="JPEG", quality=args.jpeg_quality, subsampling=0)
                else:
                    image.save(path, format="PNG", compress_level=3)
                written[
                    (item.scan.video_id, item.candidate.episode_index, item.candidate.frame_index)
                ] = path

    if len(written) != len(selected):
        raise RuntimeError(f"Expected to write {len(selected)} images but wrote {len(written)}")
    return written


def selected_row(item: SelectedFrame, path: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "split": item.scan.split,
        "image": str(path.relative_to(output_dir)),
        "source_video": str(item.scan.path),
        "video_id": item.scan.video_id,
        "episode_index": item.candidate.episode_index,
        "frame_index": item.candidate.frame_index,
        "frame_time_sec": item.candidate.frame_index / item.scan.fps,
        "category": item.candidate.category,
        "difference": item.candidate.difference,
        "difference_metric": args.difference_metric,
        "threshold": args.threshold,
        "reference_frame_index": item.candidate.reference_frame_index,
        "source_width": item.scan.width,
        "source_height": item.scan.height,
        "output_resolution": args.output_resolution,
    }


def write_outputs(
    selected: list[SelectedFrame],
    scans: list[VideoScan],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = write_selected_images(selected, output_dir, args)
    rows: list[dict[str, Any]] = []
    for item in selected:
        key = (item.scan.video_id, item.candidate.episode_index, item.candidate.frame_index)
        rows.append(selected_row(item, image_paths[key], output_dir, args))

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in rows:
            manifest.write(json.dumps(row, ensure_ascii=True) + "\n")

    validation_paths = [
        image_paths[(item.scan.video_id, item.candidate.episode_index, item.candidate.frame_index)]
        for item in selected
        if item.scan.split == "validation"
    ]
    validation_paths = [path.resolve() for path in select_evenly_paths(validation_paths, args.validation_image_count)]
    (output_dir / "validation_images.txt").write_text(
        "\n".join(str(path) for path in validation_paths) + ("\n" if validation_paths else ""),
        encoding="utf-8",
    )

    by_split: dict[str, list[SelectedFrame]] = {"train": [], "validation": []}
    for item in selected:
        by_split[item.scan.split].append(item)
    summary = {
        "source": str(args.source.expanduser().resolve()),
        "output_dir": str(output_dir),
        "algorithm": {
            "reference": (
                "first frame of each episode from LeRobot episode metadata"
                if args.episodes_dir
                else "first decoded frame of each video"
            ),
            "episodes_dir": str(args.episodes_dir.expanduser().resolve()) if args.episodes_dir else None,
            "difference_metric": args.difference_metric,
            "threshold": args.threshold,
            "pixel_threshold": args.pixel_threshold,
            "metric_width": args.metric_width,
            "roi": list(args.roi),
            "frame_stride": args.frame_stride,
            "max_frames_per_video": args.max_frames_per_video,
            "min_deformed_gap": args.min_deformed_gap,
            "neutral_ratio": args.neutral_ratio,
            "max_images": args.max_images,
            "max_deformed_per_video": args.max_deformed_per_video,
            "max_neutral_per_video": args.max_neutral_per_video,
        },
        "output": {
            "resolution": [args.output_resolution, args.output_resolution],
            "resize_mode": args.resize_mode,
            "image_format": args.image_format,
        },
        "video_count": len(scans),
        "episode_count": sum(len(scan.episodes) for scan in scans),
        "image_count": len(selected),
        "split_counts": {
            split: {
                "videos": len({item.scan.video_id for item in items}),
                "images": len(items),
                "neutral": sum(item.candidate.category == "neutral" for item in items),
                "deformed": sum(item.candidate.category == "deformed" for item in items),
            }
            for split, items in by_split.items()
        },
        "videos": [
            {
                "video_id": scan.video_id,
                "split": scan.split,
                "path": str(scan.path),
                "width": scan.width,
                "height": scan.height,
                "fps": scan.fps,
                "decoded_frames": scan.decoded_frames,
                "evaluated_frames": scan.evaluated_frames,
                "score_min": scan.score_min,
                "score_max": scan.score_max,
                "score_mean": scan.score_mean,
                "score_quantiles": scan.score_quantiles,
                "episode_count": len(scan.episodes),
                "episodes": [
                    {
                        "episode_index": episode.boundary.episode_index,
                        "from_frame": episode.boundary.from_frame,
                        "to_frame": episode.boundary.to_frame,
                        "from_timestamp": episode.boundary.from_timestamp,
                        "to_timestamp": episode.boundary.to_timestamp,
                        "evaluated_frames": len(episode.scores),
                        "selected_neutral": sum(
                            item.scan.video_id == scan.video_id
                            and item.candidate.episode_index == episode.boundary.episode_index
                            and item.candidate.category == "neutral"
                            for item in selected
                        ),
                        "selected_deformed": sum(
                            item.scan.video_id == scan.video_id
                            and item.candidate.episode_index == episode.boundary.episode_index
                            and item.candidate.category == "deformed"
                            for item in selected
                        ),
                    }
                    for episode in scan.episodes
                ],
                "selected_neutral": sum(
                    item.scan.video_id == scan.video_id and item.candidate.category == "neutral"
                    for item in selected
                ),
                "selected_deformed": sum(
                    item.scan.video_id == scan.video_id and item.candidate.category == "deformed"
                    for item in selected
                ),
            }
            for scan in scans
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def select_evenly_paths(paths: list[Path], count: int) -> list[Path]:
    if count <= 0:
        return []
    if count >= len(paths):
        return paths
    indices = np.rint(np.linspace(0, len(paths) - 1, count)).astype(int)
    return [paths[index] for index in indices]


def print_statistics(scans: list[VideoScan], selected: list[SelectedFrame], dry_run: bool) -> None:
    counts: dict[str, dict[str, int]] = {
        "train": {"neutral": 0, "deformed": 0},
        "validation": {"neutral": 0, "deformed": 0},
    }
    for item in selected:
        counts[item.scan.split][item.candidate.category] += 1
    prefix = "Dry run: " if dry_run else ""
    print(f"{prefix}videos={len(scans)} images={len(selected)}")
    for split in ("train", "validation"):
        print(
            f"{prefix}{split}: neutral={counts[split]['neutral']} "
            f"deformed={counts[split]['deformed']}"
        )
    no_deformation = sum(
        bool(episode.scores) and not episode.deformed_candidates
        for scan in scans
        for episode in scan.episodes
    )
    if no_deformation:
        print(f"Warning: {no_deformation} episode(s) had no frame above the threshold.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    source = args.source.expanduser().resolve()
    videos = discover_videos(source)
    episode_records, episodes_dir, dataset_root = load_episode_records(videos, source, args.episodes_dir)
    args.episodes_dir = episodes_dir
    if args.episodes_dir is None and episode_records is None:
        print("Warning: no episode metadata found; using one reference frame per video.")
    elif episodes_dir is not None:
        print(f"Using episode metadata from {episodes_dir}")
    split_video_entries = split_videos(videos, args, source)
    scans: list[VideoScan] = []
    for path, split, video_id in tqdm(split_video_entries, desc="Scanning videos"):
        scans.append(scan_video(path, split, video_id, args, episode_records, dataset_root))

    selected: list[SelectedFrame] = []
    for scan in scans:
        selected.extend(select_per_video(scan, args))
    selected = select_global_cap(selected, args)
    print_statistics(scans, selected, args.dry_run)

    if args.dry_run:
        return
    if not selected:
        raise RuntimeError("No frames were selected")
    write_outputs(selected, scans, args.output_dir, args)
    print(f"Wrote dataset to {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
