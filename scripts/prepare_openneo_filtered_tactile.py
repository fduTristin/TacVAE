#!/usr/bin/env python3
"""Export event-sampled tactile images from a task-filtered OpenNeo manifest.

All tactile cameras from the listed robots share one train/validation split by
episode. Each video is committed independently so an interrupted export can be
resumed without re-decoding completed videos.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_tactile_index import select_episode  # noqa: E402
from data.utils import extract_tactile_video_frames as extractor  # noqa: E402


@dataclass(frozen=True)
class VideoJob:
    robot: str
    video_key: str
    path: Path
    dataset_root: Path
    video_id: str
    records: tuple[extractor.EpisodeRecord, ...]
    selected_episodes: frozenset[int]
    validation_episodes: frozenset[int]
    episode_tasks: dict[int, tuple[str, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="OpenNeo task manifest JSON")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory containing the robot datasets")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must be outside the source dataset")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-videos", type=int, default=0, help="Limit video files for a separate smoke export")
    parser.add_argument("--dry-run", action="store_true", help="Validate and count selected videos without decoding")
    args = parser.parse_args()
    if args.workers < 1 or args.max_videos < 0 or not 0 < args.validation_ratio < 1:
        parser.error("workers must be positive, max-videos non-negative, and validation-ratio in (0, 1)")
    return args


def task_names(robot_root: Path, robot_manifest: dict) -> set[str]:
    table = pq.read_table(robot_root / "meta/tasks.parquet").to_pylist()
    mapping = {int(row["task_index"]): row["__index_level_0__"] for row in table}
    entries = robot_manifest["tasks"]
    if [entry["task_index"] for entry in entries] != robot_manifest["task_indices"]:
        raise ValueError(f"Task indices and entries differ for {robot_root.name}")
    for entry in entries:
        if mapping.get(entry["task_index"]) != entry["task"]:
            raise ValueError(f"Unknown or renamed task for {robot_root.name}: {entry}")
    return {entry["task"] for entry in entries}


def build_jobs(manifest: dict, dataset_root: Path, validation_ratio: float, seed: int) -> tuple[list[VideoJob], dict]:
    jobs = []
    selection_summary = {}
    for robot, robot_manifest in manifest["robots"].items():
        robot_root = dataset_root / robot
        info = json.loads((robot_root / "meta/info.json").read_text())
        video_keys = sorted(key for key in info["features"] if "tactile" in key)
        if not video_keys:
            raise ValueError(f"No tactile cameras for {robot}")
        selected_names = task_names(robot_root, robot_manifest)
        columns = ["episode_index", "tasks", "length"]
        for key in video_keys:
            prefix = f"videos/{key}"
            columns.extend(f"{prefix}/{field}" for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"))
        episode_files = sorted((robot_root / "meta/episodes").rglob("*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"No episode metadata for {robot}")
        rows = pq.read_table(episode_files, columns=columns).to_pylist()
        selected_rows = [row for row in rows if selected_names.intersection(row["tasks"])]
        selected_ids = {int(row["episode_index"]) for row in selected_rows}
        if len(selected_ids) != robot_manifest["episode_count"]:
            raise ValueError(f"Episode count differs from manifest for {robot}")
        if sum(int(row["length"]) for row in selected_rows) != robot_manifest["frame_count"]:
            raise ValueError(f"Frame count differs from manifest for {robot}")
        shuffled = sorted(selected_ids)
        random.Random(f"{seed}:{robot}").shuffle(shuffled)
        validation_count = max(1, min(len(shuffled) - 1, math.ceil(len(shuffled) * validation_ratio)))
        validation_ids = frozenset(shuffled[:validation_count])
        selected_tasks = {int(row["episode_index"]): tuple(row["tasks"]) for row in selected_rows}
        selection_summary[robot] = {
            "tasks": len(robot_manifest["task_indices"]),
            "episodes": len(selected_ids),
            "frames_per_camera": sum(int(row["length"]) for row in selected_rows),
            "tactile_cameras": video_keys,
            "validation_episodes": len(validation_ids),
        }

        for key in video_keys:
            prefix = f"videos/{key}"
            grouped: dict[tuple[int, int], list[extractor.EpisodeRecord]] = defaultdict(list)
            selected_by_file: dict[tuple[int, int], set[int]] = defaultdict(set)
            for row in rows:
                episode_index = int(row["episode_index"])
                file_id = (int(row[f"{prefix}/chunk_index"]), int(row[f"{prefix}/file_index"]))
                grouped[file_id].append(
                    extractor.EpisodeRecord(
                        episode_index,
                        float(row[f"{prefix}/from_timestamp"]),
                        float(row[f"{prefix}/to_timestamp"]),
                    )
                )
                if episode_index in selected_ids:
                    selected_by_file[file_id].add(episode_index)
            for (chunk_index, file_index), episode_ids in sorted(selected_by_file.items()):
                path = robot_root / info["video_path"].format(
                    video_key=key, chunk_index=chunk_index, file_index=file_index
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                video_id = f"{robot}__{key.replace('.', '_')}__chunk-{chunk_index:03d}__file-{file_index:03d}"
                jobs.append(
                    VideoJob(
                        robot=robot,
                        video_key=key,
                        path=path,
                        dataset_root=robot_root,
                        video_id=video_id,
                        records=tuple(sorted(grouped[(chunk_index, file_index)], key=lambda item: item.from_timestamp)),
                        selected_episodes=frozenset(episode_ids),
                        validation_episodes=frozenset(episode_ids & validation_ids),
                        episode_tasks={episode: selected_tasks[episode] for episode in episode_ids},
                    )
                )
    if len({job.video_id for job in jobs}) != len(jobs):
        raise ValueError("Video IDs must be unique across robots and cameras")
    return jobs, selection_summary


def extraction_options() -> SimpleNamespace:
    """Match the existing 256x144 TacVAE export and event index policy."""
    return SimpleNamespace(
        threshold=0.01,
        difference_metric="mean_abs",
        pixel_threshold=0.08,
        metric_width=160,
        roi=(0.0, 0.0, 1.0, 1.0),
        frame_stride=2,
        min_deformed_gap=1,
        neutral_ratio=0.05,
        max_deformed_per_video=0,
        max_neutral_per_video=0,
        max_frames_per_video=0,
        output_width=256,
        output_height=144,
        resize_mode="letterbox",
        image_format="jpeg",
        jpeg_quality=95,
    )


def record_paths(output_dir: Path, job: VideoJob) -> tuple[Path, Path]:
    base = output_dir / ".records" / job.video_id
    return base.with_suffix(".jsonl"), base.with_suffix(".done.json")


def source_identity(job: VideoJob) -> dict:
    stat = job.path.stat()
    return {"path": str(job.path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def is_completed(output_dir: Path, job: VideoJob) -> bool:
    records_path, done_path = record_paths(output_dir, job)
    if not done_path.is_file() or not records_path.is_file():
        return False
    done = json.loads(done_path.read_text())
    if done["source"] != source_identity(job) or done["episode_indices"] != sorted(job.selected_episodes):
        raise ValueError(f"Source or task selection changed for completed video {job.path}")
    return True


def process_job(job: VideoJob, output_dir: Path, options: SimpleNamespace) -> tuple[str, int]:
    if is_completed(output_dir, job):
        return job.video_id, -1
    chunk_index, file_index = extractor.video_file_indices(job.path)
    scan = extractor.scan_video(
        job.path,
        "train",
        job.video_id,
        options,
        {(job.video_key, chunk_index, file_index): list(job.records)},
        job.dataset_root,
        episode_filter=job.selected_episodes,
    )
    for episode in scan.episodes:
        episode.split = "validation" if episode.boundary.episode_index in job.validation_episodes else "train"
    candidates = extractor.select_per_video(scan, options)
    by_episode: dict[int, list[extractor.SelectedFrame]] = defaultdict(list)
    for item in candidates:
        by_episode[item.candidate.episode_index].append(item)
    chosen: list[extractor.SelectedFrame] = []
    for episode_index in sorted(by_episode):
        rows = [
            {
                "frame_index": item.candidate.frame_index,
                "reference_frame_index": item.candidate.reference_frame_index,
                "difference": item.candidate.difference,
                "category": item.candidate.category,
                "item": item,
            }
            for item in by_episode[episode_index]
        ]
        selected_rows, _ = select_episode(rows, min_gap=15, run_gap=2, neutral_ratio=0.10)
        chosen.extend(row["item"] for row in selected_rows)
    chosen.sort(key=lambda item: item.candidate.frame_index)
    if not chosen:
        raise RuntimeError(f"No images selected from {job.path}")
    images = extractor.write_selected_images(chosen, output_dir, options, show_progress=False)
    records_path, done_path = record_paths(output_dir, job)
    temporary = records_path.with_suffix(records_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in chosen:
            key = (item.scan.video_id, item.candidate.episode_index, item.candidate.frame_index)
            row = extractor.selected_row(item, images[key], output_dir, options)
            row.update(robot=job.robot, video_key=job.video_key, tasks=job.episode_tasks[item.candidate.episode_index])
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    temporary.replace(records_path)
    done_path.write_text(
        json.dumps(
            {
                "source": source_identity(job),
                "episode_indices": sorted(job.selected_episodes),
                "selected_images": len(chosen),
            },
            indent=2,
        )
        + "\n"
    )
    return job.video_id, len(chosen)


def finalize(output_dir: Path, jobs: list[VideoJob], config: dict, selection_summary: dict) -> dict:
    counts = {split: dict.fromkeys(("neutral", "deformed"), 0) for split in ("train", "validation")}
    validation = {category: [] for category in ("neutral", "deformed")}
    manifest_path = output_dir / "manifest.jsonl"
    index_path = output_dir / "train_event_indexzhiying.txt"
    with manifest_path.open("w", encoding="utf-8") as manifest, index_path.open("w", encoding="utf-8") as index:
        for job in jobs:
            records_path, _ = record_paths(output_dir, job)
            if not is_completed(output_dir, job):
                raise RuntimeError(f"Video export is incomplete: {job.path}")
            for line in records_path.open(encoding="utf-8"):
                row = json.loads(line)
                image = output_dir / row["image"]
                if not image.is_file():
                    raise FileNotFoundError(image)
                manifest.write(line)
                counts[row["split"]][row["category"]] += 1
                if row["split"] == "train":
                    index.write(image.name + "\n")
                else:
                    validation[row["category"]].append(image)
    chosen_validation = [
        *extractor.select_evenly_paths(validation["neutral"], 5),
        *extractor.select_evenly_paths(validation["deformed"], 15),
    ]
    if not chosen_validation or sum(counts["train"].values()) == 0:
        raise ValueError("Train and validation exports must both be nonempty")
    (output_dir / "validation_images.txt").write_text(
        "\n".join(str(path.resolve()) for path in chosen_validation) + "\n"
    )
    summary = {
        "source": str(config["dataset_root"]),
        "output_dir": str(output_dir),
        "task_manifest": config["manifest"],
        "task_manifest_sha256": config["manifest_sha256"],
        "algorithm": {
            "reference": "first frame of each episode from LeRobot episode metadata",
            "validation_unit": "episode",
            "validation_ratio": config["validation_ratio"],
            "sampling": "event onset, peak, offset, periodic anchors, and episode-balanced neutral samples",
            **vars(extraction_options()),
        },
        "output": {"resolution": [256, 144], "resize_mode": "letterbox", "image_format": "jpeg"},
        "video_count": len(jobs),
        "episode_count": sum(len(job.selected_episodes) for job in jobs),
        "unique_episode_count": sum(item["exported_episodes"] for item in selection_summary.values()),
        "image_count": sum(sum(categories.values()) for categories in counts.values()),
        "split_counts": counts,
        "robots": selection_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "train_event_indexzhiying_summary.json").write_text(
        json.dumps({"selected_total": sum(counts["train"].values()), "selected_counts": counts["train"]}, indent=2)
        + "\n"
    )
    (output_dir / "EXPORT_COMPLETE.json").write_text(
        json.dumps(
            {
                "counts": {split: sum(categories.values()) for split, categories in counts.items()},
                "videos": len(jobs),
                "completed_at": time.time(),
            },
            indent=2,
        )
        + "\n"
    )
    return summary


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.is_relative_to(dataset_root) or output_dir == dataset_root:
        raise ValueError("Output directory must be outside the source dataset root")
    manifest = json.loads(manifest_path.read_text())
    jobs, selection_summary = build_jobs(manifest, dataset_root, args.validation_ratio, args.seed)
    if args.max_videos:
        jobs = jobs[: args.max_videos]
    for robot, item in selection_summary.items():
        robot_jobs = [job for job in jobs if job.robot == robot]
        item["exported_episodes"] = len({episode for job in robot_jobs for episode in job.selected_episodes})
        item["exported_videos"] = len(robot_jobs)
    config = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "dataset_root": str(dataset_root),
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "max_videos": args.max_videos,
        "extraction": vars(extraction_options()),
        "videos": [job.video_id for job in jobs],
    }
    print(
        f"Exporting {sum(item['exported_episodes'] for item in selection_summary.values()):,} episodes "
        f"across {len(jobs)} tactile videos; output: {output_dir}",
        flush=True,
    )
    if args.dry_run:
        print(json.dumps(selection_summary, indent=2), flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".records").mkdir(exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists():
        if json.loads(run_config_path.read_text()) != config:
            raise ValueError(f"Export settings changed; choose a new output directory: {output_dir}")
    else:
        if any(output_dir.iterdir()) and set(output_dir.iterdir()) != {output_dir / ".records"}:
            raise ValueError(f"Output directory already contains other files: {output_dir}")
        run_config_path.write_text(json.dumps(config, indent=2) + "\n")
    if (output_dir / "EXPORT_COMPLETE.json").is_file():
        print("Export already complete", flush=True)
        return
    pending = [job for job in jobs if not is_completed(output_dir, job)]
    print(f"Completed {len(jobs) - len(pending)} videos; preparing {len(pending)} videos", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_job, job, output_dir, extraction_options()): job for job in pending}
        completed = len(jobs) - len(pending)
        for future in as_completed(futures):
            video_id, count = future.result()
            completed += 1
            print(f"[{completed}/{len(jobs)}] {video_id}: {count:,} images", flush=True)
    summary = finalize(output_dir, jobs, config, selection_summary)
    print(
        f"Export complete: {summary['image_count']:,} images "
        f"({sum(summary['split_counts']['train'].values()):,} train, "
        f"{sum(summary['split_counts']['validation'].values()):,} validation)",
        flush=True,
    )


if __name__ == "__main__":
    main()
