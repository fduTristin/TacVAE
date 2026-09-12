"""Build an event-aware training index from an existing tactile manifest."""
import argparse
import json
import math
from pathlib import Path


def select_evenly(rows, count):
    if count <= 0:
        return []
    if count >= len(rows):
        return list(rows)
    if count == 1:
        return [rows[0]]
    return [rows[round(i * (len(rows) - 1) / (count - 1))] for i in range(count)]


def select_episode(rows, min_gap, run_gap, neutral_ratio):
    neutral = [row for row in rows if row['category'] == 'neutral']
    deformed = [row for row in rows if row['category'] == 'deformed']
    runs = []
    for row in deformed:
        if not runs or row['frame_index'] - runs[-1][-1]['frame_index'] > run_gap:
            runs.append([])
        runs[-1].append(row)
    chosen_deformed = []
    for run in runs:
        chosen = {0, len(run) - 1, max(range(len(run)), key=lambda i: run[i]['difference'])}
        anchor_frame = run[0]['frame_index']
        for i, row in enumerate(run[1:], 1):
            if row['frame_index'] - anchor_frame >= min_gap:
                chosen.add(i)
                anchor_frame = row['frame_index']
        chosen_deformed.extend(run[i] for i in sorted(chosen))
    desired_neutral = max(1, math.ceil(len(chosen_deformed) * neutral_ratio / (1 - neutral_ratio)))
    references = [row for row in neutral if row['frame_index'] == row['reference_frame_index']]
    others = [row for row in neutral if row['frame_index'] != row['reference_frame_index']]
    chosen_neutral = references[:1]
    chosen_neutral.extend(select_evenly(others, max(0, desired_neutral - len(chosen_neutral))))
    return sorted([*chosen_neutral, *chosen_deformed], key=lambda row: row['frame_index']), len(runs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--min-deformed-gap', type=int, default=15)
    parser.add_argument('--event-run-gap', type=int, default=2)
    parser.add_argument('--neutral-ratio', type=float, default=0.10)
    args = parser.parse_args()
    if args.min_deformed_gap <= 0 or args.event_run_gap <= 0 or not 0 <= args.neutral_ratio < 1:
        raise ValueError('Invalid sampling parameters')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + '.tmp')
    counts = {'neutral': 0, 'deformed': 0}
    source_counts = {'neutral': 0, 'deformed': 0}
    episodes = events = 0
    current_key = None
    episode_rows = []

    def write_episode(handle, rows):
        nonlocal episodes, events
        if not rows:
            return
        chosen, run_count = select_episode(rows, args.min_deformed_gap, args.event_run_gap, args.neutral_ratio)
        episodes += 1
        events += run_count
        for row in chosen:
            image = Path(row['image'])
            if image.parts[0] != 'train' or image.name != image.as_posix().split('/')[-1]:
                raise ValueError(f'Unexpected training path: {image}')
            handle.write(image.name + '\n')
            counts[row['category']] += 1

    with args.manifest.open() as source, tmp.open('w') as target:
        for line in source:
            row = json.loads(line)
            if row['split'] != 'train':
                continue
            source_counts[row['category']] += 1
            key = (row['video_id'], row['episode_index'])
            if current_key is not None and key != current_key:
                write_episode(target, episode_rows)
                episode_rows = []
            current_key = key
            episode_rows.append(row)
        write_episode(target, episode_rows)
    tmp.replace(args.output)
    result = {
        'source_manifest': str(args.manifest.resolve()), 'index_file': str(args.output.resolve()),
        'source_counts': source_counts, 'selected_counts': counts, 'selected_total': sum(counts.values()),
        'episodes': episodes, 'deformation_events': events, 'min_deformed_gap': args.min_deformed_gap,
        'event_run_gap': args.event_run_gap, 'neutral_ratio_target': args.neutral_ratio,
        'policy': 'event onset, peak, offset, periodic anchors, and episode-balanced neutral samples',
    }
    args.summary.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
