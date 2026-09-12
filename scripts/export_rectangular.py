"""Re-export exactly the existing manifest's frames, without changing splits."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import time

import av
from PIL import Image


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def export_video(task):
    video, items, destination = task
    destination = Path(destination)
    started = time.monotonic()
    targets = dict(items)
    if len(targets) != len(items):
        raise ValueError(f'Duplicate frame indices in {video}')
    count = 0
    last = max(targets)
    with av.open(video) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = 2
        for index, frame in enumerate(container.decode(video=0)):
            if index in targets:
                path = destination / targets[index]
                if path.exists():
                    with Image.open(path) as image:
                        if image.size != (256, 144):
                            raise ValueError(f'Unexpected existing image size: {path}')
                        image.load()
                else:
                    image = frame.to_image()
                    if image.width * 9 != image.height * 16:
                        raise ValueError(f'Unexpected source aspect ratio: {video}')
                    image = image.resize((256, 144), Image.Resampling.BICUBIC)
                    tmp = path.with_suffix('.tmp')
                    image.save(tmp, format='JPEG', quality=95, subsampling=0)
                    tmp.replace(path)
                count += 1
            if index >= last:
                break
    if count != len(items):
        raise RuntimeError(f'{video}: expected {len(items)}, exported {count}')
    return {'video': video, 'images': count, 'seconds': time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if source == output:
        raise ValueError('Output must be distinct from source')
    output.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'validation']:
        (output / split).mkdir(exist_ok=True)
    # Lock persists for the job lifetime; a failed process releases it automatically.
    import fcntl
    lock = (output / 'export.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'EXPORT_COMPLETE.json').exists():
        print('Already complete', flush=True)
        return
    started = time.time()
    groups = {}
    counts = {'train': 0, 'validation': 0}
    manifest_tmp = output / 'manifest.jsonl.tmp'
    with (source / 'manifest.jsonl').open() as handle, manifest_tmp.open('w') as target:
        for line in handle:
            row = json.loads(line)
            rel = Path(row['image'])
            if rel.is_absolute() or '..' in rel.parts or row['split'] not in counts or rel.parts[0] != row['split']:
                raise ValueError('Invalid manifest path or split')
            groups.setdefault(row['source_video'], []).append((row['frame_index'], row['image']))
            counts[row['split']] += 1
            row['output_resolution'] = [256, 144]
            target.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest_tmp.replace(output / 'manifest.jsonl')
    summary = json.loads((source / 'summary.json').read_text())
    if sum(counts.values()) != summary['image_count']:
        raise ValueError('Manifest and summary image count differ')
    completed = []
    tasks = [(video, items, str(output)) for video, items in groups.items()]
    print(json.dumps({'started': started, 'videos': len(tasks), 'counts': counts, 'workers': args.workers}), flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(export_video, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed.append(result)
            status = {'completed_videos': len(completed), 'total_videos': len(tasks),
                      'completed_images': sum(x['images'] for x in completed),
                      'total_images': sum(counts.values()), 'elapsed_seconds': time.time()-started,
                      'last_video': result}
            atomic_json(output / 'progress.json', status)
            print(json.dumps(status), flush=True)
    for split, expected in counts.items():
        actual = sum(1 for item in os.scandir(output / split) if item.name.endswith('.jpg'))
        if actual != expected:
            raise ValueError(f'{split}: expected {expected}, found {actual}')
    validation_paths = [output / Path(p).relative_to(source) for p in (source / 'validation_images.txt').read_text().splitlines() if p]
    for path in validation_paths:
        with Image.open(path) as image:
            image.load()
            assert image.size == (256, 144)
    (output / 'validation_images.txt').write_text('\n'.join(map(str, validation_paths)) + '\n')
    summary['output_dir'] = str(output)
    summary['output'] = {'resolution': [256, 144], 'resize_mode': 'aspect_preserving_resize', 'image_format': 'jpeg'}
    summary['reexport_source_manifest'] = str(source / 'manifest.jsonl')
    atomic_json(output / 'summary.json', summary)
    atomic_json(output / 'EXPORT_COMPLETE.json', {'counts': counts, 'videos': len(tasks), 'completed_at': time.time(), 'elapsed_seconds': time.time()-started})
    print('EXPORT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
