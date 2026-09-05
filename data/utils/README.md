# 触觉视频转 AutoencoderKL 图像数据集

`extract_tactile_video_frames.py` 用于把触觉视频转换成可以被 Diffusers
`train_autoencoderkl.py` 直接读取的图像目录。它针对硅胶指尖图像大部分时间不形变的特点，避免把大量重复的静态帧直接送进 AE 训练。对于 LeRobot 数据集，脚本会按 episode 重置参考帧，而不是把整个视频文件当成一个 episode。

## 快速开始

在 `TacVAE` 根目录执行：

```bash
./scripts/prepare_tactile_ae_dataset.sh
```

当前脚本顶部内置了一个可直接修改的配置，默认完整处理 `file-138.mp4`，并输出到：

```text
/home/xhz/TacVAE/data/tactile_ae_dataset_file138
```

需要处理其他文件时，直接修改 `scripts/prepare_tactile_ae_dataset.sh` 顶部的 `VIDEO_CHUNK`、`VIDEO_FILE` 和参数变量。脚本末尾也会追加命令行参数，因此可以临时覆盖配置；例如只做局部 smoke test：

```bash
./scripts/prepare_tactile_ae_dataset.sh \
  --max-images 200 \
  --max-frames-per-video 3000
```

`--sample-count` 是 `--max-images` 的别名。完整处理时不要设置 `--max-images` 或 `--max-frames-per-video`。

处理该触觉视频目录下的全部视频时，把脚本顶部配置改为：

```bash
VIDEO_CHUNK="all"
VIDEO_FILE="all"
```

此时脚本会递归处理 `VIDEO_KEY` 下的全部视频，自动使用 `validation-ratio=0.1`，并输出到：

```text
/home/xhz/TacVAE/data/tactile_ae_dataset_full
```

`VIDEO_CHUNK` 和 `VIDEO_FILE` 必须同时设为 `all`；不能只设置其中一个。全量处理时不要设置 `--max-videos`、`--max-images` 或 `--max-frames-per-video`。

直接调用 Python 脚本时，不指定 `--source` 会使用题目中给出的
`observation.images.left_wrist_left_tactile` 目录；使用 bash 包装脚本时，则以脚本顶部的 `SOURCE` 和 `OUTPUT_DIR` 配置为准。

脚本会自动激活 `tacvae` conda 环境。也可以直接执行 Python 脚本：

```bash
conda activate tacvae
python data/utils/extract_tactile_video_frames.py --help
```

## 筛选逻辑

标准 LeRobot 数据集会自动发现 `meta/episodes/`。该目录记录了每个 episode 在视频文件中的起止时间；也可以通过 `--episodes-dir` 显式指定。当 episode 元数据可用时，每个视频中的每个 episode 独立执行以下流程：

如果输入不是标准 LeRobot 数据集且找不到 episode 元数据，脚本会退回到每个视频使用一个参考帧，并打印 warning。对于包含多个 episode 的视频，不应依赖这个退回行为。

1. episode 的第一帧作为该 episode 的无形变参考图，并且始终作为 neutral 候选。
2. 按 `--frame-stride` 解码并评估视频帧。默认每 5 个源帧评估一次。
3. 把当前帧和该 episode 的参考帧缩小为灰度图，默认缩小宽度为 160 像素，并计算归一化平均绝对差：

   ```text
   difference = mean(abs(current_gray - reference_gray)) / 255
   ```

4. `difference > --threshold` 的帧视为 `deformed`，否则视为 `neutral`。默认阈值为 `0.015`。
5. 通过 `--min-deformed-gap` 限制同一视频中形变帧的最小源帧间距，避免一个持续形变被保存成大量近乎重复的图片。
6. 在每个 episode 的静态候选帧中均匀选择无形变图，使每个 episode 的无形变样本比例约为 `--neutral-ratio`。默认目标比例为 `25%`，并且每个 episode 至少保留其第一帧。

默认使用整幅图像计算差异。如果边缘、线缆或光照变化导致误检，可以使用归一化 ROI：

```bash
--roi 0.10 0.10 0.95 0.95
```

ROI 的顺序是 `X0 Y0 X1 Y1`，取值范围是 `0` 到 `1`。如果局部形变在整幅图中被稀释，可以缩小 ROI 只包含硅胶区域。默认的 `mean_abs` 适合相机基本固定的情况；也可以尝试：

```bash
--difference-metric changed_fraction \
--pixel-threshold 0.08 \
--threshold 0.03
```

此模式计算灰度变化超过 `pixel-threshold` 的像素比例。

脚本会扫描两遍视频：第一遍只做评分和选帧，第二遍只写入最终选择的帧。这样可以先知道形变帧数量，再控制 neutral/deformed 比例；源视频不会被修改。

## 输出格式

默认输出结构如下：

```text
data/tactile_ae_dataset/
├── train/
│   ├── chunk-000__file-000__episode-000000__frame-00000000__neutral.png
│   └── ...
├── validation/
│   └── ...
├── manifest.jsonl
├── summary.json
└── validation_images.txt
```

`train` 和 `validation` 中都是扁平的 RGB 图片目录，因此可以直接作为 `imagefolder` 的 `--train_data_dir`。不需要 caption、类别标签或其他标注；manifest 只用于追溯来源，不参与训练。

`manifest.jsonl` 每行记录：

- 输出图片路径
- 原始视频路径和视频 ID
- episode ID、episode 起止帧和 episode 参考帧
- 原始帧号和时间
- `neutral` 或 `deformed` 类别
- 差异分数和使用的阈值
- 原始图像尺寸及输出尺寸

`summary.json` 包含每个视频和 episode 的分数分布、解码帧数、参考帧和最终选择数量。调阈值时重点观察 `score_quantiles`：如果大量静态视频的 `p95` 已经接近阈值，应该提高阈值；如果明显形变的最大分数仍低于阈值，应该降低阈值或缩小 ROI。

## 常用参数

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--source` | 题目中的视频目录 | 视频文件或递归视频目录 |
| `--output-dir` | `data/tactile_ae_dataset` | 输出目录 |
| `--threshold` | `0.015` | 形变差异阈值，范围 `0..1` |
| `--sample-count` / `--max-images` | `0` | 最多输出图片数，`0` 表示不限 |
| `--max-videos` | `0` | 最多处理视频数，`0` 表示全部 |
| `--max-frames-per-video` | `0` | 每个视频最多扫描的源帧数，`0` 表示完整扫描；适合快速 smoke test |
| `--video-start` | `0` | 排序后跳过的视频数 |
| `--episodes-dir` | 自动发现 | LeRobot episode parquet 目录；用于在一个视频内按 episode 重置参考帧 |
| `--frame-stride` | `5` | 每隔多少源帧评估一帧 |
| `--min-deformed-gap` | `15` | 同一视频形变样本的最小帧间距 |
| `--neutral-ratio` | `0.25` | 目标无形变样本比例 |
| `--max-deformed-per-video` | `0` | 单视频形变样本上限 |
| `--max-neutral-per-video` | `0` | 单视频无形变样本上限 |
| `--validation-ratio` | `0.1` | 按视频切分验证集 |
| `--output-resolution` | `512` | 输出方形图片边长，必须被 8 整除 |
| `--resize-mode` | `letterbox` | `letterbox` 保留完整视野，避免中心裁剪硅胶区域 |
| `--image-format` | `png` | `png` 或 `jpeg` |
| `--dry-run` | 关闭 | 只扫描并打印统计，不写图片 |

建议先用小批量检查阈值：

```bash
./scripts/prepare_tactile_ae_dataset.sh \
  --max-videos 2 \
  --max-frames-per-video 3000 \
  --max-images 100 \
  --output-resolution 128 \
  --threshold 0.015 \
  --overwrite
```

只看选择统计时：

```bash
./scripts/prepare_tactile_ae_dataset.sh \
  --max-videos 2 \
  --threshold 0.015 \
  --dry-run
```

正式运行全部视频时，通常可以从下面的设置开始，再根据 `summary.json` 调整阈值：

```bash
./scripts/prepare_tactile_ae_dataset.sh \
  --source /home/xhz/OpenNeoData/aloha/videos/observation.images.left_wrist_left_tactile \
  --output-dir /home/xhz/TacVAE/data/tactile_ae_dataset \
  --threshold 0.015 \
  --frame-stride 5 \
  --min-deformed-gap 15 \
  --neutral-ratio 0.25 \
  --validation-ratio 0.1 \
  --output-resolution 512 \
  --image-format png \
  --overwrite
```

## 用于 AutoencoderKL 训练

转换器默认输出 `512x512` 方形 RGB 图片，且 `letterbox` 保留原始 `640x360` 触觉画面的完整内容。这和示例训练脚本的默认 `--resolution 512` 一致，可以避免示例脚本的 `CenterCrop` 再次截掉左右区域。对于包含多个 episode 的视频，参考帧由 `meta/episodes` 中对应 episode 的起始帧确定。

先读取脚本生成的验证图片列表：

```bash
mapfile -t VALIDATION_IMAGES < data/tactile_ae_dataset/validation_images.txt
```

然后从 `examples/research_projects/autoencoderkl` 目录启动训练：

```bash
cd diffusers/examples/research_projects/autoencoderkl
accelerate launch train_autoencoderkl.py \
  --pretrained_model_name_or_path stabilityai/sd-vae-ft-mse \
  --train_data_dir /home/xhz/TacVAE/data/tactile_ae_dataset/train \
  --image_column image \
  --validation_image "${VALIDATION_IMAGES[@]}" \
  --resolution 512 \
  --train_batch_size 4 \
  --num_train_epochs 100 \
  --learning_rate 4.5e-6 \
  --lr_scheduler cosine \
  --output_dir /home/xhz/TacVAE/outputs/tactile-aekl
```

如果选择了其他 `--output-resolution`，训练时要把 `--resolution` 改成相同的值。训练代码只读取图片列，不使用 `neutral/deformed` 字符串作为标签；这些类别主要用于检查数据比例。

验证集按视频划分，而不是按帧随机划分。同一个视频的帧不会同时进入 train 和 validation，可以减少相邻视频帧造成的验证指标虚高；但每个视频内部的参考帧和形变判断仍然按 episode 独立进行。

## 环境检查

脚本在 `tacvae` 环境中需要：

```bash
conda activate tacvae
python -c "import av, numpy, PIL, pyarrow, tqdm; print('dependencies are available')"
```

本实现使用 PyAV 解码 HEVC，不要求 OpenCV。数据集输出、训练输出、日志和原始数据链接都会由 TacVAE 根目录的 `.gitignore` 忽略，不应把生成图片或原始视频提交到 Git。
