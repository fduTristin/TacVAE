# 触觉视频转 AutoencoderKL 图像数据集

## OpenNeo 多本体任务筛选

`scripts/prepare_openneo_filtered_tactile.py` 接受 OpenNeo 任务筛选 JSON，从所列本体的所有触觉相机中只选匹配任务的 episode。训练/验证按 `(本体, episode_index)` 统一划分，因此同一 episode 的不同相机不会跨 split。它沿用下文的 256×144 JPEG、参考帧差分和 event index 参数，但只写出最终训练索引选中的图片，避免先导出大量随后弃用的图片。每个视频单独记录完成状态；使用相同参数和输出目录重新运行即可继续。

在含有 `av`、`Pillow`、`numpy`、`pyarrow`、`tqdm` 的 Python 环境中，从 TacVAE 根目录执行：

```bash
python scripts/prepare_openneo_filtered_tactile.py \
  --manifest ../../assets/openneo_downstream_tasks.json \
  --dataset-root ../../../OpenNeoData \
  --output-dir /data/workspace/VTLA/tacvae_openneo_downstream_256x144 \
  --workers 16
```

先加 `--dry-run` 可核对 episode 和视频文件数量，不写图片。输出目录应与源数据分开；完成后包含 `train/`、`validation/`、`manifest.jsonl`、`train_event_indexzhiying.txt`、`validation_images.txt`、`summary.json` 和 `EXPORT_COMPLETE.json`，供 AutoencoderKL 训练入口使用。`--max-videos` 可在另一个输出目录做局部验证。

`extract_tactile_video_frames.py` 用于把触觉视频转换成可以被 Diffusers
`train_autoencoderkl.py` 直接读取的图像目录。它针对硅胶指尖图像大部分时间不形变的特点，避免把大量重复的静态帧直接送进 AE 训练。对于 LeRobot 数据集，脚本会按 episode 重置参考帧，而不是把整个视频文件当成一个 episode。

## 快速开始

在 `TacVAE` 根目录执行：

```bash
./scripts/prepare_tactile_ae_dataset.sh
```

当前脚本顶部内置了一个可直接修改的配置，默认完整处理 `VIDEO_KEY` 下的全部视频，并直接输出为 `256x144` JPEG 图片到：

```text
/home/xhz/TacVAE/data/tactile_ae_dataset_256x144zhiying
```

抽帧完成后，脚本还会自动生成 `train_event_indexzhiying.txt` 和对应的统计文件；训练时可以直接使用这个索引减少连续形变帧的重复。单文件模式默认使用 `validation_ratio=0.1`，并按 episode 划分 train/validation；同一个 episode 的帧不会同时出现在两个 split 中。

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
/home/xhz/TacVAE/data/tactile_ae_dataset_256x144zhiying
```

全量模式的划分方式由脚本中的 `VALIDATION_UNIT` 决定；当前默认配置按完整 episode 划分，同一个 episode 的帧不会同时进入 train 和 validation。

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
├── validation_images.txt
├── train_event_indexzhiying.txt
├── train_event_indexzhiying_summary.json
└── EXPORT_COMPLETE.json
```

`train` 和 `validation` 中都是扁平的 RGB 图片目录，因此可以直接作为 `imagefolder` 的 `--train_data_dir`。包装脚本默认写入 `256x144` JPEG；直接调用 Python 脚本时仍可使用方形默认值，或通过 `--output-size` 指定矩形尺寸。不需要 caption、类别标签或其他标注；manifest 只用于追溯来源，不参与训练。

`manifest.jsonl` 每行记录：

- 输出图片路径
- 原始视频路径和视频 ID
- episode ID、episode 起止帧和 episode 参考帧
- 原始帧号和时间
- `neutral` 或 `deformed` 类别
- 差异分数和使用的阈值
- 原始图像尺寸及输出尺寸

`validation_images.txt` 默认从 validation split 均匀选择 5 张 neutral 和 15 张 deformed 图片，用于训练期间在 WandB 中比较固定样本的重建效果。这里只写已有图片的路径，不会复制图片。

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
| `--validation-ratio` | `0.1` | 按 `--validation-unit` 切分验证集 |
| `--validation-unit` | `video` | `video` 按视频切分，`episode` 按每个视频内的 episode 切分 |
| `--validation-neutral-count` | `5` | 写入 `validation_images.txt` 的 neutral 图片数量 |
| `--validation-deformed-count` | `15` | 写入 `validation_images.txt` 的 deformed 图片数量 |
| `--validation-image-count` | 未设置 | 兼容旧用法：不区分类别地选择总图片数，不能和上述两个参数同时使用 |
| `--output-resolution` | `512` | 输出方形图片边长，必须被 8 整除；与 `--output-size` 二选一 |
| `--output-size` | 未设置 | 输出 `宽 高`，两个尺寸必须被 8 整除；设置后优先于 `--output-resolution` |
| `--resize-mode` | `letterbox` | `letterbox` 保留完整视野，避免中心裁剪硅胶区域 |
| `--image-format` | `png` | `png` 或 `jpeg` |
| `--dry-run` | 关闭 | 只扫描并打印统计，不写图片 |

建议先用小批量检查阈值：

```bash
./scripts/prepare_tactile_ae_dataset.sh \
  --max-videos 2 \
  --max-frames-per-video 3000 \
  --max-images 100 \
  --output-size 128 128 \
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
  --output-size 512 512 \
  --image-format png \
  --overwrite
```

## 用于 AutoencoderKL 训练

直接调用转换器时默认输出 `512x512` 方形 RGB 图片；bash 包装脚本则配置为直接输出 `256x144`，`letterbox` 在原始 `640x360` 比例下不会产生额外裁剪或填充。对于包含多个 episode 的视频，参考帧由 `meta/episodes` 中对应 episode 的起始帧确定。

当前正式训练配置已经集中在 `scripts/train_tactile_autoencoderkl.sh`。全量 `256x144` 数据集和 event index 生成完成后，直接执行：

```bash
./scripts/train_tactile_autoencoderkl.sh
```

脚本默认使用：

```text
dataset: data/tactile_ae_dataset_256x144zhiying
GPU: configured by `GPU_IDS` and `NUM_PROCESSES` in the training script
per-device batch size: 8
max train steps: 100000
KL scale: 1.0416666667e-5 on per-latent mean KL, warmed up over 5000 steps
perceptual scale: 0.0
GAN: disabled until step 1000000000
output: outputs/zhiying-scratch-256x144-eventindex
```

需要调整 GPU、batch size、训练步数或 loss 权重时，直接编辑该脚本顶部的配置变量。也可以临时追加训练参数覆盖脚本默认值，例如：

```bash
./scripts/train_tactile_autoencoderkl.sh \
  --max_train_steps 1000 \
  --output_dir /mnt/sdc/xhz/TacVAE/outputs/tactile-aekl-smoke-full
```

训练脚本会先检查 `train/`、`validation_images.txt` 和所有验证图片是否存在，再启动 `accelerate`。`kl_loss` 现在是 FP32 计算的 per-latent mean，`kl_sum` 仅作为原始总量诊断；训练日志还会记录 `weighted_kl_loss`、原始及 scaling-factor 校正后的 `latent_mean/latent_std`、`posterior_mean_abs` 和 `posterior_std`，用于判断 encoder 的分布是否在改善。不要在正式训练命令中添加 `--scale_lr`，除非你明确希望根据 batch size 和 GPU 数量放大学习率。

先读取脚本生成的验证图片列表：

```bash
mapfile -t VALIDATION_IMAGES < data/tactile_ae_dataset_256x144zhiying/validation_images.txt
```

然后从 `examples/research_projects/autoencoderkl` 目录启动训练：

```bash
cd diffusers/examples/research_projects/autoencoderkl
accelerate launch train_autoencoderkl.py \
  --pretrained_model_name_or_path stabilityai/sd-vae-ft-mse \
  --train_data_dir /home/xhz/TacVAE/data/tactile_ae_dataset_256x144zhiying/train \
  --image_column image \
  --validation_image "${VALIDATION_IMAGES[@]}" \
  --image_width 256 \
  --image_height 144 \
  --train_batch_size 4 \
  --num_train_epochs 100 \
  --learning_rate 4.5e-6 \
  --lr_scheduler cosine \
  --output_dir /home/xhz/TacVAE/outputs/tactile-aekl
```

如果选择了其他 `--output-size`，训练时要把 `--image_width` 和 `--image_height` 改成相同的值。训练代码只读取图片列，不使用 `neutral/deformed` 字符串作为标签；这些类别主要用于检查数据比例。

默认验证集按视频划分，而不是按帧随机划分。同一个视频的帧不会同时进入 train 和 validation，可以减少相邻视频帧造成的验证指标虚高。需要在单个视频内建立验证集时，使用 `--validation-unit episode`；此时按完整 episode 划分，同一个 episode 的帧不会同时进入 train 和 validation，而每个 episode 的参考帧和形变判断仍然独立进行。

## 环境检查

脚本在 `tacvae` 环境中需要：

```bash
conda activate tacvae
python -c "import av, numpy, PIL, pyarrow, tqdm; print('dependencies are available')"
```

本实现使用 PyAV 解码 HEVC，不要求 OpenCV。数据集输出、训练输出、日志和原始数据链接都会由 TacVAE 根目录的 `.gitignore` 忽略，不应把生成图片或原始视频提交到 Git。
