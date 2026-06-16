# Geo-AVS 中文使用指南

本文档面向第一次使用本项目的同学，目标是让你知道 Geo-AVS 做什么、怎么准备环境、怎么跑 UAVScenes 完整实验、每个输出文件是什么意思，以及如何继续扩展到其他点云数据集。

## 1. 任务定义

Geo-AVS 解决的是 **无人机场景无监督/免训练自动开放词汇 3D 点云语义分割**：

1. 输入：UAV 图像、LiDAR 点云、相机内参/投影信息。
2. 不输入人工类别表，先由 VLM/captioner 自动生成当前场景候选词表。
3. 用 Caption2Tag 和遥感同义词表把自然语言词规范成遥感类别词。
4. 用 SegEarth-OV3/SAM3 对这些词生成 2D logits 和 presence score。
5. 把 2D evidence 通过 UAVScenes 投影关系提升到 3D superpoint。
6. 用 QFE 对 superpoint footprint 上的 evidence 做鲁棒聚合。
7. 通过 CaptionFreq、SegEarth presence、QFE_TopP、AreaCoverage 验证自动词表。
8. 输出 superpoint 和 point 级预测，并给出闭集指标与开放词表指标。

一句话概括：Geo-AVS 不是把 2D mask 硬投到点云，而是把“自动词表发现”和“2D foundation evidence”整理成可验证的 3D superpoint 分割流程。

## 2. 目录结构

核心包：

- `geo_avs/autovoc/`：VLM caption、Caption2Tag、遥感同义词归一化、词表评分与验证。
- `geo_avs/evidence/`：SegEarth/SAM3 logits 和 presence score 缓存格式。
- `geo_avs/superpoints/`：voxel superpoint，可外接 GrowSP/SPT/EZ-SP partition。
- `geo_avs/lifting/`：SPFE/QFE 2D-to-3D evidence lifting。
- `geo_avs/segmentation/`：verified vocabulary 到 superpoint/point label。
- `geo_avs/evaluation/`：Hungarian Acc/mIoU、NMI、ARI、开放词表指标。
- `scripts/`：从 00 到 07 的完整流水线脚本，以及已验证的 UAVScenes 主实验脚本。

## 3. 环境准备

推荐在服务器上使用：

```bash
cd /home/work/research/geo_avs
pip install -r requirements.txt
```

如果要跑满血版 VLM captioner，需要安装或准备：

- Qwen2.5-VL 权重，例如 `/home/work/research/weights/qwen/Qwen2.5-VL-7B-Instruct`
- SegEarth-OV-3 仓库，例如 `/home/work/research/upstreams_full/sources/SegEarth-OV-3-main`
- SAM3 checkpoint，例如 `weights/sam3/sam3.pt`

如果这些大模型权重暂时不可用，`01_generate_vlm_autovoc.py` 和 `02_extract_segearth_evidence.py` 都提供 fallback，只用于 smoke test，不建议作为论文主结果。

## 4. UAVScenes 完整实验

服务器默认路径：

```bash
DATASET_ROOT=/home/work/research/datasets/UAVScenes/extracted
ROOT_DIR=/home/work/research/geo_avs
```

一键完整实验：

```bash
cd /home/work/research/geo_avs
bash scripts/run_full_geo_avs_uavscenes.sh
```

这会依次执行：

1. 生成 100 张 UAVScenes 图像列表。
2. 运行 `01_generate_vlm_autovoc.py`，生成 `cache/geo_avs/uavscenes_vlm_autovoc_100.json`。
3. 调用已验证的 `geo_avs_qfe_autovoc_uavscenes.py`，把 caption-json 真正接入 AutoVoc selection。
4. 输出 `results/geo_avs_full_vlm_qfe_uavscenes100/geo_avs_qfe_autovoc_report.json`。
5. 运行 `07_evaluate_open_vocab.py`，输出 `open_vocab_eval.json`。
6. 扫描服务器上 SensatUrban、DALES、SemanticKITTI、S3DIS、ScanNet、H3D 等数据集可用性。

常用环境变量：

```bash
QWEN_MODEL=/home/work/research/weights/qwen/Qwen2.5-VL-7B-Instruct \
SAM3_CKPT=weights/sam3/sam3.pt \
DEVICE=cuda \
bash scripts/run_full_geo_avs_uavscenes.sh
```

如果只想跑已经验证过的 QFE AutoVoc 快速路径：

```bash
bash scripts/run_geo_avs_qfe_autovoc_fullscene100.sh
```

## 5. 分步运行

生成图像索引：

```bash
python3 scripts/00_prepare_uavscenes_index.py \
  --dataset-root /home/work/research/datasets/UAVScenes/extracted \
  --limit 100 \
  --out-image-list cache/uavscenes_image_list_100.txt
```

生成 VLM 自动词表：

```bash
python3 scripts/01_generate_vlm_autovoc.py \
  --image-list cache/uavscenes_image_list_100.txt \
  --model /home/work/research/weights/qwen/Qwen2.5-VL-7B-Instruct \
  --out cache/geo_avs/uavscenes_vlm_autovoc_100.json
```

提取 SegEarth/SAM3 evidence cache：

```bash
python3 scripts/02_extract_segearth_evidence.py \
  --image-list cache/uavscenes_image_list_100.txt \
  --autovoc-json cache/geo_avs/uavscenes_vlm_autovoc_100.json \
  --segearth-root /home/work/research/upstreams_full/sources/SegEarth-OV-3-main \
  --sam3-checkpoint weights/sam3/sam3.pt \
  --out-dir cache/geo_avs/evidence/uavscenes_100
```

构建 superpoint cache：

```bash
python3 scripts/03_build_superpoints.py \
  --dataset-root /home/work/research/datasets/UAVScenes/extracted \
  --frames-file results/uavscenes_fullscene_100_frames.txt \
  --method voxel \
  --target-superpoints 420 \
  --out-dir cache/geo_avs/superpoints/voxel_100
```

QFE lifting：

```bash
python3 scripts/04_lift_evidence_qfe.py \
  --evidence-dir cache/geo_avs/evidence/uavscenes_100 \
  --superpoint-dir cache/geo_avs/superpoints/voxel_100 \
  --out-dir cache/geo_avs/qfe/voxel_100
```

验证自动词表：

```bash
python3 scripts/05_verify_autovoc.py \
  --autovoc-json cache/geo_avs/uavscenes_vlm_autovoc_100.json \
  --qfe-dir cache/geo_avs/qfe/voxel_100 \
  --top-k 8 \
  --out cache/geo_avs/uavscenes_verified_autovoc_100.json
```

生成分割预测：

```bash
python3 scripts/06_run_geo_avs_segmentation.py \
  --qfe-dir cache/geo_avs/qfe/voxel_100 \
  --verified-vocab cache/geo_avs/uavscenes_verified_autovoc_100.json \
  --out-dir results/geo_avs_vlm_qfe_voxel_100
```

评估：

```bash
python3 scripts/07_evaluate_open_vocab.py \
  --pred-dir results/geo_avs_vlm_qfe_voxel_100 \
  --out results/geo_avs_vlm_qfe_voxel_100/eval.json
```

## 6. 结果文件怎么看

主结果目录一般是：

```bash
results/geo_avs_full_vlm_qfe_uavscenes100
```

重要文件：

- `geo_avs_qfe_autovoc_report.json`：完整主实验结果，包含每帧词表、四组方法指标、平均指标。
- `GEO_AVS_QFE_AUTOVOC_REPORT_CN.md`：Markdown 摘要。
- `geo_avs_qfe_autovoc_metrics.png`：指标柱状图。
- `uavscenes_qfe_autovoc_3davs_format.json`：兼容 3D-AVS 风格的 `{scene:frame: [terms...]}` 自动词表。
- `open_vocab_eval.json`：开放词表诊断指标。
- `public_dataset_probe.json`：服务器公开数据集扫描结果。

主表指标含义：

- `hungarian_acc`：无监督聚类/分割常用的最优匹配准确率。
- `hungarian_miou`：经过 Hungarian matching 后的 mIoU。
- `nmi`：Normalized Mutual Information。
- `ari`：Adjusted Rand Index。
- `full_candidate_gap`：完整候选词表上界与自动词表结果的差距，越小越说明 AutoVoc 验证有效。

## 7. 论文叙事建议

可以准确表述为：

> We formulate UAV-oriented 3D Auto-Vocabulary Segmentation and build a complete training-free pipeline that connects VLM-based scene vocabulary proposal, remote-sensing Caption2Tag normalization, SegEarth/SAM3 open-vocabulary evidence, superpoint-level QFE lifting, and open-vocabulary evaluation on UAVScenes.

不要夸大为：

- 完整复现 3D-AVS 官方所有模块。
- 完整训练 Superpoint Transformer。
- 直接用 SAM3 完成 3D 分割。

更准确的说法是：

- 借鉴 3D-AVS 的 AutoVoc 范式，但为 UAV 场景重建了 captioner/Caption2Tag/verification 链路。
- 使用 SegEarth-OV3/SAM3 作为 2D foundation evidence extractor。
- 使用 superpoint token 作为 2D-to-3D evidence lifting 单元。
- 提出 QFE 作为 UAV 投影噪声下的 superpoint footprint evidence 聚合策略。

## 8. 扩展到其他数据集

先扫描服务器数据：

```bash
python3 scripts/probe_public_pointcloud_datasets.py \
  --roots /home/work/research /home/work/research/datasets \
  --out results/public_dataset_probe.json
```

是否能做完整 Geo-AVS 多模态测试，取决于三个条件：

1. 是否有图像和点云配对。
2. 是否有相机内外参或可恢复投影关系。
3. 是否有标签可计算客观指标。

SensatUrban、DALES、S3DIS、ScanNet、SemanticKITTI 如果只有点云或缺少配对图像/相机投影，仍可做“点云单模态退化实验”或“伪正射图/BEV 渲染实验”，但不能等同于 UAVScenes 上的完整多模态 Geo-AVS。

## 9. 常见问题

**Q: 没有 Qwen2.5-VL 权重怎么办？**  
A: `01_generate_vlm_autovoc.py --backend auto` 会 fallback 到图像先验 captioner，只用于检查流程是否通。论文主实验应使用真实 VLM。

**Q: 没有 SAM3 checkpoint 怎么办？**  
A: 可以用 `02_extract_segearth_evidence.py --fallback` 检查 cache 格式和 QFE 流水线，但论文主结果必须用 SegEarth/SAM3 logits。

**Q: 为什么还保留旧的 `geo_avs_qfe_autovoc_uavscenes.py`？**  
A: 这是已经在 UAVScenes 100 帧上验证过的高性能主实验路径。新的 `01-07` 脚本把论文需要的模块显式化，旧脚本负责稳定复现主指标。

**Q: 这是不是完全无监督？**  
A: 训练上是免训练/无监督；评估时使用 UAVScenes 标签计算客观指标。词表来自 VLM 和 foundation evidence，不由人工给定固定类别表。

