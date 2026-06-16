# Geo-AVS 满血版补齐状态

本文件对照 `Geo_AVS_full_build_plan_review_20260616.txt` 记录当前仓库补齐情况。

| 指导意见缺口 | 当前实现 | 关键文件 |
|---|---|---|
| VLM/captioner 自动词表 | 已补齐，支持 Qwen2.5-VL，缺权重时提供 smoke-test fallback | `geo_avs/autovoc/vlm_captioner.py`, `scripts/01_generate_vlm_autovoc.py` |
| Caption2Tag + 遥感同义词归一化 | 已补齐，包含 canonical terms、synonym mapper、停用词过滤 | `geo_avs/autovoc/caption2tag.py`, `geo_avs/autovoc/remote_sensing_lexicon.py` |
| SegEarth/SAM3 logits + presence cache | 已补齐 cache schema 与 adapter；主实验仍复用已验证 SAM3 fast path | `geo_avs/evidence/*`, `scripts/02_extract_segearth_evidence.py` |
| superpoint partition adapter | 已补齐 voxel 可运行实现，GrowSP/SPT/EZ-SP adapter 保留外部分区接口 | `geo_avs/superpoints/*`, `scripts/03_build_superpoints.py` |
| SPFE/QFE 模块化 | 已补齐正式模块与 cache lifting 脚本 | `geo_avs/lifting/*`, `scripts/04_lift_evidence_qfe.py` |
| AutoVoc verification | 已补齐 CaptionFreq + Presence + QFE_TopP + AreaCoverage 联合评分 | `geo_avs/autovoc/vocabulary_scoring.py`, `scripts/05_verify_autovoc.py` |
| 3D segmentation 输出 | 已补齐 verified vocab 到 superpoint/point label 的脚本 | `geo_avs/segmentation/*`, `scripts/06_run_geo_avs_segmentation.py` |
| 开放词表评估 | 已补齐 vocab PR/F1、full-candidate gap、TPSS lexical proxy | `geo_avs/evaluation/*`, `scripts/07_evaluate_open_vocab.py` |
| 一键完整 UAVScenes 实验 | 已补齐，接入 VLM caption-json 并保留已验证主实验路径 | `scripts/run_full_geo_avs_uavscenes.sh` |
| 复现信息 | 已补齐 requirements、LICENSE、中文使用指南 | `requirements.txt`, `LICENSE`, `docs/USAGE_CN.md` |

当前仍需注意：

1. 若要在论文中声明 “VLM AutoVoc”，主实验必须使用真实 Qwen2.5-VL 或同级 VLM 权重，而不是 fallback captioner。
2. 若要声明 “SPT/EZ-SP partition”，需要实际导入 upstream partition 输出；当前主结果使用的是可控的 voxel superpoint token。
3. SensatUrban、DALES、S3DIS、ScanNet、SemanticKITTI 只有在具备图像-点云配对和投影关系时，才能作为完整多模态 Geo-AVS 数据集；否则只能作为退化/补充实验。

