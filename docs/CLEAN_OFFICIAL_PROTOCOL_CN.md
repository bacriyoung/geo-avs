# Geo-AVS Clean AutoVoc 官方协议说明

## 1. 任务定义

输入为 UAVScenes 的配对无人机 RGB 图像、LiDAR 点云和标定信息。方法在预测阶段不能读取人工类别表或点级真值，目标是由 VLM 自动发现当前场景实体词汇，再利用 2D 基础模型产生开放词汇证据，并通过超点投影生成 3D 点级语义预测。

官方 18 类名称只允许出现在评估映射器中。Rule、SBERT 和 LAVE-Qwen 映射在评估前冻结，其作用等价于把自由词预测翻译成 benchmark ID，不参与前向预测。

## 2. Clean 主流程

1. `Qwen2.5-VL-7B-Instruct` 对整图和 2x2 局部裁剪生成名词短语。
2. `CleanCaptionTagger` 去重和清理短语，不追加 RSlex、官方类别或人工提示词。
3. `SegEarth-OV3/SAM3` 针对自由短语生成 2D soft evidence。
4. 相机标定将 2D evidence 投影到 3D 点，并在约 1,200 个超点内聚合。
5. 比较 center、mean、q75、max、fixed-QFE、Rank-QFE 与 equal-rank 七种聚合。
6. 使用冻结的 Rule、SBERT、LAVE-Qwen 三种评估映射器计算官方 18 类指标。

## 3. 固定实验协议

- 数据源：UAVScenes interval5，共 24,126 帧、20 个场景。
- 当前正式开发协议：每个场景在 10%-90% 时间区间内均匀取 5 帧，共 100 帧。
- 抽样不读取 GT，协议文件保存在 `results/geo_avs_clean/protocol_scene_stratified_v1/`。
- 全量协议文件已冻结在 `results/geo_avs_clean/protocol_full_interval5/`。
- 单卡默认使用物理 GPU1；全流程支持 Caption2Tag 原子落盘和 evidence 断点续跑。

## 4. 当前结果

100 帧协议共评估 7x3=21 个 clean 组合，全部结果保存在 `table1_clean_main.csv`，没有筛除失败设置。

| 设置 | mIoU | mAcc | OA | Vocab-F1 | Top-3 Recall |
|---|---:|---:|---:|---:|---:|
| q75 + SBERT | **13.03** | 27.63 | 21.13 | 0.600 | 0.365 |
| mean + Rule | 12.99 | 25.38 | **29.93** | 0.483 | 0.244 |
| q75 + LAVE-Qwen | 12.61 | **28.66** | 18.12 | **0.673** | **0.409** |
| Rank-QFE + SBERT | 11.26 | 24.60 | 20.34 | 0.600 | 0.373 |
| equal-rank + LAVE-Qwen | 8.62 | 25.41 | 21.77 | 0.673 | 0.435 |

`q75 + SBERT` 是 clean 主协议的最佳 mIoU。Rank-QFE 没有超过 q75，因此不能把 Rank-QFE 涨点写成已验证贡献。

## 5. 受控消融与诊断

- 仅整图 Caption2Tag 的最佳 mIoU 为 11.25；整图加 2x2 裁剪达到 13.03，提升 1.78。
- 投影有效率为 98.40%，说明主要问题不是相机投影缺失。
- 平均归一化证据熵为 0.976，说明候选之间区分度很弱。
- 域内/官方提示诊断上界为 15.98 mIoU，仅比 clean 主结果高 2.95。
- Rank-QFE + SBERT 的 Top-3、Top-5 和词表 oracle 分别为 26.66、34.36、63.46 mIoU。

这些证据表明，当前主要瓶颈是“自由词到可靠视觉证据的校准与排序”，其次是缺少跨帧、多视一致性；继续单独修改几何门控不太可能带来决定性提升。

## 6. 可复现命令

100 帧协议：

```bash
cd /home/work/research/geo_avs
GPU=1 PROTOCOL=scene_stratified_v1 bash scripts/run_geo_avs_clean_official.sh
```

24,126 帧全量协议：

```bash
cd /home/work/research/geo_avs
GPU=1 PROTOCOL=full_interval5 bash scripts/run_geo_avs_clean_official.sh
```

全量运行预计需要约 50-60 GPU 小时和约 40-50 GB 新缓存空间。该入口是可恢复的，但 24,126 帧全量结果尚不能冒充本轮已完成的 100 帧实验。

## 7. 结论边界

当前代码已经实现了完整、无人工词表注入的自动开放词汇 UAV 点云分割闭环，并在真实多模态数据上通过 100 帧、20 场景协议验证。它证明了任务和系统可行性，但 13.03 mIoU 还不足以支持“顶会级性能已经完成”的结论。下一阶段最值得投入的是开放域 prompt-evidence 校准、跨帧词表记忆和多视 3D 一致性，而不是把诊断 oracle 或官方提示结果写成主方法。
