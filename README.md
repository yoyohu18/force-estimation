# ForceEstimation

用 Go2 的本体感知信号（关节位置/速度、机体线速度/角速度、机体系重力方向，共 33 维）
估计四个足端的归一化垂直力 `foot_sensors`（4 维）。数据是 12 段 `.h5` 录制，
统一用 50 Hz 的 `interp2` 组。

仓库保持扁平结构，所有 `.py` 都在根目录、都能直接 `python xxx.py` 跑。
本文件只做**分类说明**：每个文件属于哪一类、依赖谁、什么时候跑。

环境：conda env `go2_isaac_ros2`（唯一装了 h5py 的那个）。

---

## 依赖关系

```
features.py ──┐
              ├──> dataset.py ──┬──> train.py ──> baselines.py
modes.py ─────┘                 │       │
                                │    model.py
                                └──> evaluate.py
```

- 无人 import 的独立脚本：`inspect_h5.py`、`gait_analysis.py`、
  `visualize_all.py`、`visualize_features.py`、`examine_data.py`
- 已被取代、无人 import：`dataloader.py`、`dataset_step5.py`

---

## 一、核心库（被 import，不是入口）

这四个文件定义"数据长什么样、模型长什么样"，本身不产出结果。

| 文件 | 职责 | 关键约定 |
|---|---|---|
| [features.py](features.py) | 从一个 h5 组构建 33 维输入 + 标准化统计量 | 0:12 关节位置，12:24 关节速度，24:27 机体线速度，27:30 角速度，30:33 机体系单位重力。**只标准化 0:30**，30:33 是单位向量不能逐分量除 std。`__main__` 跑不变性自检 |
| [modes.py](modes.py) | 逐帧运动模式标注 | `QUAD_WALK=0` / `QUAD_STAND=1` / `REDUCED=2`。由 `foot_mask` 推出，**只用于划分数据，绝不能当输入特征**。`foot_mask==0` 是支撑相，`==1` 是摆动相 |
| [dataset.py](dataset.py) | 滑窗 Dataset（当前在用的版本） | 一个样本 = `history` 帧连续窗口 → `(history,33)` / `(4,)`。窗口不允许跨文件边界、模式边界、时间空洞（`MAX_GAP_MS=25`）。划分：`VAL_FILES` 两个四足文件整体留出；reduced 数据按时间 72% 切 `TRAIN_SPLIT`/`VAL_SPLIT` |
| [model.py](model.py) | 两层 MLP baseline | 展平窗口 → Linear → ReLU → Linear → 4。**输出层不加 sigmoid**（会在 1.06 处饱和并杀掉冲击峰的梯度），改用 `negative_fraction` 监控负值 |

## 二、训练与评估（可执行入口）

按这个顺序跑。

| 文件 | 用途 | 命令 |
|---|---|---|
| [baselines.py](baselines.py) | 先跑这个。三个非学习参照点：预测训练均值 `MSE 0.01625`、k-NN `MSE 0.01053`、8-bit 量化下限 `MSE 0.0000021` | `python baselines.py` |
| [train.py](train.py) | 训练循环 + 验证 + early stop，存最佳权重和 stats 到 checkpoint | `python train.py --history 20 --hidden 64 --monitor quad` |
| [evaluate.py](evaluate.py) | 单个 MSE 掩盖的诊断：分足误差、支撑相 vs 摆动相、峰值跟踪、RMSE/MAE 比、四力之和、负输出比例 | `python evaluate.py --ckpt checkpoints/mlp.pt` |

checkpoint 里存 `{model, stats, epoch, val_mse}`——`stats` 必须跟着权重走，
evaluate 用它复现训练时的标准化。

## 三、数据检查与可视化（独立工具，不进管线）

理解数据用的，跑不跑都不影响训练。

| 文件 | 输出 | 命令 |
|---|---|---|
| [inspect_h5.py](inspect_h5.py) | 打印 h5 结构：每个 group / dataset 的 shape 和取值范围 | `python inspect_h5.py [path.h5]` |
| [gait_analysis.py](gait_analysis.py) | 从触地时序推断步态（trot / pace / bound / pronk / walk），控制台表格 + 步态图 | `python gait_analysis.py [-f xxx.h5]` |
| [visualize_all.py](visualize_all.py) | 把每个录制的每个通道画成 PNG，按 joints / body / feet 分图，支撑相灰底 | `python visualize_all.py [-g interp2] [--overview-only]` |
| [visualize_features.py](visualize_features.py) | 只画送进模型的那 33 维特征 | `python visualize_features.py [-f xxx.h5 -s 40 -d 15] [--all]` |

五个 h5 组的采样率：`lowstate` 500 Hz、`highstate` ~300 Hz、
`interp`/`interp2` 50 Hz、`lidar` ~10 Hz。模型只用 `interp2`。

## 四、历史遗留（已被取代，保留作参考）

**不要在新代码里 import 这三个。**

| 文件 | 被谁取代 | 说明 |
|---|---|---|
| [dataloader.py](dataloader.py) | `dataset.py` + `features.py` | 最早的一体化 loader，特征构建和 Dataset 揉在一起，没有模式边界处理 |
| [dataset_step5.py](dataset_step5.py) | `dataset.py` | Step 5 的单帧 Dataset（一帧进一帧出），Step 6 换成滑窗后作废 |
| [examine_data.py](examine_data.py) | `visualize_all.py` | 最早的 foot_mask 探索脚本，只画一种图 |

---

## 目录

| 路径 | 内容 | 入库 |
|---|---|---|
| `dataset/` | 12 个 `.h5` 录制（296 MB） | 否，已在 `.gitignore` |
| `dataset/figures/` | 所有可视化脚本的 PNG 输出目录 | 否，随 `dataset/` 一起忽略 |
| `checkpoints/` | 训练权重 `mlp.pt` / `mlp_quad.pt` | 否，已在 `.gitignore`（可由 `train.py` 重新生成） |

所有脚本都用 `os.path.dirname(os.path.abspath(__file__))` 定位 `dataset/`，
所以文件必须留在仓库根目录——移动任何 `.py` 都要同步改路径解析。
