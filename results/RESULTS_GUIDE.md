# 结果解读指南 (Results Interpretation Guide)

本文档解释 `results/experiments.jsonl` 中每条 CNN-LSTM 实验记录的含义:
**训练曲线应该怎么变、每个 metric 是什么、以及"结果好"时应该期待的数值范围。**

所有定义都与 `src/cnn_lstm.py` 中的实现严格一致(`compute_metrics`、`train_loop`)。

---

## 0. 实验在做什么(背景)

- **任务**:用过去 100 个 tick 的订单流(order flow, OF, 20 维 = 10 档 bid/ask)预测 Polymarket / NBA 市场未来 **mid-price 收益**。
- **多 horizon**:同时预测 `h = 1, 2, 3, 5, 10` 步之后的收益,模型输出 5 个数。
- **训练目标**:对收益做**标准化**(用训练集的均值 `target_mu`、标准差 `target_sigma`),再用 **MSE** 损失回归。
- **三个评估块**:
  - `out_of_sample` —— **测试集**(最重要,真正的泛化表现)
  - `in_sample` —— 训练集(看模型有没有学到东西/有没有过拟合)
  - `linear_benchmark_oos` —— Ridge 线性回归在测试集上的表现(基线对照)

---

## 1. 训练 / 验证损失随 epoch 如何变化

`training.history` 里有逐 epoch 的 `train_loss` 和 `val_loss`,它们是**标准化空间里的 MSE**。

### 关键基准:损失 = 1.0 意味着什么

因为目标用训练集统计量标准化,所以:

- **"只预测均值(输出 0)"的朴素模型,训练损失 ≈ 1.0**(标准化后训练集方差恰好是 1)。
- 因此 **`train_loss < 1.0` 才说明模型真的学到了信号**;越低越好(但金融数据里降幅通常很小)。
- 验证集的"朴素基线损失"等于**验证集收益在训练标准化尺度下的方差**。如果训练/验证分布一致,这个值应接近 1;如果远大于 1,说明存在**分布漂移 / 非平稳**。

### 好的训练曲线长什么样

| 阶段 | `train_loss` | `val_loss` | 解读 |
|---|---|---|---|
| Epoch 1 | ≈ 1.0 | ≈ 验证基线(理想 ≈ 1) | 起点,基本等于预测均值 |
| 训练中 | 平滑**单调下降**到 < 1.0 | 先下降后趋平 | 模型在学习 |
| 收敛 | 缓慢下降并趋于平稳 | 触底后不再改善 → **早停触发** | `patience=5` 内无改善即停 |

理想特征:
- **两条曲线都下降**,且 `val_loss` 与 `train_loss` 的**差距小且稳定**(良好泛化)。
- `val_loss` 触底后回升 = 开始过拟合,早停应在此前后停下(记录 `best_val_loss`)。

### 不好的几种曲线(诊断)

- **`train_loss` 卡在 ≈ 1.0 不动** → 没学到东西(欠拟合 / 学习率/数据问题)。当前 quick 跑就是这种(0.99)。
- **`train_loss` 很低但 `val_loss` 远高且持续上升** → 过拟合。
- **`val_loss` ≫ `train_loss`(如当前的 10 vs 1)** → 训练/验证分布漂移,这是**数据层面**的问题,需要排查标准化口径、市场量纲、时间切分,而不是单纯调模型。
- **出现 NaN/inf** → `run_epoch` 会中止并(在 GPU 上)自动回退 CPU 重试。

---

## 2. 每个 Metric 的定义与解读

每个 metric 都是**按 horizon**(`"1","2","3","5","10"`)给一个值,外加一个跨 horizon 的均值(`*_mean`)。

### 2.1 `r2_oos` —— 样本外 R²(核心指标)

```
r2_oos = 1 - MSE_model / MSE_naive
```

- `MSE_naive` = "恒预测训练均值"的误差。
- **`> 0`**:模型比朴素基线更准(**有预测力**)。
- **`= 0`**:与朴素基线无异。
- **`< 0`**:比"什么都不预测"还差(常见于过拟合,如本项目的线性基线)。
- ⚠️ 金融高频收益的 R² 天然极小,**0.001 ~ 0.05 就已经是有意义的信号**,不要期待 0.5 这种数值。

### 2.2 `directional_accuracy` —— 方向准确率

- 预测收益符号与真实符号一致的比例。
- **只在真实收益 ≠ 0 的 tick 上计算**(大量 tick 恰好不动,计入会变成"测模型多常输出 0")。
- 基线 = **0.5**(随机)。**> 0.5 即方向上有信息**;0.55+ 已相当不错。

### 2.3 `directional_coverage` —— 有变动的占比

- 测试集中真实收益 ≠ 0 的 tick 比例(即方向准确率的样本覆盖面)。
- **这不是性能指标**,是上下文信息:horizon 越长,价格越可能动,coverage 越高(数据里从 ~0.19 升到 ~0.38 是正常的)。

### 2.4 `sharpe` —— 年化夏普比率(交易价值指标)

- 策略:按预测方向做多/做空 1 单位,`pnl = sign(pred) · true_return`。
- `sharpe = mean(pnl) / std(pnl) × ann_factor`,`ann_factor = sqrt(252) ≈ 15.87`。
- 衡量**方向策略的风险调整后收益**:**> 0 有正期望**;年化 **> 1 很好**;**< 0 是亏损方向**。

### 2.5 `mean_pnl` —— 每笔平均盈亏

- 上述策略每个 tick 的平均收益(原始收益单位)。> 0 代表平均盈利。规模很小是正常的。

### 2.6 `mse_model` / `mse_naive` —— 原始 MSE

- 分别是模型与朴素基线在**原始收益单位**下的均方误差。
- 直接对比:`mse_model < mse_naive` ⇔ `r2_oos > 0`。
- 若两者**几乎逐位相等**(如当前 quick 跑),说明模型实际上退化成了朴素均值预测。

### 2.7 `*_mean` —— 跨 horizon 平均

- `r2_oos_mean`、`directional_accuracy_mean`、`sharpe_mean` 是 5 个 horizon 上(忽略 NaN)的算术平均,用于一眼看总体水平。

---

## 3. "结果好"时应该期待的数值

> 重要前提:这是**微观市场收益预测**,信噪比极低。"好"= 稳定地、显著地略胜基线,而不是高 R²。

### 3.1 训练层面

| 量 | 当前 quick(差) | 可接受 | 好 |
|---|---|---|---|
| 最终 `train_loss` | ≈ 0.99(没动) | < 0.97 且仍在降 | < 0.95,平滑下降 |
| `val_loss` | ≈ 10(分布漂移) | 接近验证基线、稳定 | 下降后平稳,与 train 差距小 |
| `epochs_run` | 2(冒烟) | 跑到早停(十几~几十) | 早停在 val 触底处 |
| train/val 损失差 | ~10× | < 2× | 接近 1× |

### 3.2 测试集指标(out_of_sample,最关键)

| Metric | 当前 quick(差) | 可接受 | 好 |
|---|---|---|---|
| `r2_oos_mean` | ≈ -0.0007(≈0) | 0.001 ~ 0.005 | **> 0.01**,且大多数 horizon 为正 |
| `directional_accuracy_mean` | ≈ 0.54 | 0.52 ~ 0.55 | **> 0.55** |
| `sharpe_mean` | ≈ -0.02(≈0) | 0.3 ~ 0.8 | **> 1.0** |
| `mse_model` vs `mse_naive` | 几乎相等 | model 略低 | model 在所有 horizon 明显更低 |

### 3.3 与基线对照(必须满足)

- **跑赢 naive**:`r2_oos > 0`(测试集多数 horizon 为正)。
- **跑赢线性基线**:CNN-LSTM 的 `r2_oos_mean` 和 `sharpe_mean` 应优于 `linear_benchmark_oos`。
  - 注意当前线性基线 `r2_oos_mean ≈ -0.74`(严重过拟合),所以"比它好"门槛很低;真正的目标是 CNN-LSTM 自己的 `r2_oos` 转正。
- **in-sample vs out-of-sample 一致**:两者方向一致(都为正)说明信号真实;若 in-sample 很好而 out-of-sample 崩掉,则是过拟合 / 非平稳。

### 3.4 一句话判断标准

> **好的结果** = `val_loss` 平滑收敛且与 `train_loss` 接近;测试集 `r2_oos_mean > 0`、`directional_accuracy_mean > 0.55`、`sharpe_mean > 0.5`;并且在大多数 horizon 上稳定优于 naive 与线性基线。

---

## 4. 复现完整(非 quick)运行

当前 `experiments.jsonl` 里的两条是 `--quick`(2 epoch)冒烟测试,**不能作为最终结论**。完整评估建议:

```bash
python src/cnn_lstm.py \
  --max-epochs 50 --patience 5 \
  --batch-size 256 --lr 1e-3 \
  --train-stride 25 --eval-stride 10 \
  --tag full-run
```

跑完后回到本指南第 3 节核对训练曲线与各 metric。若 `val_loss` 仍比 `train_loss` 大一个数量级,先排查数据的分布漂移(标准化口径 / 市场量纲 / 时间切分),再谈调参。
