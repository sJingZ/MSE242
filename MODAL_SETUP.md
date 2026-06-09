# Modal 配置与使用说明

这份说明告诉你如何配置 Modal，并用它在云端 GPU 上跑 `src/modal_app.py`（当前包装的是 `src/cnn_lstm.py`）。
本地代码本身不需要改动，Modal 会把数据集和模型源码打包进容器、在远程 GPU 上训练，然后把结果拉回到本地 `results/` 目录。

---

## 0. 前提条件

- 一个 Modal 账号：去 https://modal.com 注册（用 GitHub / Google 登录即可，免费额度足够做 smoke test）。
- Python 3.10+（建议 3.11，和容器内版本一致）。
- 拿到完整的项目代码 **以及数据集**（见第 2 步，很重要）。

---

## 1. 安装依赖并登录 Modal

在项目根目录（`MS&E242_final/`）下执行：

```bash
# 建议用虚拟环境
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 安装依赖（包含 modal 客户端）
pip install -r requirements.txt

# 一次性认证（会自动打开浏览器，登录后授权即可）
modal setup
```

`modal setup` 会在本机写入 token（默认在 `~/.modal.toml`），之后就不用再登录了。
如果浏览器没自动弹出，可以改用：

```bash
modal token new
```

---

## 2. 准备数据集

Modal 在构建镜像时会把本地的数据目录整个打包上传：

```
data/processed/of/
```

代码里写死了这个路径（`modal_app.py` 中的 `DATA_DIR`），如果目录不存在会直接报错：

```
Dataset dir not found: .../data/processed/of
Run the OF pipeline notebook first to produce data/processed/of/.
```

**注意：** 仓库里通常只带了 `data/processed/of/_config.json`，真正的行情数据是 12 个 `.npz` 文件（在 `_config.json` 里列出，比如 `Lakers_vs_Rockets_Lakers.npz` 等）。你需要确保这些 `.npz` 文件都在本地这个目录下。两种获取方式二选一：

1. 直接找我（Christy）要打包好的 `data/processed/of/` 文件夹，解压到对应位置；或
2. 自己跑 `notebooks/colab_of_pipeline.ipynb` 这个 pipeline 生成。

确认一下文件齐了：

```bash
ls data/processed/of/
# 应该能看到 _config.json + 一堆 .npz 文件
```

---

## 3. 跑起来

所有命令都在项目根目录执行。

```bash
# (a) 先看有哪些模型可选
modal run src/modal_app.py --list-models

# (b) 快速 smoke test（T4 GPU，几分钟跑完，验证环境通不通）
modal run src/modal_app.py --model cnn_lstm --quick

# (c) 完整训练，自己选 GPU 和超参
modal run src/modal_app.py --model cnn_lstm --gpu A10G \
    --max-epochs 50 --batch-size 256 --lr 1e-3 --markets all

# (d) 想把训练好的权重 (.pt) 也拉回本地，加 --save-model
modal run src/modal_app.py --model cnn_lstm --quick --save-model
```

GPU 可选：`T4 | L4 | A10G | A100 | H100`（默认 `T4`，越往后越贵越快）。

---

## 4. 结果在哪

跑完后，结果会自动写回本地（和本地直接跑的格式完全一致）：

```
results/experiments.jsonl              # 每跑一次追加一行汇总
results/runs/<model>_<timestamp>.json  # 单次完整记录
results/runs/<model>_<timestamp>.pt    # 仅当加了 --save-model
```

容器是临时的，远程不会留任何文件——所有持久化都发生在你本地。

---

## 5. 常用参数速查

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `--model` | 跑哪个模型 | `cnn_lstm` |
| `--gpu` | GPU 类型 | `T4` |
| `--quick` | 极简快速 smoke test | 关 |
| `--save-model` | 把 `.pt` 权重拉回本地 | 关 |
| `--markets` | 用哪些市场 | `all` |
| `--max-epochs` | 最大训练轮数 | `50` |
| `--batch-size` | batch 大小 | `256` |
| `--lr` | 学习率 | `1e-3` |
| `--seed` | 随机种子 | `42` |
| `--tag` | 这次运行的标签 | `modal` |

完整参数见 `src/modal_app.py` 里 `main()` 的函数签名。

---

## 6. 常见问题

- **`Dataset dir not found`** → 见第 2 步，`.npz` 数据没放到 `data/processed/of/`。
- **`Unknown model 'xxx'`** → 用 `--list-models` 看可选名字，目前只有 `cnn_lstm`。
- **认证报错 / 提示未登录** → 重新跑 `modal setup` 或 `modal token new`。
- **本地没装 torch/numpy 也能跑吗？** → 可以。训练在远程容器里做（镜像里已装好 torch/numpy/scikit-learn）；本地只用 `modal` 客户端和标准库把结果写回，所以本地环境很轻。
- **想加新模型？** → 在 `modal_app.py` 的 `MODEL_REGISTRY` 里加一项，指向新的源码文件即可，会自动随镜像上传。
