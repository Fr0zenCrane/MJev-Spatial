<div align="center">

# ![MJev-Spatial: System One Spatial Intelligence via Native Attention Matching](assets/mjev-spatial-title.png)

**通过有限候选决策实现 System 1 多模态空间智能**

[![许可证](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![模型](https://img.shields.io/badge/%F0%9F%A4%97%20Model-MJev--Spatial--Qwen--3.5--4B-yellow)](https://huggingface.co/Fr0zencr4nE/MJev-Spatial-Qwen-3.5-4B)
[![arXiv](https://img.shields.io/badge/arXiv-2610.22276-b31b1b.svg)](https://arxiv.org/abs/2610.22276)

[English](README.md) · 简体中文

</div>

MJev-Spatial 是一个用于空间理解的多模态决策模型。它把空间问题转化为有限候选项上的选择，因此模型可以直接做出判断，不需要先生成答案字符串再解析。

模型基于 Qwen3.5-4B，复用主干模型预训练的 query 和 key 投影来为候选项打分。发布版本不额外添加独立的预测头。每次决策只需要一次前向计算，也不会生成答案 token。

当前模型 checkpoint 位于 [Fr0zencr4nE/MJev-Spatial-Qwen-3.5-4B](https://huggingface.co/Fr0zencr4nE/MJev-Spatial-Qwen-3.5-4B)。对应论文是 [MJev-Spatial: System One Spatial Intelligence via Native Attention Matching](https://arxiv.org/abs/2610.22276)。

![MJev-Spatial 总览](assets/figure-1-teaser.png)

## 模型能做什么

MJev-Spatial 用同一种候选打分接口处理三类输出：

- **选择题：** 从请求提供的候选项中选择一个。
- **数值预测：** 先选择粗粒度数值区间，再选择更细的区间，并将结果解码为数值。
- **Pointing：** 连续三轮从 3×3 网格中选择区域，在轮次之间重新编码所选 crop，最后将网格单元中心解码为归一化图像坐标。

候选集合是任务接口的一部分。模型只对当前阶段合法的候选项打分，屏蔽补齐位置，并返回预测结果、每轮分数和决策路径。

![任务形式](assets/figure-2-task-inference.png)

## 模型读出

对于决策位置和候选位置，模型读取语言主干产生的 hidden state，并使用最后一个 full-attention block 的原生投影：

\[
z_i = \frac{1}{H}\sum_{h=1}^{H}
\frac{Q_h(h_{\mathrm{decision}})^\top K_h(h_{\mathrm{candidate}})}{\sqrt{d_h}}.
\]

发布配置使用最后一个 full-attention 层、原生 Q/K normalization、grouped-query key 对齐和多模态 rotary position encoding。score logits 在当前候选集合上归一化。

![原生 Q/K 读出](assets/figure-3-architecture.png)

## 论文报告结果

在论文所述评测协议下，发布模型报告：

- 十二个空间 benchmark 等权平均 **65.64%**；
- 在评测模型中有 **7/12** 个 benchmark 取得最佳结果；
- 十二项延迟协议的平均单请求延迟为 **160.6 ms**；
- 相比最快的自回归 baseline 和未修改的 Qwen3.5-4B，延迟分别约低 **2.5×** 和 **5.6×**。

这些数字对应论文中的固定评测设置。模型换到新的数据集或部署环境后，应使用自己的数据重新评测。

## 安装

公开代码支持 Python 3.10 或更高版本，并需要可用的 CUDA PyTorch 环境。

```bash
git clone https://github.com/Fr0zenCrane/MJev-Spatial.git
cd MJev-Spatial
python -m venv .venv
source .venv/bin/activate
pip install -e '.[inference,train]'
```

如果环境中尚未安装 PyTorch，请先根据 CUDA 驱动安装匹配的 PyTorch 版本。

## Checkpoint 目录

推理和训练命令都接收本地 checkpoint 目录。发布 checkpoint 包含模型配置、processor 文件和语言模型权重，目录结构与 `scripts/predict.py` 和 `scripts/train.py` 一致。

```bash
export MJEV_CHECKPOINT=/path/to/mjev-spatial-checkpoint
```

模型权重发布在 [Hugging Face](https://huggingface.co/Fr0zencr4nE/MJev-Spatial-Qwen-3.5-4B)，本仓库包含源代码和示例。

## 运行推理

推理输入是 JSON 对象，包含图像、问题和候选空间。`examples/` 中提供了少量合成请求，可以在没有私有数据集的情况下跑通流程。

```bash
python scripts/predict.py \
  --checkpoint "$MJEV_CHECKPOINT" \
  --input examples/requests.jsonl \
  --output predictions.jsonl
```

输出包含解码后的答案、决策路径以及每一轮候选项的分数。

### 请求示例

选择题：

```json
{
  "id": "choice-0001",
  "input": {
    "media": [{"kind": "image", "uri": "examples/images/scene.png"}],
    "question": "红色方块相对于蓝色圆形在哪里？",
    "answer_space": {
      "kind": "choice",
      "options": [
        {"id": "left", "text": "left"},
        {"id": "right", "text": "right"}
      ]
    }
  }
}
```

数值题：

```json
{
  "id": "scalar-0001",
  "input": {
    "media": [{"kind": "image", "uri": "examples/images/scene.png"}],
    "question": "两个物体之间的大致距离是多少？",
    "answer_space": {"kind": "scalar", "quantity": "distance", "unit": "m"}
  }
}
```

定位题：

```json
{
  "id": "point-0001",
  "input": {
    "media": [{"kind": "image", "uri": "examples/images/scene.png"}],
    "question": "请指出蓝色圆形的位置。",
    "answer_space": {
      "kind": "point",
      "coordinate_system": "normalized_xy",
      "num_points": 1
    }
  }
}
```

## 使用自己的数据继续训练

公开训练接口读取同一种带类型的标注记录。它不要求使用特定 benchmark 或数据源。准备自己的图像和标注，写成 JSONL，并在同一条记录中提供任务空间和 target。

```json
{
  "id": "train-choice-0001",
  "input": {
    "media": [{"kind": "image", "uri": "my_data/images/example.png"}],
    "question": "箱子在椅子左边吗？",
    "answer_space": {
      "kind": "choice",
      "options": [
        {"id": "yes", "text": "yes"},
        {"id": "no", "text": "no"}
      ]
    }
  },
  "target": {"option_id": "yes"}
}
```

数值记录使用 `target.value` 和 `answer_space` 中声明的单位；定位记录使用 `target.points` 中的归一化 `[x, y]`：

```json
{"target": {"value": 2.4, "unit": "m"}}
```

```json
{"target": {"points": [[0.62, 0.41]]}}
```

用示例数据运行一个小规模监督更新：

```bash
python scripts/train.py \
  --checkpoint "$MJEV_CHECKPOINT" \
  --data examples/train.jsonl \
  --config configs/finetune.json \
  --output runs/example-finetune
```

训练命令会保存可重新加载的 checkpoint。之后可以将它交给 `scripts/predict.py` 和 `scripts/evaluate.py`。

默认训练配置冻结视觉编码器和视觉语言 projector，更新用于决策读出的语言侧参数。学习率、batch size、梯度累积、精度和 checkpoint 保存频率均可配置。

## 评测 checkpoint

评测使用相同的公开记录格式，并从评测 JSONL 读取 target：

```bash
python scripts/evaluate.py \
  --checkpoint runs/example-finetune \
  --data examples/eval.jsonl \
  --output runs/example-finetune/evaluation.json
```

基础评测器会根据记录中的任务类型报告选择题准确率、数值误差和定位距离/命中指标。

## 代码范围

本仓库包含模型运行时、基础训练循环、公开评测器、示例和测试。预期使用方式是加载发布模型运行自己的输入，或使用自己的标注记录继续训练。

## 许可证

源代码采用 Apache License 2.0。第三方归属信息见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。
