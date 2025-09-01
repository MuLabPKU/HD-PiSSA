# HD-PiSSA
[EMNLP 2025] HD-PiSSA: High-Rank Distributed Orthogonal Adaptation 

paper link: https://arxiv.org/pdf/2505.18777

## Abstract

Existing parameter-efficient fine-tuning (PEFT) methods for large language models (LLMs), such as LoRA and PiSSA, constrain model updates to low-rank subspaces, limiting their expressiveness and leading to suboptimal performance on complex tasks. To address this, we introduce **H**igh-rank **D**istributed **PiSSA (HD-PiSSA)**, a distributed PEFT approach that initializes orthogonal adapters across different devices and aggregates their delta updates collectively on W for fine-tuning. Unlike Data Parallel LoRA or PiSSA, which maintain identical adapters across all devices, HD-PiSSA assigns different principal components of the pre-trained weights to each GPU, significantly expanding the range of update directions. This results in over **16x higher effective updated ranks** than data-parallel LoRA or PiSSA when fine-tuning on 8 GPUs with the same per-device adapter rank. Empirically, we evaluate HD-PiSSA across various challenging downstream tasks, including mathematics, code generation, and multi-task learning. In the multi-task setting, HD-PiSSA achieves average gains of **10.0** absolute points **(14.63%)** over LoRA and **4.98** points **(6.60%)** over PiSSA across 12 benchmarks, demonstrating its benefits from the extra optimization flexibility.

## Getting Started

1. Clone HD-PiSSA:
```bash
git clone [https://github.com/zfw1226/D2A](https://github.com/MuLabPKU/HD-PiSSA.git)
cd HD-PiSSA
```
2. Install HD-PiSSA Environment
```bash
conda create -n hdpissa python==3.11
conda activate hdpissa
pip install -r requirements.txt  --no-deps
```

## Usage
1. Set the configuration in ```run.sh```
2. Set your desired prompt template in ```hd_pissa.py```:
```
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)
```
3. Run the experiment.
```bash
bash run.sh
```

## Citation
```bibtex
@article{wang2025hd,
  title={HD-PiSSA: High-Rank Distributed Orthogonal Adaptation},
  author={Wang, Yiding and Meng, Fauxu and Zhang, Xuefeng and Jiang, Fan and Tang, Pingzhi and Zhang, Muhan},
  journal={arXiv preprint arXiv:2505.18777},
  year={2025}
}
```
