# Empowering LLMs to Understand and Generate Complex Vector Graphics

<div align="center" style="line-height: 1.2;">

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Framework LLaMA-Factory](https://img.shields.io/badge/LLaMA--Factory-Supported-brightgreen?style=for-the-badge)](https://github.com/hiyouga/LLaMA-Factory)
[![Framework Unsloth](https://img.shields.io/badge/Unsloth-Supported-ffba08?style=for-the-badge)](https://github.com/unslothai/unsloth)
[![Framework Transformers](https://img.shields.io/badge/Transformers-Supported-blueviolet?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/docs/transformers/index)
[![Framework TRL](https://img.shields.io/badge/TRL-Supported-orange?style=for-the-badge)](https://huggingface.co/docs/trl/index)
[![Inference vLLM](https://img.shields.io/badge/Inference-vLLM-success?style=for-the-badge)](https://github.com/vllm-project/vllm)

[![CVPR 2025](https://img.shields.io/badge/CVPR%202025-Paper-4169E1?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2412.11102)
[![arXiv](https://img.shields.io/badge/arXiv-2412.11102-8A2BE2?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2412.11102)
[![Project Website](https://img.shields.io/badge/Website-Project%20Page-4682B4?style=for-the-badge&logo=github&logoColor=white)](https://ximinng.github.io/LLM4SVGProject/)
[![Dataset SVGX-Core-250k](https://img.shields.io/badge/Dataset-SVGX--Core--250k-informational?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/xingxm/SVGX-Core-250k)
[![Dataset SVGX-SFT-1M](https://img.shields.io/badge/Dataset-SVGX--SFT--1M-informational?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/xingxm/SVGX-SFT-1M)
</div>

---

Official implementation for **"Empowering LLMs to Understand and Generate Complex Vector Graphics"**. This project
enables Large Language Models to process, understand, and generate complex Scalable Vector Graphics (SVG).

## Table of Contents

- [🎉 News](#-news)
- [✨ Highlights](#-highlights)
- [📊 SVGX-SFT Dataset](#-svgx-sft-dataset)
- [📦 Installation & Data Preparation](#-installation--data-preparation)
- [🚀 Training Examples](#-training-examples)
    - [Based on `LLaMA-Factory`](#based-on-llama-factory)
    - [Based on `unsloth`](#based-on-unsloth)
    - [Based on `transformers`](#based-on-transformers)
    - [Based on `trl`](#based-on-trl)
- [🔧 Inference using vLLM](#-inference-using-vllm)
- [🔑 Tips for Best Results](#-tips-for-best-results)
- [💘 Acknowledgements](#-acknowledgements)
- [📎 Citation](#-citation)
- [📄 License](#-license)
- [📬 Contact](#-contact)

## 🎉 News

- **[04/2025]** 🎉 Official release of LLM4SVG code,
  datasets ([SVGX-Core-250k](https://huggingface.co/datasets/xingxm/SVGX-Core-250k), [SVGX-SFT-1M](https://huggingface.co/datasets/xingxm/SVGX-SFT-1M)),
  and [Pretrained Model Weights]()! 🎉 *(Link for weights pending)*

## ✨ Highlights

- 🧠 **Multi-model Support**: Fine-tune a wide range of popular foundation models, including Llama 3.2, Qwen2.5-VL, Gemma
  3, DeepSeek, Falcon, Phi-2, GPT2-XL, and more.
- 📦 **Specialized SVGX Dataset**: Includes curated pretraining data (`SVGX-Core-250k`) and extensive supervised
  fine-tuning data (`SVGX-SFT-1M`).
- ⚡ **Accelerated Training & Inference**: Leverages efficient training frameworks like `LLaMA-Factory`, `unsloth`,
  `transformers`, and `trl`. Integrated with `vLLM` for high-throughput, low-latency inference.
- 🔍 **Multimodal Capabilities**: Fully supports text and vision inputs for comprehensive SVG understanding and
  generation tasks.
- ⚙️ **Flexible Training Options**: Supports various training techniques including LoRA and full fine-tuning, along with
  distributed training setups (Multi-GPU, Multi-Node).

## 📊 SVGX-SFT Dataset

Our SVGX-SFT Dataset is a comprehensive collection designed specifically for training LLMs to work effectively with
vector graphics.

- **Available Datasets on Hugging Face:**
    - [`xingxm/SVGX-Core-250k`](https://huggingface.co/datasets/xingxm/SVGX-Core-250k): Core pretraining data (250k
      examples).
    - [`xingxm/SVGX-SFT-1M`](https://huggingface.co/datasets/xingxm/SVGX-SFT-1M): Supervised fine-tuning data (1M
      examples).

- **Usage Example:**

```python
# Login using `huggingface-cli login` if the dataset requires authentication
from datasets import load_dataset

# Load SVGX-Core-250k
svgx_core_250k_dataset = load_dataset("xingxm/SVGX-Core-250k")

# Load SVGX-SFT-1M
svgx_sft_1m_dataset = load_dataset("xingxm/SVGX-SFT-1M")
```

## 📦 Installation & Data Preparation

```shell
# Step 1: Set up the environment (torch & unsloth & trl)
conda env create -f environment.yml && conda activate llm4svg

# Step 2: Download the datasets and place them in `dataset/SVGX-dataset`
bash script/download_dataset.sh

# Step 3: Set up the datasets
bash script/setup_dataset.sh

# Step 4: Install LLaMA-Factory
cd LLaMA-Factory && pip install -e ".[torch,metrics]"
```

### 🚀 Training Examples

We provide example configurations for fine-tuning various models using different frameworks.

---

#### Based on `LLaMA-Factory`:

- Model: `Qwen/Qwen2.5-VL-7B`

```shell
# Fine-tune with LoRA (using special SVG Tokens for encoding)
llamafactory-cli train examples/train_lora/svgx_qwen2vl_lora_sft_enc.yaml

# Example: Multi-Node Distributed Training (4 nodes)
# Set MASTER_ADDR, MASTER_PORT, NODE_RANK accordingly
FORCE_TORCHRUN=1 NNODES=4 NODE_RANK=<0,1,2,3> MASTER_ADDR=<your_master_node_ip> MASTER_PORT=29500 \
llamafactory-cli train examples/train_lora/svgx_qwen2vl_lora_sft_enc.yaml
```

- Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

```shell
llamafactory-cli train examples/train_lora/svgx_deepseekr1_qwen_lora_sft_enc.yaml
```

- Model: `Google/Gemma-3`

```shell
llamafactory-cli train examples/train_lora/svgx_gemma3_lora_sft_enc.yaml
```

- Model: `Falcon-7B`

```shell
llamafactory-cli train examples/train_lora/svgx_falcon_lora_sft_enc.yaml
```

---

#### Based on [`unsloth`](https://github.com/unslothai/unsloth):

- Model: `unsloth/llama-3.2-Instruct-4bit` (Example using 4-bit quantized Llama-3.2)

```shell
python main.py x=llama3-sft-unsloth project_dir="workspace/llm4svg-via-unsloth/llama3-4bit"
```

---

#### Based on `transformers` & `accelerate`:

- Model: `openai/GPT2-XL`

```shell
# Default training using accelerate
accelerate launch main.py x=gpt2-sft data.load_from_disk='/path/to/dataset'
# Example using multiple GPUs with Data Parallelism (DDP)
accelerate launch --config_file configs/accelerate/ddp_config.yaml main.py x=gpt2-sft x.seq_len=2048 x.train_batch_size=2 project_dir="workspace/llm4svg-gpt2xl-maxL2048"
# or
accelerate launch --multi_gpu main.py x=gpt2-sft data.text_prompt='[name]' x.seq_len=1024 x.train_batch_size=2 project_dir="workspace/llm4svg-gpt2xl-maxL1024"
```

---

#### Based on `trl` & `accelerate`:

- Model: `microsoft/Phi-2`

```shell
accelerate launch --config_file configs/accelerate/fsdp_config.yaml main.py x=phi2-sft-trl project_dir="workspace/llm4svg-phi2-fsdp-maxL2048"
```

## 🔧 Inference using vLLM

Get started with fast inference using our **vLLM-powered API** server:

```shell
API_PORT=8000 llamafactory-cli api \
--model_name_or_path=Qwen/Qwen2-VL-7B-SVGX-SFT-Encode-Model \
--max_length=4096 --max_new_tokens=4096 \
--template=qwen2_vl --trust_remote_code=True --infer_backend=vllm
```

Refer to the [vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) for more details on
interacting with the API endpoint.

## 🔑 Tips for Best Results

- **Distributed Training:** For datasets >50k examples, consider using multi-node setups (like DeepSpeed or FSDP via
  `accelerate` or `llamafactory-cli`) to significantly reduce training time.
- **Context Length:*** Set an appropriate `max_seq_length` (e.g., x.seq_len or via YAML config) for complex SVG
  generation. We recommend 2048 or higher.
- **Batch Optimization:** Adjust `per_device_train_batch_size` and `gradient_accumulation_steps` based on your available
  GPU memory to maximize throughput.
- **Inference Acceleration:** Utilize vLLM as your inference backend (`--infer_backend=vllm`) for optimized
  performance (up to 2x faster generation compared to standard Hugging Face pipelines).
- **Model Choice:** Experiment with different base models. Models with strong visual grounding (like Qwen-VL) or coding
  capabilities might show better performance on SVG tasks.
- **SVG Tokenization:** Using dedicated SVG tokens (enabled via `_enc` configs in examples) can potentially improve model
  understanding and generation of SVG syntax.

## 💘 Acknowledgements

This project is built upon the incredible work of:

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Unsloth](https://github.com/unslothai/unsloth)

Special thanks to the authors and contributors of both projects!

## 📎 Citation

If you find this project useful in your research, please consider citing our paper:

```
@article{xing2024llm4svg,
  title={Empowering LLMs to Understand and Generate Complex Vector Graphics},
  author={Xing, Ximing and Hu, Juncheng and Liang, Guotao and Zhang, Jing and Xu, Dong and Yu, Qian},
  journal={arXiv preprint arXiv:2412.11102},
  year={2024}
}
```

## 📄 License

This project is licensed under the **MIT License**.

## 📬 Contact

For questions, bug reports, or collaboration inquiries, please feel free to open an issue or submit a pull request on
GitHub.