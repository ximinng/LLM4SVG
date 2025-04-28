# 🚀 Training Examples & Model Cards

Try LLM4SVG with different foundation models based on `LLaMA-Factory`:

- foundation model: `Qwen2.5-VL-7B`

```shell
# w/o SVG Tokens
llamafactory-cli train examples/train_lora/svgx_qwen2vl_lora_sft.yaml

# with SVG Tokens
llamafactory-cli train examples/train_lora/svgx_qwen2vl_lora_sft_enc.yaml

# Multi-Node Multi-GPUs, 4 different devices (nodes)
FORCE_TORCHRUN=1 NNODES=4 NODE_RANK=1 MASTER_ADDR=your_master_addr MASTER_PORT=29500 llamafactory-cli train examples/train_lora/svgx_qwen2vl_lora_sft_enc.yaml
```

- foundation model: `DeepSeek-R1-Distill-Qwen-7B`

```shell
llamafactory-cli train examples/train_full/svgx_deepseekr1_qwen_lora_sft_enc.yaml
```

- foundation model: `Gemma3`

```shell
llamafactory-cli train examples/train_lora/svgx_gemma3_lora_sft_enc.yaml
```

- foundation model: `Llama-3.2-11B-Vision-Instruct`

```shell
llamafactory-cli train examples/train_full/svgx_llama3_vision_lora_sft.yaml
```

- foundation model: `Falcon-7B`

```shell
llamafactory-cli train examples/train_lora/svgx_falcon_lora_sft_enc.yaml
```

Try LLM4SVG with different foundation models based on [`unsloth`](https://github.com/unslothai/unsloth):

- foundation model: `Llama-3.2-4bit`

```shell
# origin:
CUDA_VISIBLE_DEVICES=1 python main.py x=llama3-sft-unsloth \
  data.load_from_disk='path/to/dataset' \
  data.syntactic_encode=false \
  x.train_batch_size=64 \
  project_dir="workspace/train-via-unsloth/llama3-4bit"
# llm4svg:
CUDA_VISIBLE_DEVICES=1 python main.py x=llama3-sft-unsloth \
  data.load_from_disk='path/to/dataset' \
  data.syntactic_encode=true \
  x.train_batch_size=32 \
  project_dir="workspace/llm4svg-via-unsloth/llama3-4bit"
```

Try LLM4SVG with different foundation models based on `transformers`:

- foundation model: `GPT2-XL`

```shell
# default:
accelerate launch main.py x=gpt2-sft data.load_from_disk='/path/to/dataset'

# prompt=name+blip_caption
CUDA_VISIBLE_DEVICES=0,1 HF_DATASETS_OFFLINE=1 accelerate launch --num_processes 4 \ 
  --config_file configs/accelerate/ddp_config.yaml \ 
  main.py x=gpt2-sft \
  data.load_from_disk='path/to/dataset' data.text_prompt='[name,blip_caption]' \
  x.model_name='path/to/openai-community/gpt2-xl' \
  x.seq_len=2048 x.train_batch_size=2 \
  project_dir="workspace/llm4svg-gpt2xl-maxL2048-warmup200"

# prompt=name
CUDA_VISIBLE_DEVICES=0,1 HF_DATASETS_OFFLINE=1 accelerate launch --multi_gpu main.py x=gpt2-sft \
  data.load_from_disk='path/to/dataset' data.text_prompt='[name]' \
  x.model_name='path/to/openai-community/gpt2-xl' \
  x.seq_len=2048 x.train_batch_size=2 \
  project_dir="workspace/llm4svg-gpt2xl-maxL2048-warmup200"
```

Try LLM4SVG with different foundation models based on `trl`:

- foundation model: `Phi-2`

```shell
# prompt=name+blip_caption
CUDA_VISIBLE_DEVICES=4,5 accelerate launch --config_file configs/accelerate/fsdp_config.yaml main.py \
  x=phi2-sft-trl \
  data.load_from_disk='path/to/dataset' \
  data.text_prompt='[name,blip_caption]' \
  x.model_name='path/to/phi-2' \
  x.seq_len=1024 \
  x.train_batch_size=1 \
  x.use_bf16=true \
  project_dir="workspace/llm4svg-phi2-bf16-fsdp-maxL2048-warmup200"
```

## 🔧 Quick API Deployment

Get started with inference using our vLLM-powered API:

```shell
API_PORT=8000 CUDA_VISIBLE_DEVICES=0 llamafactory-cli api \
--model_name_or_path=Qwen/Qwen2-VL-7B-Instruct \
--adapter_name_or_path=saves/qwen2vl-7B-lora-train/checkpoint-7000 \
--max_length=4096 --max_new_tokens=4096 \
--template=qwen2_vl --trust_remote_code=True --infer_backend=vllm
```
