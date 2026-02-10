# TorchTitan Model Integration for lm-evaluation-harness

## Overview

The `torchtitan.py` module provides a custom model adapter that enables evaluation of TorchTitan-trained models using the lm-evaluation-harness framework. This integration is specifically designed for **Llama3 MoE (Mixture of Experts)** models trained with TorchTitan's native training infrastructure.

### Key Features

- **Native DCP Checkpoint Loading**: Loads models directly from TorchTitan's Distributed Checkpoint (DCP) format
- **No HuggingFace Dependency**: Does not inherit from HFLM; implements custom model initialization
- **MoE Support**: Specifically designed for Llama3 MoE architectures
- **Evaluation-Optimized**: Simplified checkpoint loading (model weights only, no optimizer/scheduler state)

## Architecture

### Class Hierarchy

```
TemplateLM (lm_eval.api.model)
    └── TorchTitanLM
```

Unlike the HuggingFace integration (`huggingface.py`), `TorchTitanLM` does **not** inherit from `HFLM` because:

1. **Different Model Initialization**: TorchTitan uses meta device initialization + manual weight loading
2. **Different Checkpoint Format**: DCP (Distributed Checkpoint Protocol) vs. safetensors
3. **No config.json**: TorchTitan uses Python-based configuration dictionaries

## Installation & Setup

### Required Dependencies

- PyTorch with distributed checkpoint support (`torch.distributed.checkpoint`)
- TorchTitan package (for model classes and configs)
- Transformers (for tokenizer only)

## Usage

### Basic Command

```bash
lm_eval --model torchtitan \
        --model_args model_name=llama3_moe,model_flavor=1B,checkpoint_path=/path/to/checkpoint,tokenizer_path=/path/to/tokenizer \
        --tasks hellaswag \
        --batch_size 1
```

### Model Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model_name` | str | `"llama3_moe"` | TorchTitan model architecture name |
| `model_flavor` | str | `"1B"` | Model size/configuration variant |
| `checkpoint_path` | str | `None` | Path to DCP checkpoint directory |
| `tokenizer_path` | str | `None` | Path to tokenizer (defaults to checkpoint_path) |
| `max_length` | int | `2048` | Maximum sequence length |
| `device` | str | `"cuda"` | Device to load model on |
| `batch_size` | int | `1` | Batch size for evaluation |
| `dtype` | str/torch.dtype | `torch.bfloat16` | Model precision |

### Available Model Flavors

From `torchtitan/models/llama3_moe/__init__.py`:

#### Debug Models (Small, for testing)
- `debugmodel_1exp`: 256 dim, 6 layers, 1 expert
- `debugmodel_2exp`: 256 dim, 6 layers, 2 experts
- `debugmodel_4exp`: 256 dim, 6 layers, 4 experts
- `debugmodel_8exp`: 256 dim, 6 layers, 8 experts
- `debugmodel_8exp_small`: 64 dim, 6 layers, 8 experts

#### Production Models
- `1B`: 2048 dim, 16 layers, 8 experts, top-k=2
- `3B`: 3072 dim, 28 layers, 8 experts, top-k=2
- `8B`: 4096 dim, 32 layers, 2 experts
- `8B_4exp`: 4096 dim, 32 layers, 4 experts
- `8B_8exp`: 4096 dim, 32 layers, 8 experts

#### Testing Variants
- `1B_2layer`: 2-layer version of 1B (for testing)
- `3B_2layer`: 2-layer version of 3B (for testing)
- `1B_2layer_halfmoe_vg`: Virtual Group MoE variant
- `3B_2layer_halfmoe_vg`: Virtual Group MoE variant

### Example Commands

#### Evaluate 1B Model on Multiple Tasks

```bash
lm_eval --model torchtitan \
        --model_args model_name=llama3_moe,model_flavor=1B,checkpoint_path=/checkpoints/llama3_moe_1B,tokenizer_path=/tokenizers/llama3 \
        --tasks hellaswag,arc_easy,arc_challenge \
        --batch_size 8 \
        --device cuda
```

#### Evaluate with Custom Max Length

```bash
lm_eval --model torchtitan \
        --model_args model_name=llama3_moe,model_flavor=3B,checkpoint_path=/checkpoints/llama3_moe_3B,max_length=4096 \
        --tasks mmlu \
        --batch_size 4
```

#### Debug Model Evaluation

```bash
lm_eval --model torchtitan \
        --model_args model_name=llama3_moe,model_flavor=debugmodel_8exp,checkpoint_path=/checkpoints/debug \
        --tasks lambada_openai \
        --batch_size 1
```

## Implementation Details

### Model Loading Process

The model loading follows TorchTitan's training initialization pattern:

```python
# 1. Import model class and configs
from torchtitan.models.llama3_moe import (
    Llama3MoE,
    llama3_moe_configs,
    Llama3MoEStateDictAdapter,
)

# 2. Get model configuration
model_args = llama3_moe_configs[model_flavor]

# 3. Initialize on meta device (no memory allocation)
with torch.device("meta"):
    model = Llama3MoE(model_args)

# 4. Move to target device and initialize weights
model.to_empty(device=device)
model.init_weights()

# 5. Load checkpoint using DCP
import torch.distributed.checkpoint as dcp
model_wrapper = ModelWrapper([model])
state_dict = model_wrapper.state_dict()
dcp.load(state_dict, checkpoint_id=checkpoint_path)
model_wrapper.load_state_dict(state_dict)

# 6. Set to eval mode
model.eval()
```

### Comparison with TorchTitan Training

| Aspect | TorchTitan Training | TorchTitanLM (Eval) |
|--------|---------------------|---------------------|
| **Checkpoint Loading** | Full state (model, optimizer, scheduler, dataloader) | Model weights only |
| **Parallelism** | Supports FSDP, TP, PP | Single device/GPU |
| **Optimizer State** | Loaded and restored | Not loaded |
| **Dataloader State** | Restored for resumption | Not applicable |
| **Gradient Computation** | Enabled | Disabled (`torch.set_grad_enabled(False)`) |
| **Mode** | Training mode | Eval mode (`.eval()`) |

### Comparison with HuggingFace Integration

| Feature | HuggingFace (`huggingface.py`) | TorchTitan (`torchtitan.py`) |
|---------|--------------------------------|------------------------------|
| **Base Class** | `HFLM` (inherits from `TemplateLM`) | `TemplateLM` (direct) |
| **Model Loading** | `AutoModelForCausalLM.from_pretrained()` | Manual: meta device + DCP load |
| **Checkpoint Format** | safetensors, PyTorch bins | DCP (Distributed Checkpoint) |
| **Configuration** | `config.json` from HuggingFace Hub | Python dict from `llama3_moe_configs` |
| **Tokenizer** | `AutoTokenizer.from_pretrained()` | `AutoTokenizer.from_pretrained()` (same) |
| **Multi-GPU** | Accelerate framework | Single device (no distributed) |
| **Quantization** | Supports GPTQ, GGUF, etc. | Not supported |
| **PEFT/LoRA** | Supported | Not supported |
| **Attention Mask** | Used in forward pass | Not used (TorchTitan models don't expect it) |

### Forward Pass Differences

**HuggingFace:**
```python
def _model_call(self, inps, attn_mask=None, labels=None):
    return self._model(inps, attention_mask=attn_mask).logits
```

**TorchTitan:**
```python
def _model_call(self, inps, attn_mask=None, labels=None):
    # TorchTitan models only accept input_ids
    return self._model(inps)
```

## Checkpoint Format

### TorchTitan DCP Structure

```
checkpoint_dir/
├── __0_0.distcp          # Rank 0 checkpoint
├── __1_0.distcp          # Rank 1 checkpoint (if distributed)
├── .metadata             # Checkpoint metadata
└── ...
```

### Loading from DCP

The implementation uses PyTorch's Distributed Checkpoint Protocol:

```python
import torch.distributed.checkpoint as dcp

# Create state dict container
state_dict = model_wrapper.state_dict()

# Load from DCP directory
dcp.load(state_dict, checkpoint_id=checkpoint_path)

# Apply to model
model_wrapper.load_state_dict(state_dict)
```

### State Dict Adapter (Not Used for Native Checkpoints)

The `Llama3MoEStateDictAdapter` is passed to `_load_checkpoint()` but **not used** when loading native TorchTitan checkpoints. It would only be needed for:

- Loading from HuggingFace safetensors format
- Converting between different checkpoint formats
- Handling key name mismatches

## Evaluation Methods

### 1. Log-Likelihood (`loglikelihood`)

Computes the log probability of continuation tokens given context:

```python
def _loglikelihood_tokens(self, requests, ...):
    # For each (context, continuation) pair:
    # 1. Concatenate and truncate to max_length
    # 2. Get logits from model
    # 3. Compute log softmax
    # 4. Extract log probs for continuation tokens
    # 5. Check if greedy decoding matches
    # Returns: (log_prob_sum, is_greedy_match)
```

**Used by tasks**: Most multiple-choice tasks (HellaSwag, ARC, MMLU, etc.)

### 2. Log-Likelihood Rolling (`loglikelihood_rolling`)

Computes perplexity over long sequences using rolling windows:

```python
def loglikelihood_rolling(self, requests, ...):
    # For each long sequence:
    # 1. Break into overlapping windows
    # 2. Compute log-likelihood for each window
    # 3. Sum across all windows
    # Returns: total_log_likelihood
```

**Used by tasks**: Perplexity evaluation (WikiText, etc.)

### 3. Generation (`generate_until`)

Generates text continuations until stop sequences:

```python
def generate_until(self, requests, ...):
    # For each context:
    # 1. Tokenize and truncate context
    # 2. Generate tokens greedily
    # 3. Stop at EOS or stop sequences
    # 4. Return generated text (without context)
```

**Used by tasks**: Open-ended generation tasks

**Note**: Currently implements simple greedy decoding. Advanced sampling (temperature, top-p, top-k) not yet implemented.

## Tokenization

### Tokenizer Loading

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    use_fast=True,
)

# Configure padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Special Tokens

- **BOS (Beginning of Sequence)**: Used as prefix token for log-likelihood
- **EOS (End of Sequence)**: Used as padding token and generation stop token
- **PAD**: Set to EOS if not defined

## Limitations & Future Work

### Current Limitations

1. **Single Device Only**: No distributed evaluation support (unlike training)
2. **Greedy Decoding Only**: No sampling, temperature, or nucleus sampling
3. **No Quantization**: Full precision only (bfloat16/float32)
4. **MoE Only**: Currently only supports `llama3_moe` architecture
5. **No Attention Mask**: TorchTitan models don't use attention masks

### Planned Enhancements

1. **Multi-GPU Support**: Add data parallelism for faster evaluation
2. **Advanced Generation**: Implement sampling strategies
3. **More Architectures**: Support standard Llama3 (non-MoE)
4. **Quantization**: Add int8/int4 quantization support
5. **Attention Mask**: Add optional attention mask support for padding

## Troubleshooting

### Common Issues

#### 1. Import Error: TorchTitan Not Found

```
ImportError: Failed to import TorchTitan modules
```

**Solution**: Ensure TorchTitan is installed:
```bash
cd repos/torchtitan-fork
pip install -e .
```

#### 2. Checkpoint Not Found

```
WARNING: No checkpoint found at /path/to/checkpoint, using random initialization
```

**Solution**: Verify checkpoint path exists and contains DCP files:
```bash
ls /path/to/checkpoint/
# Should show: __0_0.distcp, .metadata, etc.
```

#### 3. Model Flavor Not Found

```
ValueError: Model flavor 'XYZ' not found. Available: ['1B', '3B', '8B', ...]
```

**Solution**: Use one of the available flavors listed in the error message or check `llama3_moe_configs` in TorchTitan.

#### 4. Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
- Reduce `batch_size`
- Reduce `max_length`
- Use smaller model flavor
- Use CPU: `--model_args device=cpu`

#### 5. Tokenizer Not Found

```
RuntimeError: Failed to load tokenizer
```

**Solution**: Provide valid tokenizer path:
```bash
--model_args tokenizer_path=/path/to/tokenizer
```

## Performance Considerations

### Memory Usage

Approximate GPU memory requirements (bfloat16):

| Model Flavor | Parameters | GPU Memory (Eval) |
|--------------|------------|-------------------|
| debugmodel_8exp | ~1M | <1 GB |
| 1B | ~1B | ~4 GB |
| 3B | ~3B | ~8 GB |
| 8B | ~8B | ~20 GB |

### Batch Size Recommendations

| Model Size | Recommended Batch Size | Max Sequence Length |
|------------|------------------------|---------------------|
| 1B | 8-16 | 2048 |
| 3B | 4-8 | 2048 |
| 8B | 1-4 | 2048 |

### Optimization Tips

1. **Use bfloat16**: Default dtype, good balance of speed and accuracy
2. **Batch Evaluation**: Increase batch_size for throughput
3. **Truncate Sequences**: Set appropriate max_length for your tasks
4. **Cache Results**: lm-eval automatically caches results

## Code Examples

### Custom Evaluation Script

```python
from lm_eval.models.torchtitan import TorchTitanLM
from lm_eval import evaluator

# Initialize model
model = TorchTitanLM(
    model_name="llama3_moe",
    model_flavor="1B",
    checkpoint_path="/checkpoints/llama3_moe_1B",
    tokenizer_path="/tokenizers/llama3",
    batch_size=8,
    device="cuda",
)

# Run evaluation
results = evaluator.simple_evaluate(
    model=model,
    tasks=["hellaswag", "arc_easy"],
    num_fewshot=0,
    batch_size=8,
)

print(results)
```

### Batch Processing

```python
# Process multiple checkpoints
checkpoints = [
    "/checkpoints/step_1000",
    "/checkpoints/step_2000",
    "/checkpoints/step_3000",
]

for ckpt in checkpoints:
    model = TorchTitanLM(
        model_name="llama3_moe",
        model_flavor="1B",
        checkpoint_path=ckpt,
    )
    results = evaluator.simple_evaluate(model=model, tasks=["hellaswag"])
    print(f"{ckpt}: {results['results']['hellaswag']['acc']}")
```

## References

### Related Files

- **Model Implementation**: `repos/lm-eval-fork/lm_eval/models/torchtitan.py`
- **TorchTitan MoE Model**: `repos/torchtitan-fork/torchtitan/models/llama3_moe/`
- **TorchTitan Checkpoint**: `repos/torchtitan-fork/torchtitan/components/checkpoint.py`
- **HuggingFace Integration**: `repos/lm-eval-fork/lm_eval/models/huggingface.py`

### Documentation

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)

## Contributing

To extend this integration:

1. **Add New Architectures**: Modify `_create_model()` to support more model types
2. **Improve Generation**: Enhance `_model_generate()` with sampling strategies
3. **Add Distributed Support**: Implement multi-GPU evaluation
4. **Optimize Performance**: Add model compilation, quantization, etc.

## License

This integration follows the licenses of:
- lm-evaluation-harness (MIT)
- TorchTitan (BSD-3-Clause)