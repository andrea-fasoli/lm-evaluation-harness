"""
TorchTitan model adapter for lm-evaluation-harness.

Enables evaluation of TorchTitan models by loading native DCP checkpoints.
"""

import logging
import os
from datetime import timedelta
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator
from lm_eval.models.utils_hf import pad_and_concat

eval_logger = logging.getLogger(__name__)


@register_model("torchtitan", "tt")
class TorchTitanLM(TemplateLM):
    """
    TorchTitan model adapter for lm-evaluation-harness.

    Loads TorchTitan models directly from native DCP checkpoints.
    Does NOT inherit from HFLM since TorchTitan uses different:
    - Model initialization (meta device + manual loading)
    - Checkpoint format (DCP, not safetensors)
    - Configuration system (no config.json)

    Supports multiple parallelization strategies:
    - Data Parallelism (DDP) via HuggingFace Accelerate
    - Tensor Parallelism (TP) for memory-efficient inference
    - Expert Parallelism (EP) for MoE models
    - Hybrid EP+TP for maximum efficiency

    Usage:
        # Single GPU evaluation
        lm_eval --model torchtitan \
                --model_args model_name=llama3_moe,model_flavor=1B,checkpoint_path=/path/to/checkpoint,tokenizer_path=/path/to/tokenizer \
                --tasks hellaswag

        # Multi-GPU with data parallelism (4 GPUs, model replicated)
        accelerate launch --num_processes=4 \
                -m lm_eval --model torchtitan \
                --model_args model_name=llama3_moe,model_flavor=1B,checkpoint_path=/path/to/checkpoint,tokenizer_path=/path/to/tokenizer \
                --tasks hellaswag

        # Tensor Parallelism (4 GPUs, model sharded)
        torchrun --nproc_per_node=4 \
                -m lm_eval --model torchtitan \
                --model_args model_name=llama3_moe,model_flavor=1B,checkpoint_path=/path/to/checkpoint,tp_degree=4 \
                --tasks hellaswag

        # Expert Parallelism (8 GPUs, experts distributed)
        torchrun --nproc_per_node=8 \
                -m lm_eval --model torchtitan \
                --model_args model_name=llama3_moe,model_flavor=1B,checkpoint_path=/path/to/checkpoint,ep_degree=8 \
                --tasks hellaswag

        # Hybrid TP+EP (8 GPUs: 2-way TP, 4-way EP)
        torchrun --nproc_per_node=8 \
                -m lm_eval --model torchtitan \
                --model_args model_name=llama3_moe,model_flavor=1B,checkpoint_path=/path/to/checkpoint,tp_degree=2,ep_degree=4,enable_ep_tp=True \
                --tasks hellaswag

        # With model overrides
        lm_eval --model torchtitan \
                --model_args "model_name=llama3_moe,model_flavor=1B,checkpoint_path=/path/to/checkpoint,tokenizer_path=/path/to/tokenizer,model_overrides={'custom_moe_impl':'virtual_group','n_moe_layers':14,'moe_inter_dim':4096},moe_overrides={'num_experts':128,'route_scale':2,'hf_ffn_hidden_dim':8192}" \
                --tasks hellaswag
    """

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        model_name: str = "llama3_moe",
        model_flavor: str = "1B",
        checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        model_overrides: Optional[dict] = None,
        moe_overrides: Optional[dict] = None,
        # Parallelism options
        tp_degree: int = 1,
        ep_degree: int = 1,
        enable_ep_tp: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize TorchTitan model for evaluation.

        Args:
            model_name: TorchTitan model name (e.g., "llama3_moe", "llama3")
            model_flavor: Model size/config (e.g., "1B", "8B")
            checkpoint_path: Path to TorchTitan DCP checkpoint directory
            tokenizer_path: Path to tokenizer directory
            max_length: Maximum sequence length
            device: Device to load model on (ignored if using parallelism)
            batch_size: Batch size for evaluation
            dtype: Model dtype (default: bfloat16)
            model_overrides: Dict of model config overrides (e.g., {"custom_moe_impl": "virtual_group", "is_moe_list": [False, True, ...]})
            moe_overrides: Dict of MoE config overrides (e.g., {"num_experts": 128, "route_scale": 2})
            tp_degree: Tensor parallelism degree (default: 1, no TP)
            ep_degree: Expert parallelism degree (default: 1, no EP)
            enable_ep_tp: Enable hybrid EP+TP parallelism (default: False)
        """

        super().__init__()
        self.model_name = model_name
        self.model_flavor = model_flavor
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path or checkpoint_path
        self._max_length = max_length or self._DEFAULT_MAX_LENGTH
        self._batch_size = int(batch_size) if isinstance(batch_size, str) else batch_size
        self._dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        self.model_overrides = model_overrides or {}
        self.moe_overrides = moe_overrides or {}

        # Store parallelism configuration
        self.tp_degree = tp_degree
        self.ep_degree = ep_degree
        self.enable_ep_tp = enable_ep_tp

        # Set backend to causal (decoder-only)
        self.backend = "causal"

        # Determine parallelism strategy
        self.use_model_parallel = (tp_degree > 1 or ep_degree > 1)

        if self.use_model_parallel:
            # Model parallelism (TP/EP) - requires torch.distributed
            self._init_model_parallelism()
        else:
            # Try data parallelism via Accelerate
            self._init_data_parallelism(device)

        # Initialize model and tokenizer
        self._create_model()
        self._create_tokenizer()

        eval_logger.info(f"TorchTitanLM initialized: {model_name}/{model_flavor}")

    def _init_model_parallelism(self) -> None:
        """Initialize model parallelism (TP/EP) using torch.distributed."""
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()

        # Validate world size matches parallelism configuration
        expected_world_size = self.tp_degree * self.ep_degree
        if self._world_size != expected_world_size:
            raise ValueError(
                f"World size ({self._world_size}) does not match "
                f"tp_degree ({self.tp_degree}) * ep_degree ({self.ep_degree}) = {expected_world_size}"
            )

        # Create device mesh for model parallelism
        from torch.distributed.device_mesh import init_device_mesh

        if self.tp_degree > 1 and self.ep_degree > 1:
            # 2D mesh: (ep, tp)
            self.device_mesh = init_device_mesh(
                "cuda",
                (self.ep_degree, self.tp_degree),
                mesh_dim_names=("ep", "tp")
            )
            eval_logger.info(
                f"Model parallelism enabled: TP={self.tp_degree}, EP={self.ep_degree}, "
                f"rank={self._rank}/{self._world_size}, 2D mesh"
            )
        elif self.tp_degree > 1:
            # 1D mesh: tp only
            self.device_mesh = init_device_mesh(
                "cuda",
                (self.tp_degree,),
                mesh_dim_names=("tp",)
            )
            eval_logger.info(
                f"Tensor parallelism enabled: TP={self.tp_degree}, "
                f"rank={self._rank}/{self._world_size}"
            )
        elif self.ep_degree > 1:
            # 1D mesh: ep only
            self.device_mesh = init_device_mesh(
                "cuda",
                (self.ep_degree,),
                mesh_dim_names=("ep",)
            )
            eval_logger.info(
                f"Expert parallelism enabled: EP={self.ep_degree}, "
                f"rank={self._rank}/{self._world_size}"
            )

        self._device = torch.device(f"cuda:{self._rank}")

        # Create a minimal accelerator-like object for compatibility with lm-eval
        # This provides the gather() method needed by the evaluator
        self.accelerator = self._create_minimal_accelerator()

    def _create_minimal_accelerator(self):
        """Create a minimal accelerator-like object for lm-eval compatibility."""
        class MinimalAccelerator:
            def __init__(self, rank, world_size, device):
                self.local_process_index = rank
                self.num_processes = world_size
                self.device = device

            def gather(self, tensor):
                """Gather tensors from all ranks using torch.distributed."""
                if not torch.distributed.is_initialized():
                    return tensor

                # Ensure tensor is on the correct device
                if not tensor.is_cuda:
                    tensor = tensor.to(self.device)

                # Gather from all ranks
                world_size = torch.distributed.get_world_size()
                gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_tensors, tensor)

                # Stack into single tensor
                return torch.stack(gathered_tensors)

            def wait_for_everyone(self):
                """Synchronize all processes using a barrier."""
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

        return MinimalAccelerator(self._rank, self._world_size, self._device)

    def _init_data_parallelism(self, device: Optional[str]) -> None:
        """Initialize data parallelism via Accelerate."""
        try:
            from accelerate import Accelerator, InitProcessGroupKwargs

            accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])

            if accelerator.num_processes > 1:
                self.accelerator = accelerator
                self._rank = accelerator.local_process_index
                self._world_size = accelerator.num_processes
                self._device = accelerator.device
                eval_logger.info(
                    f"Data parallelism enabled: {self._world_size} processes, "
                    f"rank {self._rank}, device {self._device}"
                )
            else:
                # Single process mode
                self._rank = 0
                self._world_size = 1
                self._device = torch.device(device if device else "cuda")
                eval_logger.info(f"Single process mode, using device: {self._device}")
        except ImportError:
            # Accelerate not available, fall back to single process
            eval_logger.info("Accelerate not available, using single process mode")
            self._rank = 0
            self._world_size = 1
            self._device = torch.device(device if device else "cuda")

    def _create_model(self) -> None:
        """
        Create and load TorchTitan model following train.py pattern:
        1. Import model class and configs
        2. Get model args for specified flavor
        3. Initialize model on meta device
        4. Load checkpoint using CheckpointManager
        5. Move to device and set eval mode
        """
        eval_logger.info(f"Loading TorchTitan model: {self.model_name}/{self.model_flavor}")

        # Import TorchTitan modules
        try:
            if self.model_name == "llama3_moe":
                from torchtitan.models.llama3_moe import (
                    Llama3MoE,
                    llama3_moe_configs,
                    Llama3MoEStateDictAdapter,
                )

                model_cls = Llama3MoE
                configs = llama3_moe_configs
                state_dict_adapter_cls = Llama3MoEStateDictAdapter
            # elif self.model_name == "llama3":
            #     from torchtitan.models.llama3 import (
            #         Transformer,
            #         llama3_configs,
            #         Llama3StateDictAdapter,
            #     )
            #     from torchtitan.components.checkpoint import CheckpointManager, ModelWrapper
            #     from torchtitan.config import TORCH_DTYPE_MAP

            #     model_cls = Transformer
            #     configs = llama3_configs
            #     state_dict_adapter_cls = Llama3StateDictAdapter
            else:
                raise ValueError(f"Unsupported model_name: {self.model_name}")
        except ImportError as e:
            raise ImportError(
                f"Failed to import TorchTitan modules. Ensure torchtitan is installed. Error: {e}"
            )

        # Get model configuration
        if self.model_flavor not in configs:
            raise ValueError(
                f"Model flavor '{self.model_flavor}' not found. "
                f"Available: {list(configs.keys())}"
            )

        model_args = configs[self.model_flavor]

        # Apply model overrides (following update_from_config pattern in args.py)
        if self.model_overrides:
            eval_logger.info(f"Applying model overrides: {self.model_overrides}")
            for key, value in self.model_overrides.items():
                if value is not None:
                    # Handle special case for is_moe_list which might be passed as string
                    if key == "is_moe_list" and isinstance(value, str):
                        # Evaluate string representation of list
                        import ast
                        value = ast.literal_eval(value)

                    # Handle n_moe_layers convenience field (from custom_args.py)
                    if key == "n_moe_layers":
                        if value > model_args.n_layers - 1:
                            raise ValueError(
                                f"n_moe_layers={value} must be <= n_layers-1={model_args.n_layers-1}"
                            )
                        model_args.is_moe_list = (
                            (model_args.n_layers - value - 1) * [False]
                            + value * [True]
                            + [False]
                        )
                        eval_logger.info(f"  Set is_moe_list from n_moe_layers={value}")
                    elif hasattr(model_args, key):
                        setattr(model_args, key, value)
                        eval_logger.info(f"  Set {key} = {value}")
                    else:
                        eval_logger.warning(f"  Unknown model arg: {key}")

        # Apply MoE overrides (following update_from_config pattern in args.py)
        if self.moe_overrides and hasattr(model_args, 'moe_args'):
            eval_logger.info(f"Applying MoE overrides: {self.moe_overrides}")
            for key, value in self.moe_overrides.items():
                if value is not None and hasattr(model_args.moe_args, key):
                    setattr(model_args.moe_args, key, value)
                    eval_logger.info(f"  Set moe_args.{key} = {value}")
                elif value is not None:
                    eval_logger.warning(f"  Unknown MoE arg: {key}")

        # Validate is_moe_list length (from args.py lines 97-103)
        if (
            getattr(model_args, "is_moe_list", None) is not None
            and len(model_args.is_moe_list) != model_args.n_layers
        ):
            raise ValueError(
                f"is_moe_list must be None or have {model_args.n_layers} elements. "
                f"Got {len(model_args.is_moe_list)} elements."
            )

        self.model_args = model_args

        eval_logger.info(f"Final model config: {model_args}")

        # Initialize model on meta device (following train.py pattern)
        with torch.device("meta"):
            self._model = model_cls(model_args)

        # Apply model parallelism BEFORE materializing weights
        if self.use_model_parallel:
            self._apply_model_parallelism()

        # Move to device and initialize weights
        self._model.to_empty(device=self._device)
        with torch.no_grad():
            self._model.init_weights()

        # Load checkpoint if provided
        if self.checkpoint_path and os.path.isdir(self.checkpoint_path):
            eval_logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            self._load_checkpoint(
                checkpoint_path=self.checkpoint_path,
                model_args=model_args,
                state_dict_adapter_cls=state_dict_adapter_cls,
            )
        else:
            eval_logger.warning(
                f"No checkpoint found at {self.checkpoint_path}, using random initialization"
            )

        # Set to eval mode
        self._model.eval()
        torch.set_grad_enabled(False)

        eval_logger.info(f"Model loaded successfully on {self._device}")

    def _apply_model_parallelism(self) -> None:
        """Apply TP/EP parallelism to the model using torchtitan functions."""
        eval_logger.info("Applying model parallelism...")

        # Import parallelization functions from torchtitan
        try:
            from torchtitan.experiments.llama4.infra.parallelize import (
                apply_moe_ep_tp,
                apply_non_moe_tp,
            )
        except ImportError:
            raise ImportError(
                "Failed to import torchtitan parallelization functions. "
                "Ensure torchtitan is installed with llama4 experiments."
            )

        # Determine which meshes to use
        tp_mesh = self.device_mesh["tp"] if self.tp_degree > 1 else None
        ep_mesh = self.device_mesh["ep"] if self.ep_degree > 1 else None
        ep_tp_mesh = None

        if self.tp_degree > 1 and self.ep_degree > 1:
            ep_tp_mesh = self.device_mesh["ep", "tp"] if self.enable_ep_tp else None

        # Apply TP to non-MoE layers
        if tp_mesh is not None:
            eval_logger.info(f"Applying Tensor Parallelism (TP={self.tp_degree})...")
            apply_non_moe_tp(
                self._model,
                tp_mesh,
                loss_parallel=False,  # No loss parallelism for inference
                enable_float8_tensorwise_tp=False,
            )

        # Apply EP/TP to MoE layers
        if tp_mesh is not None or ep_mesh is not None:
            eval_logger.info(
                f"Applying MoE parallelism (TP={self.tp_degree}, EP={self.ep_degree}, "
                f"EP+TP={self.enable_ep_tp})..."
            )
            apply_moe_ep_tp(
                self._model,
                tp_mesh=tp_mesh,
                ep_mesh=ep_mesh,
                ep_tp_mesh=ep_tp_mesh,
                etp_enabled=self.enable_ep_tp,
            )

        eval_logger.info("Model parallelism applied successfully")

    def _load_checkpoint(
        self,
        checkpoint_path: str,
        model_args: Any,
        state_dict_adapter_cls: type,
    ) -> None:
        """
        Load TorchTitan checkpoint using DCP.

        Mimics the checkpoint loading from train.py but simplified for eval:
        - No optimizer/lr_scheduler loading
        - No dataloader state
        - Model weights only

        Note: state_dict_adapter_cls is passed but not used for native DCP loading.
        It would only be needed if loading from HuggingFace safetensors format.
        """
        import torch.distributed.checkpoint as dcp
        from torchtitan.components.checkpoint import ModelWrapper


        # Wrap model for checkpoint loading
        model_wrapper = ModelWrapper([self._model])
        state_dict = model_wrapper.state_dict()

        # Load checkpoint using native DCP format
        # (not using sd_adapter since we're loading native TorchTitan checkpoints)
        eval_logger.info(f"Loading DCP checkpoint from {checkpoint_path}")
        dcp.load(state_dict, checkpoint_id=checkpoint_path)

        # Apply loaded state to model
        # NOTE: hardcoded to strict=False but also works with strict=True
        model_wrapper.load_state_dict(state_dict)

        eval_logger.info("Checkpoint loaded successfully")

    def _create_tokenizer(self) -> None:
        """Load tokenizer from tokenizer_path."""
        try:
            from transformers import AutoTokenizer

            if not self.tokenizer_path or not os.path.isdir(self.tokenizer_path):
                raise ValueError(
                    f"tokenizer_path must be a valid directory. Got: {self.tokenizer_path}"
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                use_fast=True,
            )

            # Configure padding
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            eval_logger.info(f"Tokenizer loaded from {self.tokenizer_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        Forward pass through TorchTitan model.

        TorchTitan models expect only input_ids (no attention_mask).
        """
        with torch.no_grad():
            logits = self._model(inps)
            return logits

    def _debug_test_generation(self):
        """
        Debug method to test model generation capability.
        Generates 64 tokens for the prompt "What is the capital of France?"
        and prints the output before exiting.
        """
        import sys

        eval_logger.info("=" * 80)
        eval_logger.info("DEBUG: Testing model generation capability")
        eval_logger.info("=" * 80)

        # Create test prompt
        test_prompt = "The key to life is"
        eval_logger.info(f"Test prompt: {test_prompt}")

        # Tokenize prompt
        prompt_tokens = self.tok_encode(test_prompt)
        eval_logger.info(f"Prompt tokens: {prompt_tokens}")
        eval_logger.info(f"Prompt length: {len(prompt_tokens)} tokens")

        # Convert to tensor
        context_tensor = torch.tensor(
            prompt_tokens, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        # Generate 64 tokens
        max_new_tokens = 64
        eos_token_id = self.tokenizer.eos_token_id

        eval_logger.info(f"Generating {max_new_tokens} tokens...")

        generated_tokens = []
        current_context = context_tensor

        for step in range(max_new_tokens):
            with torch.no_grad():
                # Check context length vs freqs_cis
                eval_logger.info(f"Step {step}: Context length: {current_context.shape[1]}, freqs_cis length: {self._model.freqs_cis.shape[0]}")

                logits = self._model(current_context)

                # Log logits statistics at each step
                logits_last = logits[:, -1, :]
                top5_tokens = torch.topk(logits_last, k=5, dim=-1)
                top5_token_ids = top5_tokens.indices[0].tolist()
                top5_logits = top5_tokens.values[0].tolist()

                # Decode top-5 tokens for readability
                top5_decoded = [self.tok_decode([tid]) for tid in top5_token_ids]

                eval_logger.info(f"Step {step}: Top-5 token IDs: {top5_token_ids}")
                eval_logger.info(f"Step {step}: Top-5 tokens: {top5_decoded}")
                eval_logger.info(f"Step {step}: Top-5 logits: {[f'{v:.4f}' for v in top5_logits]}")

                # Greedy decoding
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                eval_logger.info(f"Step {step}: Selected token ID: {next_token.item()}, decoded: '{self.tok_decode([next_token.item()])}'")

                current_context = torch.cat([current_context, next_token], dim=1)
                generated_tokens.append(next_token.item())

                # Stop if EOS token is generated
                if next_token.item() == eos_token_id:
                    eval_logger.info(f"EOS token generated at step {step}")
                    break

        # Decode full output (prompt + generation)
        full_output = self.tok_decode(current_context[0].tolist())

        # Decode only generated part
        generated_text = self.tok_decode(generated_tokens)

        eval_logger.info("=" * 80)
        eval_logger.info("GREEDY DECODING RESULTS:")
        eval_logger.info("=" * 80)
        eval_logger.info("FULL OUTPUT (prompt + generation):")
        eval_logger.info("-" * 80)
        print(full_output)
        eval_logger.info("-" * 80)
        eval_logger.info("GENERATED TEXT (generation only):")
        eval_logger.info("-" * 80)
        print(generated_text)
        eval_logger.info("-" * 80)
        eval_logger.info(f"Generated {len(generated_tokens)} tokens")
        eval_logger.info("=" * 80)

        # Now try with temperature sampling
        eval_logger.info("\n" + "=" * 80)
        eval_logger.info("TESTING WITH TEMPERATURE SAMPLING (temperature=0.8)")
        eval_logger.info("=" * 80)

        # Reset context
        context_tensor = torch.tensor(
            prompt_tokens, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        generated_tokens_temp = []
        current_context_temp = context_tensor

        for step in range(min(max_new_tokens, 20)):  # Only 20 tokens for temperature test
            with torch.no_grad():
                logits = self._model(current_context_temp)

                # Temperature sampling
                temperature = 0.8
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                eval_logger.info(f"Temp step {step}: Selected token ID: {next_token.item()}, decoded: '{self.tok_decode([next_token.item()])}'")

                current_context_temp = torch.cat([current_context_temp, next_token], dim=1)
                generated_tokens_temp.append(next_token.item())

                if next_token.item() == eos_token_id:
                    eval_logger.info(f"EOS token generated at step {step}")
                    break

        generated_text_temp = self.tok_decode(generated_tokens_temp)
        eval_logger.info("-" * 80)
        eval_logger.info("TEMPERATURE SAMPLING OUTPUT:")
        eval_logger.info("-" * 80)
        print(generated_text_temp)
        eval_logger.info("-" * 80)
        eval_logger.info("=" * 80)

        # Exit the program
        eval_logger.info("DEBUG: Exiting program after generation test")
        sys.exit(0)

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """
        Generate text using simple greedy decoding. Implement
        basic greedy generation.
        """
        # DEBUG: Test generation on first call
        # if not hasattr(self, '_debug_test_done'):
        #     self._debug_test_done = True
        #     self._debug_test_generation()

        max_new_tokens = generation_kwargs.get("max_new_tokens", self.max_gen_toks)
        eos_token_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self._model(context)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                context = torch.cat([context, next_token], dim=1)

                if next_token.item() == eos_token_id:
                    break

                if stop:
                    decoded = self.tokenizer.decode(context[0])
                    if any(stop_seq in decoded for stop_seq in stop):
                        break

        return context

    @property
    def max_length(self):
        """Maximum sequence length supported by the model."""
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        """Maximum number of tokens to generate."""
        return 256

    @property
    def batch_size(self):
        """Batch size for evaluation."""
        return self._batch_size

    @property
    def device(self):
        """Device model is on."""
        return self._device

    @property
    def eot_token_id(self):
        """End of text token ID."""
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        """Prefix token ID for loglikelihood."""
        return self.tokenizer.bos_token_id or self.tokenizer.eos_token_id

    def tok_encode(self, string: str, **kwargs):
        """Encode string to token IDs."""
        return self.tokenizer.encode(string, **kwargs)

    def tok_decode(self, tokens, **kwargs):
        """Decode token IDs to string."""
        return self.tokenizer.decode(tokens, **kwargs)

    def _loglikelihood_tokens(
        self,
        requests,
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
        **kwargs,
    ):
        """
        Compute log-likelihood of continuation tokens given context tokens.

        This is the core method for loglikelihood evaluation. It processes batches
        of (context, continuation) pairs and returns log probabilities.

        Args:
            requests: List of tuples ((context_str, continuation_str), context_enc, continuation_enc)
            disable_tqdm: Whether to disable progress bar
            override_bs: Override batch size if provided

        Returns:
            List of tuples (log_prob, is_greedy) for each request
        """
        res = []

        def _collate(req):
            """Sort requests by total length (descending)."""
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        # Reorder requests by length for efficient batching
        re_ord = Collator(requests, sort_fn=_collate)

        batch_size = override_bs if override_bs is not None else self._batch_size
        # Ensure batch_size is valid
        if batch_size is None or batch_size <= 0:
            batch_size = 1
        chunks = re_ord.get_batched(n=batch_size, batch_fn=None)

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )

        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            padding_len_inp = None

            for _, context_enc, continuation_enc in chunk:
                # Sanity checks
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # Concatenate context and continuation, truncate from left if needed
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                (inplen,) = inp.shape

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # Pad and batch inputs
            # Ensure padding_len_inp is not None
            assert padding_len_inp is not None, "padding_len_inp should not be None"
            batched_inps = pad_and_concat(
                padding_len_inp, inps, padding_side="right"
            )

            # Get logits from model
            with torch.no_grad():
                logits = self._model_call(batched_inps)
                multi_logits = F.log_softmax(logits, dim=-1)

            # Process each request in the batch
            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                contlen = len(cont_toks)

                # Select continuation tokens from logits
                # Discard context tokens and right padding
                logits = logits[inplen - contlen : inplen]
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if greedy decoding matches continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks_tensor = torch.tensor(
                    cont_toks, dtype=torch.long, device=self.device
                ).unsqueeze(0)
                max_equal = (greedy_tokens == cont_toks_tensor).all()

                # Get log probabilities at continuation token positions
                logits = torch.gather(
                    logits, 2, cont_toks_tensor.unsqueeze(-1)
                ).squeeze(-1)

                answer = (float(logits.sum()), bool(max_equal))
                res.append(answer)

                self.cache_hook.add_partial("loglikelihood", request_str, answer)
                pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        """
        Compute log-likelihood for rolling window evaluation (for perplexity).

        This method processes long sequences by breaking them into overlapping windows
        and computing the log-likelihood for each window.

        Args:
            requests: List of Instance objects with args=(string,)
            disable_tqdm: Whether to disable progress bar

        Returns:
            List of total log-likelihoods for each request
        """
        from lm_eval import utils

        loglikelihoods = []

        for (string,) in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Running loglikelihood_rolling",
        ):
            # Tokenize the string
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # Process each window
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            # Compute log-likelihood for all windows
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True,
            )

            # Sum log-likelihoods across all windows
            string_nll = sum(nll[0] for nll in string_nll)
            loglikelihoods.append(string_nll)

            self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_nll)

        return loglikelihoods

    def generate_until(self, requests, disable_tqdm: bool = False):
        """
        Generate text continuations until stopping criteria are met.

        This method generates text for each request using the model's generation
        capabilities, stopping when a stop sequence is encountered or max length
        is reached.

        Args:
            requests: List of Instance objects with args=(context, gen_kwargs)
            disable_tqdm: Whether to disable progress bar

        Returns:
            List of generated strings (continuations only, without context)
        """
        res = []

        def _collate(req):
            """Sort requests by context length (descending)."""
            # req is a tuple (context, gen_kwargs)
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        # Extract args from Instance objects and group by gen_kwargs
        re_ord = Collator(
            [req.args for req in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )

        batch_size = self._batch_size if self._batch_size else 1
        chunks = re_ord.get_batched(n=batch_size, batch_fn=None)

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running generate_until requests",
        )

        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # All gen_kwargs in batch are the same (ensured by grouping)
            gen_kwargs = all_gen_kwargs[0]

            # Extract generation parameters
            until = gen_kwargs.get("until", [])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            # Calculate max context length
            max_ctx_len = self.max_length - max_gen_toks
            assert max_ctx_len > 0, (
                f"max_gen_toks ({max_gen_toks}) must be less than max_length ({self.max_length})"
            )

            # Process each context in the batch
            for context in contexts:
                # Tokenize and truncate context
                context_enc = self.tok_encode(context)
                context_enc = context_enc[-max_ctx_len:]
                context_tensor = torch.tensor(
                    context_enc, dtype=torch.long, device=self.device
                ).unsqueeze(0)

                # Generate
                generated = self._model_generate(
                    context_tensor,
                    max_length=context_tensor.shape[1] + max_gen_toks,
                    stop=until,
                    **gen_kwargs,
                )

                # Decode only the generated part (remove context)
                generated_tokens = generated[0, context_tensor.shape[1]:].tolist()
                generated_text = self.tok_decode(generated_tokens)

                # Handle stop sequences
                for stop_seq in until:
                    if len(stop_seq) > 0 and stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                        break

                res.append(generated_text)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), generated_text)
                pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)
