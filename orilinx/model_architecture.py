import os
import warnings
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from peft import LoraConfig, get_peft_model, TaskType

# Suppress the alibi size increase warning from DNABERT when using long sequences
warnings.filterwarnings("ignore", message=".*Increasing alibi size.*")

# Suppress transformers logging warnings about uninitialized weights and sharding
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.fsdp").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.fsdp").setLevel(logging.ERROR)


def _find_dnabert_local_path():
    """Find a DNABERT installation path.

    Priority order:
    - ORILINX_DNABERT_PATH env var (if set)
    - Any directory under a `models/` folder (searched upward from CWD and from the package location)
      whose name contains 'dnabert' (case-insensitive), or that contains a
      recognizable pretrained model marker file like 'config.json'.
    - None if nothing is found.
    """
    # Environment override
    env = os.environ.get("ORILINX_DNABERT_PATH") or os.environ.get("ORILINX_DNABERT")
    if env:
        return env

    # Build an ordered, deduplicated list of candidate roots to search for a `models/` directory.
    # This includes the current working directory (and its parents) as well as the package
    # directory (and its parents) so the code locates models/ even when the user runs the
    # CLI from elsewhere (e.g., $HOME or an HPC working directory).
    cwd = Path.cwd()
    pkg_dir = Path(__file__).resolve().parent
    candidates = []
    for p in [cwd] + list(cwd.parents) + [pkg_dir] + list(pkg_dir.parents):
        if p not in candidates:
            candidates.append(p)

    for ancestor in candidates:
        models_dir = ancestor / "models"
        if not models_dir.exists() or not models_dir.is_dir():
            continue
        # Prefer child names that mention 'dnabert'
        for child in models_dir.iterdir():
            if "dnabert" in child.name.lower():
                return str(child)
        # Otherwise, if any child looks like a HuggingFace local model (config/pytorch files), return it
        for child in models_dir.iterdir():
            if (child / "config.json").exists() or (child / "pytorch_model.bin").exists() or (child / "pytorch_model.bin.index.json").exists() or (child / "flax_model.msgpack").exists():
                return str(child)
    return None


class DnaBertOriginModel(nn.Module):
    """DNABERT-2 model with LoRA fine-tuning for origin prediction."""

    def __init__(
        self,
        model_name=None,
        enable_grad_checkpointing=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve the model path if not explicitly provided
        if model_name is None:
            found = _find_dnabert_local_path()
            if found is not None:
                model_name = found
            else:
                raise RuntimeError(
                    "DNABERT path not found. Please set ORILINX_DNABERT_PATH to a local DNABERT folder "
                    "or place DNABERT under a 'models/' directory (searched upward from CWD)."
                )

        # 1. Load the model configuration, ensuring Flash Attention is disabled
        cfg = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            return_dict=True,
        )

        # These dramatically increase activation memory in inference. Keep them off by default.
        # (Callers can still request them explicitly via kwargs on the forward call.)
        for attr, val in (
            ("output_hidden_states", False),
            ("output_attentions", False),
            ("return_dict", True),
        ):
            if hasattr(cfg, attr):
                try:
                    setattr(cfg, attr, val)
                except Exception:
                    pass
        for attr in ("use_flash_attn", "flash_attn", "use_flash_attn_mha"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, False)

        # 2. Load the base DNABERT-2 model
        # Temporarily suppress INFO/WARNING level logs to avoid sharding messages
        # Store original levels and restore after loading
        transformers_logger = logging.getLogger("transformers")
        torch_logger = logging.getLogger("torch")
        original_transformers_level = transformers_logger.level
        original_torch_level = torch_logger.level
        transformers_logger.setLevel(logging.ERROR)
        torch_logger.setLevel(logging.ERROR)
        try:
            self.dnabert = AutoModel.from_pretrained(
                model_name,
                config=cfg,
                trust_remote_code=True,
                local_files_only=True,
            )
        finally:
            transformers_logger.setLevel(original_transformers_level)
            torch_logger.setLevel(original_torch_level)

        # 3. Define the LoRA configuration ONCE
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,              # LoRA rank (optimisation parameter)
            lora_alpha=lora_alpha, # LoRA scaling (optimisation parameter)
            target_modules=[
                "Wqkv",            # Attention: Query, Key, Value
                "dense",           # Attention: Output projection
                "gated_layers",    # FFN: The first feed-forward layer
                "wo",              # FFN: The second feed-forward layer ('output')
            ],
            lora_dropout=lora_dropout,
            bias="none",
        )

        # 4. Apply PEFT to the base model ONCE
        # This function handles freezing the base model and setting up LoRA layers
        self.dnabert = get_peft_model(self.dnabert, peft_config)

        # 5. (Optional but recommended) Print trainable parameters to verify setup
        # Silenced: trainable parameter logging
        # self.dnabert.print_trainable_parameters()

        # 6. Define the classification head
        self.hidden_size = self.dnabert.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, 1)

        # 7. Enable gradient checkpointing for memory efficiency
        # Only enable gradient checkpointing if the flag is True
        if enable_grad_checkpointing:
            try:
                self.dnabert.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled.")
            except Exception as e:
                print(f"Could not enable gradient checkpointing: {e}")

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Forward pass for the model.

        Processes the output from the base model, applies the classification head,
        and returns logits for the loss function.
        """
        # Get the output from the base PEFT-tuned DNABERT model
        outputs = self.dnabert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Handle both tuple (some models / grad checkpointing) and object outputs
        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
            hidden_states = outputs[2] if len(outputs) > 2 else None
        else:
            last_hidden_state = outputs.last_hidden_state
            hidden_states = outputs.hidden_states

        # Use the embedding of the [CLS] token (the first token) for classification
        cls_embedding = last_hidden_state[:, 0, :]

        # Pass the [CLS] embedding through the classifier to get final scores (logits)
        logits = self.classifier(cls_embedding).squeeze(-1)

        # Return the logits and the hidden states in a tuple
        return logits, hidden_states


def disable_unpad_and_flash_everywhere(model):
    """Hard-disable all fast/unpadded attention flags across all model modules.
    
    Forces the model to use standard PyTorch attention paths instead of optimized
    kernels, which can be more stable in some environments.
    """
    # Try on top-level config
    cfg = getattr(getattr(model, "dnabert", model), "config", None)
    if cfg is not None:
        for nm in ("use_flash_attn", "flash_attn", "use_memory_efficient_attention", "unpad"):
            if hasattr(cfg, nm):
                try:
                    setattr(cfg, nm, False)
                except Exception:
                    pass
        # set attention_probs_dropout_prob to non-zero to force PyTorch path
        # BertUnpadSelfAttention checks: if self.p_dropout or flash_attn_qkvpacked_func is None
        # We set p_dropout > 0 to always take the PyTorch path
        if hasattr(cfg, "attention_probs_dropout_prob"):
            try:
                cfg.attention_probs_dropout_prob = 0.01  # Small nonzero value
            except Exception:
                pass
    # Try on all submodules (DNABERT variants differ in attribute names/placement)
    for m in model.modules():
        for nm in ("use_flash_attn", "flash_attn", "use_memory_efficient_attention", "unpad"):
            if hasattr(m, nm):
                try:
                    setattr(m, nm, False)
                except Exception:
                    pass
        # directly set p_dropout on BertUnpadSelfAttention modules
        if hasattr(m, "p_dropout"):
            try:
                m.p_dropout = 0.01
            except Exception:
                pass
        # Set attention_probs_dropout_prob on config objects in submodules too
        if hasattr(m, "config") and hasattr(m.config, "attention_probs_dropout_prob"):
            try:
                m.config.attention_probs_dropout_prob = 0.01
            except Exception:
                pass
