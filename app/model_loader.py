import os

# ─────────────────────────────────────────────────────────────
# HuggingFace Environment Setup
# ─────────────────────────────────────────────────────────────
os.environ["HF_HOME"] = os.path.abspath("./models")
os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("./models")
os.environ["HF_DATASETS_CACHE"] = os.path.abspath("./models")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import huggingface_hub.constants as hf_constants
hf_constants.HF_TOKEN_PATH = os.path.abspath("./models/token")

os.makedirs("./models", exist_ok=True)

# ─────────────────────────────────────────────────────────────

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from config.settings import MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT

LOCAL_MODEL_PATH = "./models/qwen2.5-7b"


class ModelLoader:

    def __init__(self):
        print("CUDA available:", torch.cuda.is_available())
        self.model = None
        self.tokenizer = None

    def load(self):
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()

        # ─────────────────────────────────────────────
        # Load Tokenizer
        # ─────────────────────────────────────────────
        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=LOCAL_MODEL_PATH,
            use_fast=True
        )

        # 🔥 FIX: ensure padding is defined
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # ─────────────────────────────────────────────
        # 4-bit Quantization Config (STABLE)
        # ─────────────────────────────────────────────
        print("Configuring 4-bit quantization...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,   # 🔥 CRITICAL FIX
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # ─────────────────────────────────────────────
        # Load Base Model
        # ─────────────────────────────────────────────
        print("Loading base model...")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=LOCAL_MODEL_PATH,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16  # 🔥 CRITICAL FIX
        )

        # ─────────────────────────────────────────────
        # Attach LoRA Adapter
        # ─────────────────────────────────────────────
        print("Attaching LoRA adapter...")

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(model, lora_config)

        # 🔥 FIX: stabilize LoRA weights
        print("Stabilizing LoRA weights...")
        for name, param in self.model.named_parameters():
            if "lora" in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

        self.model.eval()

        print("Model + LoRA ready!")

        return self.model, self.tokenizer


# ─────────────────────────────────────────────────────────────
# Singleton instance
# ─────────────────────────────────────────────────────────────
model_loader = ModelLoader()