from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import Config


class InferenceEngine:
    def __init__(self, config: Config, use_cuda: Optional[bool] = None):
        self.config = config
        self.use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
        self.tokenizer = None
        self.model = None

    def load(self, adapter_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.use_cuda:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                ),
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.eval()
        return self

    def generate_completion(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Inference engine has not been loaded. Call load(adapter_dir) first.")

        device = "cuda" if self.use_cuda else "cpu"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
