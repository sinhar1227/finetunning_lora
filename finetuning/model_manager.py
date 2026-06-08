import gc
import os
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    TrainingArguments,
    Trainer,
)

from .config import Config


class ModelManager:
    def __init__(self, config: Config, use_cuda: Optional[bool] = None):
        self.config = config
        self.use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def _clear_memory(self) -> None:
        gc.collect()
        if self.use_cuda:
            torch.cuda.empty_cache()

    def load_base_model(self) -> AutoModelForCausalLM:
        self._clear_memory()
        if self.use_cuda:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        model.config.use_cache = False
        return model

    def create_peft_model(self, base_model: AutoModelForCausalLM) -> PeftModel:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        return get_peft_model(base_model, lora_config)

    def build_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> DataCollatorForLanguageModeling:
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def build_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=max(1, int(self.config.warmup_ratio * self.config.max_steps if self.config.max_steps > 0 else self.config.num_train_epochs * 100)),
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            logging_first_step=self.config.logging_first_step,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            fp16=self.use_cuda,
            bf16=False,
            report_to="none",
            remove_unused_columns=False,
        )

    def build_trainer(
        self,
        model: PeftModel,
        train_dataset,
        eval_dataset,
        data_collator: DataCollatorForLanguageModeling,
    ) -> Trainer:
        return Trainer(
            model=model,
            args=self.build_training_args(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

    def save_adapter(self, model: PeftModel, tokenizer: PreTrainedTokenizerBase, adapter_dir: str) -> None:
        Path(adapter_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    def merge_adapter(self, adapter_dir: str, merged_model_dir: str) -> None:
        Path(merged_model_dir).mkdir(parents=True, exist_ok=True)
        if self.use_cuda:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        merged = PeftModel.from_pretrained(base_model, adapter_dir).merge_and_unload()
        merged.save_pretrained(merged_model_dir)
