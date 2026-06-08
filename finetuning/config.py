import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    pdf_path: str = "content/Metformin-Lipid-Therapy-Knowledge.pdf"
    model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    output_dir: str = "content/pharma_tinyllama_lora_output"
    adapter_dir: str = "content/pharma_tinyllama_lora_adapter"
    processed_data_dir: str = "content/pharma_processed_data"
    min_chars_per_paragraph: int = 80
    block_size: int = 512
    test_size: float = 0.15
    seed: int = 42
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    num_train_epochs: float = 10.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 1
    logging_first_step: bool = True
    eval_steps: int = 10
    save_steps: int = 25
    save_total_limit: int = 2
    max_steps: int = -1

    def ensure_directories(self) -> None:
        for path in (self.output_dir, self.adapter_dir, self.processed_data_dir):
            os.makedirs(path, exist_ok=True)

    def resolve_paths(self) -> None:
        self.output_dir = str(Path(self.output_dir))
        self.adapter_dir = str(Path(self.adapter_dir))
        self.processed_data_dir = str(Path(self.processed_data_dir))

    def initialize(self) -> None:
        self.resolve_paths()
        self.ensure_directories()
