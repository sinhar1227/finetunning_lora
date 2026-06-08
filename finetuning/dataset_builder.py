from typing import Any, Dict, List

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from .config import Config


class DatasetBuilder:
    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerBase):
        self.config = config
        self.tokenizer = tokenizer

    def build_dataset(self, paragraph_records: List[Dict[str, Any]]) -> DatasetDict:
        if len(paragraph_records) < 2:
            raise ValueError("The extracted corpus is too small. Please provide more data or lower min_chars_per_paragraph.")

        text_dataset = Dataset.from_list(paragraph_records)
        split_dataset = text_dataset.train_test_split(test_size=self.config.test_size, seed=self.config.seed)
        return DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
        })

    def tokenize_dataset(self, dataset: DatasetDict) -> DatasetDict:
        def tokenize_function(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
            return self.tokenizer(examples["text"])

        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing text corpus",
        )

    def create_training_blocks(self, tokenized_datasets: DatasetDict) -> DatasetDict:
        def create_blocks(tokenized_examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
            all_input_ids = []
            all_attention_masks = []
            for input_ids in tokenized_examples["input_ids"]:
                all_input_ids.extend(input_ids)
            for attention_mask in tokenized_examples["attention_mask"]:
                all_attention_masks.extend(attention_mask)

            total_tokens = len(all_input_ids)
            usable_tokens = (total_tokens // self.config.block_size) * self.config.block_size
            if usable_tokens == 0:
                return {"input_ids": [], "attention_mask": [], "labels": []}

            all_input_ids = all_input_ids[:usable_tokens]
            all_attention_masks = all_attention_masks[:usable_tokens]

            input_id_blocks = []
            attention_mask_blocks = []
            for start_index in range(0, usable_tokens, self.config.block_size):
                end_index = start_index + self.config.block_size
                input_id_blocks.append(all_input_ids[start_index:end_index])
                attention_mask_blocks.append(all_attention_masks[start_index:end_index])

            return {
                "input_ids": input_id_blocks,
                "attention_mask": attention_mask_blocks,
                "labels": input_id_blocks.copy(),
            }

        return tokenized_datasets.map(
            create_blocks,
            batched=True,
            desc=f"Creating fixed-size training blocks of {self.config.block_size} tokens",
        )
