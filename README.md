# LLM Fine-Tuning for Pharma Domain

This repository contains an end-to-end experiment for continued pretraining and fine-tuning of causal language models using pharmaceutical PDF content.

## Overview

- Extracts text from PDF files.
- Cleans and normalizes extracted text.
- Creates a Hugging Face `Dataset` for causal LM training.
- Tokenizes and packs text into fixed-size training blocks.
- Applies LoRA/QLoRA adapters for efficient fine-tuning.
- Supports model saving, adapter saving, and inference.

## Repository Structure

- `finetunning.ipynb` - Main notebook for exploring the pipeline and running the fine-tuning workflow interactively.
- `tokenizer.ipynb` - Notebook for tokenizer exploration and testing.
- `main.py` - Project entry point placeholder.
- `pyproject.toml` - Python packaging configuration and dependencies.
- `requirements.txt` - Required Python packages for model training and dataset processing.
- `.venv/` - Local virtual environment directory.
- `content/` - Suggested location for PDF or data files.

## Requirements

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you prefer the project packaging workflow:

```bash
pip install .
```

## Recommended Setup

1. Create or activate a Python virtual environment.
2. Install dependencies using `requirements.txt`.
3. Place your PDF files into the `content/` folder or update the paths in the notebook.
4. Open `finetunning.ipynb` in VS Code or Jupyter.

## Usage

### Interactive Notebook

Open `finetunning.ipynb` and run the notebook cells in order:

1. Load dependencies and configuration.
2. Extract PDF text.
3. Clean and split paragraphs.
4. Build datasets and tokenize.
5. Train the model with LoRA/QLoRA.
6. Save the adapter and run inference.

### Custom Script

If you want to add a script for automation, use `main.py` or create a new Python script that imports your reusable pipeline components.

## Notes

- This repository is designed for experimentation, not production deployment.
- Fine-tuning large models is resource-intensive; use a GPU-enabled environment.
- The current notebooks are the best starting point for adapting the workflow to a new PDF corpus.

## License

Add your license information here.
