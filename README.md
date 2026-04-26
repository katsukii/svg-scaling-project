# Scaling Laws for Language Models on SVG Code

An empirical study of neural scaling laws applied to SVG (Scalable Vector Graphics) code generation using decoder-only Transformers. This project trains GPT-style language models of varying sizes on SVG data and fits power-law scaling curves to characterize how validation loss decreases with model size.

The project also compares **Standard Parameterization (SP)** with **Maximal Update Parameterization (muP)** to evaluate learning rate transferability across model scales.

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU recommended for training (tested on A100 40GB via Google Colab)

### Installation

```bash
git clone https://github.com/<user>/svg-scaling-project.git
cd svg-scaling-project
pip install -r requirements.txt
```

## Data Preparation

### 1. Preprocessing

Download and clean SVG data from HuggingFace (`nicholasKluge/svg-icons-simple`):

```bash
python -m src.preprocess \
    --input-dir data/raw/svg-icons-simple \
    --output-dir data/processed \
    --min-len 50
```

This pipeline:
- Strips HTML/XML comments
- Removes metadata elements (`<metadata>`, `<title>`, `<desc>`)
- Normalizes coordinate precision to 1 decimal place
- Compresses unnecessary whitespace
- Validates well-formed XML
- Validates rendering via CairoSVG
- Filters out SVGs shorter than 50 characters

### 2. Tokenization

Train a BPE tokenizer and convert to binary format:

```bash
python -m src.tokenize_data \
    --input-dir data/processed \
    --output-dir data/tokenized \
    --vocab-size 4096 \
    --max-token-len 2048
```

Sequences exceeding the maximum token length are filtered out.

## Training

### Model Configurations

Five model sizes are provided in `configs/`:

| Config | Params | Layers | Heads | d_model | d_ff  |
|--------|--------|--------|-------|---------|-------|
| tiny   | ~1.3M  | 4      | 4     | 128     | 512   |
| small  | ~5.8M  | 6      | 6     | 256     | 1024  |
| medium | ~21.3M | 8      | 8     | 512     | 2048  |
| large  | ~47.4M | 10     | 10    | 640     | 2560  |
| xl     | ~88.1M | 12     | 12    | 768     | 3072  |

All models use the same effective token batch size (16,384 tokens/step). The XL config uses gradient accumulation (`grad_accum_steps: 2`) to match this while fitting in GPU memory.

### Standard Parameterization (SP)

```bash
python -m src.train --config configs/tiny.yaml --run-name tiny
python -m src.train --config configs/xl.yaml --run-name xl
```

### muP (Maximal Update Parameterization)

Uses the [mup](https://github.com/microsoft/mup) package for width-independent hyperparameter transfer:

```bash
python -m src.train --config configs/tiny.yaml --run-name tiny --mup
python -m src.train --config configs/xl.yaml --run-name xl --mup
```

### Resume Training

```bash
python -m src.train --config configs/xl.yaml --run-name xl --mup --resume
```

## Generation

Generate SVG samples from a trained checkpoint:

```bash
# Unconditional generation
python -m src.generate \
    --config configs/xl.yaml \
    --checkpoint results/runs/mup_scaling_study/xl/best_model.pt \
    --mup \
    --num-samples 10 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.95

# Prefix-conditioned generation
python -m src.generate \
    --config configs/xl.yaml \
    --checkpoint results/runs/mup_scaling_study/xl/best_model.pt \
    --mup \
    --prefix '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"' \
    --temperature 0.8
```

## Evaluation

Run quantitative evaluation on generated samples:

```bash
python -m src.evaluate \
    --config configs/xl.yaml \
    --checkpoint results/runs/mup_scaling_study/xl/best_model.pt \
    --mup \
    --samples-dir results/samples/ \
    --test-data data/tokenized/test.bin
```

Metrics: test perplexity, XML validity rate, SVG render rate, structural validity.

## Analysis Scripts

```bash
# SP vs muP scaling law comparison + power law fit + extrapolation
python scripts/analyze_mup.py

# SP-only scaling analysis
python scripts/analyze_scaling.py

# muP coordinate check (activation norm stability across widths)
python scripts/coord_check.py

# Token sequence length histogram
python scripts/plot_token_histogram.py

# Dataset example renders (simple/medium/complex grid)
python scripts/render_examples.py
```

## Repository Structure

```
svg-scaling-project/
├── configs/              # Model size configurations (tiny through xl)
├── data/                 # Raw and processed data (gitignored)
├── docs/                 # Project specification
├── report/               # Final PDF report
├── results/
│   ├── plots/            # Generated analysis plots
│   ├── runs/             # Training results per model (gitignored)
│   └── samples/          # Generated SVG samples (gitignored)
├── scripts/              # Analysis, visualization, and experiment scripts
│   ├── analyze_mup.py    # SP vs muP scaling comparison
│   ├── analyze_scaling.py# SP scaling law analysis
│   ├── coord_check.py    # muP coordinate check
│   ├── plot_token_histogram.py
│   ├── render_examples.py
│   └── colab_*.ipynb     # Colab training notebooks
├── src/
│   ├── preprocess.py     # SVG cleaning and filtering pipeline
│   ├── tokenize_data.py  # BPE tokenization
│   ├── model.py          # GPT model (SP and muP modes)
│   ├── train.py          # Training loop with gradient accumulation
│   ├── generate.py       # SVG generation with top-k/top-p sampling
│   └── evaluate.py       # Quantitative evaluation
├── tokenizer/            # Trained BPE tokenizer files
├── requirements.txt
└── README.md
```

## Attribution

The model architecture and training loop are adapted from [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy. Key modifications and original implementations:

**Borrowed from nanoGPT** (with modifications):
- GPT class structure (CausalSelfAttention, MLP, Block)
- Batch sampling from memory-mapped binary data
- Cosine learning rate schedule with warmup

**Modified or implemented from scratch:**
- muP integration via the `mup` package (MuReadout, MuAdamW, base shape setup)
- SVG-specific preprocessing pipeline (coordinate normalization, render validation)
- BPE tokenizer training on SVG data
- Gradient accumulation for uniform token batch sizes
- Top-p (nucleus) sampling
- Evaluation pipeline (XML validity, render rate, structural checks)
- All analysis and visualization scripts
- Power law fitting with confidence intervals

## References

- Kaplan et al. (2020). "Scaling Laws for Neural Language Models"
- Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models" (Chinchilla)
- Yang et al. (2022). "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (muP)
