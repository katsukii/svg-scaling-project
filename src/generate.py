"""Generate SVG samples from a trained model.

Usage:
    python src/generate.py --checkpoint results/runs/tiny_xxx/best_model.pt \
                           --config configs/tiny.yaml \
                           --num-samples 5
"""

import argparse
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

from model import ModelConfig, GPT


def load_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    mup: bool = False,
    mup_base_width: int = 128,
) -> GPT:
    """Load a trained model from config + checkpoint."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)['model']

    mc = ModelConfig(
        vocab_size=cfg['vocab_size'],
        block_size=cfg['block_size'],
        n_layer=cfg['n_layer'],
        n_head=cfg['n_head'],
        n_embd=cfg['n_embd'],
        d_ff=cfg['d_ff'],
        dropout=cfg['dropout'],
        bias=cfg['bias'],
        mup=mup,
        mup_base_width=mup_base_width,
    )
    model = GPT(mc).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Handle both raw state_dict and checkpoint dict
    if 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_svg(text: str) -> str | None:
    """Extract the first complete SVG from generated text.

    Returns the SVG string from <svg to </svg> inclusive,
    or None if no complete SVG is found (missing <svg or missing </svg>).
    """
    start = text.find('<svg')
    if start == -1:
        return None
    end = text.find('</svg>', start)
    if end == -1:
        return None
    return text[start:end + len('</svg>')]


def generate_svg(
    model: GPT,
    tokenizer: Tokenizer,
    prefix: str = "<svg",
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int | None = 50,
    device: torch.device = torch.device('cpu'),
) -> tuple[str, str | None]:
    """Generate a single SVG from a prefix string.

    Prepends <bos> to match training distribution, then appends
    prefix tokens. Stops at <eos> or max_new_tokens.

    Returns:
        (raw_text, svg_text) where svg_text is the extracted complete SVG,
        or None if the model did not produce a complete <svg>...</svg>.
    """
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    # Start with [bos] + prefix, matching training format: <bos> <svg ...> </svg> <eos>
    encoded = tokenizer.encode(prefix)
    input_ids = [bos_id] + encoded.ids
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    output = model.generate(
        idx, max_new_tokens,
        temperature=temperature, top_k=top_k,
        eos_token_id=eos_id,
    )
    generated_ids = output[0].tolist()

    # Strip special tokens before decoding
    special_ids = {bos_id, eos_id}
    generated_ids = [t for t in generated_ids if t not in special_ids]

    raw_text = tokenizer.decode(generated_ids)
    svg_text = extract_svg(raw_text)

    return raw_text, svg_text


def main():
    parser = argparse.ArgumentParser(description='Generate SVG samples')
    parser.add_argument('--config', type=str, required=True, help='Model config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--tokenizer', type=str, default='tokenizer/bpe_svg.json',
                        help='Tokenizer path')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--prefix', type=str, default='<svg')
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Save samples to this directory')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--mup', action='store_true',
                        help='Use µP parameterization (must match training)')
    parser.add_argument('--mup-base-width', type=int, default=128,
                        help='Base width for µP (default: 128)')
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Device: {device}")

    # Load model and tokenizer
    model = load_model(args.config, args.checkpoint, device,
                       mup=args.mup, mup_base_width=args.mup_base_width)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"Model loaded from {args.checkpoint}")

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples
    complete_count = 0
    for i in range(args.num_samples):
        print(f"\n--- Sample {i+1}/{args.num_samples} ---")
        raw_text, svg_text = generate_svg(
            model, tokenizer,
            prefix=args.prefix,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )

        if svg_text is not None:
            complete_count += 1
            print(svg_text[:500])
            if len(svg_text) > 500:
                print(f"... ({len(svg_text)} chars total)")
            print(f"  complete: yes ({len(svg_text)} chars)")
            if args.output_dir:
                with open(out_dir / f'sample_{i}.svg', 'w') as f:
                    f.write(svg_text)
        else:
            print(raw_text[:500])
            if len(raw_text) > 500:
                print(f"... ({len(raw_text)} chars total)")
            print(f"  complete: no (missing </svg>)")
            if args.output_dir:
                with open(out_dir / f'sample_{i}_incomplete.txt', 'w') as f:
                    f.write(raw_text)

    print(f"\n--- Summary ---")
    print(f"  Complete SVGs: {complete_count}/{args.num_samples}")
    if args.output_dir:
        print(f"  Saved to {args.output_dir}")


if __name__ == '__main__':
    main()
