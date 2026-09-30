"""Run CPIC on one image or a directory of images."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image

PROMPT = (
    "Please suggest the single best aesthetic crop region for this image. "
    "Use normalized coordinates in a 0-1000 image coordinate system: left edge x=0, "
    "right edge x=1000, top edge y=0, bottom edge y=1000. "
    'Return only JSON in this exact format: {"bbox_2d": [x1, y1, x2, y2]}.'
)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_box(text, width, height):
    """Convert the prompted 0–1000 xyxy coordinates to pixel coordinates."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object in model response: {text!r}")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    box = obj["bbox_2d"]
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"Expected four coordinates: {box!r}")
    if not all(type(v) in (int, float) and math.isfinite(v) for v in box):
        raise ValueError(f"Invalid coordinates: {box!r}")
    values = [min(1000.0, max(0.0, v)) for v in box]
    x1, x2 = sorted((values[0] * width / 1000, values[2] * width / 1000))
    y1, y2 = sorted((values[1] * height / 1000, values[3] * height / 1000))
    x1, y1 = min(x1, width - 1), min(y1, height - 1)
    pixels = [x1, y1, min(width, max(x2, x1 + 1)), min(height, max(y2, y1 + 1))]
    return box, pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/CPIC"))
    parser.add_argument("--input", required=True, type=Path, help="Image or image directory.")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    source = args.input.resolve()
    if source.is_file():
        images, root = [source], source.parent
    elif source.is_dir():
        root = source
        images = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else:
        parser.error(f"Input does not exist: {source}")
    if not images:
        parser.error("No supported images found.")
    output = args.output.resolve()
    if source.is_dir() and (output == source or source in output.parents):
        parser.error("Place --output outside the input directory.")
    if (output / "predictions.json").exists():
        parser.error("Output already contains predictions.json; choose a new --output directory.")
    if not args.model.is_dir():
        parser.error(f"Download the checkpoint first: {args.model}")

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for CPU inference.")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, local_files_only=True,
    ).to(args.device).eval()
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, path in enumerate(images, 1):
        relative = path.relative_to(root)
        record = {"image": relative.as_posix()}
        try:
            with Image.open(path) as original:
                image = original.convert("RGB")
            width, height = image.size
            inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(args.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=128)
            text = processor.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            record.update(width=width, height=height, response=text)
            box, pixels = parse_box(text, width, height)
            crop = Path("crops") / relative.parent / (relative.name + ".png")
            (output / crop).parent.mkdir(parents=True, exist_ok=True)
            integer_box = (math.floor(pixels[0]), math.floor(pixels[1]), math.ceil(pixels[2]), math.ceil(pixels[3]))
            image.crop(integer_box).save(output / crop)
            record.update(bbox_2d=box, bbox_xyxy=pixels, crop=crop.as_posix())
        except (OSError, ValueError, KeyError, TypeError) as error:
            record["error"] = str(error)
        results.append(record)
        print(f"[{index}/{len(images)}] {relative}: {'error' if 'error' in record else 'ok'}", flush=True)
    (output / "predictions.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    if any("error" in record for record in results):
        raise SystemExit("Some images failed; see predictions.json for details.")


if __name__ == "__main__":
    main()
