"""Render one Krea 2 image through the Easy Diffusion HTTP API."""

import argparse
import base64
import io
import json
import time

import requests
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:9000")
    parser.add_argument("--model", default="krea2_turbo_int8_convrot")
    parser.add_argument("--lora", action="append")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--output", default="krea2-check.png")
    args = parser.parse_args()

    request = {
        "prompt": "A red fox in a snowy forest, watercolor illustration",
        "seed": args.seed, "width": 512, "height": 512, "num_outputs": 1,
        "num_inference_steps": 8, "guidance_scale": 1, "sampler_name": "euler",
        "scheduler_name": "simple", "use_stable_diffusion_model": args.model,
        "session_id": "krea2-check", "output_format": "png", "stream_image_progress": args.preview,
    }
    loras = ["krea2_darkbrush"] if args.lora is None else [name for name in args.lora if name]
    if loras:
        request.update(
            use_lora_model=loras if len(loras) > 1 else loras[0],
            lora_alpha=[0.8] * len(loras) if len(loras) > 1 else 0.8,
        )
    response = requests.post(f"{args.url}/render", json=request, timeout=30)
    response.raise_for_status()
    stream = response.json()["stream"]

    deadline = time.monotonic() + 600
    decoder = json.JSONDecoder()
    while time.monotonic() < deadline:
        response = requests.get(f"{args.url}{stream}", stream=True, timeout=(5, 600))
        if response.status_code == 425:
            time.sleep(0.5)
            continue
        response.raise_for_status()
        response.encoding = "utf-8"
        buffer = ""
        for chunk in response.iter_content(chunk_size=65536, decode_unicode=True):
            buffer += chunk
            while buffer:
                try:
                    event, used = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                buffer = buffer[used:].lstrip()
                if "step" in event:
                    print(f"Step {event['step']}/{event['total_steps']}")
                    if event.get("output"):
                        print("Live preview received")
                if event.get("status") == "failed":
                    raise RuntimeError(event.get("detail", event))
                if event.get("status") == "succeeded":
                    data = base64.b64decode(event["output"][0]["data"].split(",", 1)[-1])
                    image = Image.open(io.BytesIO(data))
                    assert image.size == (512, 512), image.size
                    image.save(args.output)
                    print(f"Saved {args.output}")
                    return
        time.sleep(0.5)
    raise TimeoutError("Krea 2 did not render an image within ten minutes")


if __name__ == "__main__":
    main()
