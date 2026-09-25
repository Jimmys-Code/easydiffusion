"""Install public ComfyUI Krea 2 weights in a shared models folder."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request


REPOSITORY = "Comfy-Org/Krea-2"
LORAS = (
    "darkbrush", "dotmatrix", "kidsdrawing", "neondrip", "rainywindow",
    "retroanime", "softwatercolor", "sunsetblur", "vintagetarot",
)
MODELS = {
    "turbo-int8": "diffusion_models/krea2_turbo_int8_convrot.safetensors",
    "turbo-nvfp4": "diffusion_models/krea2_turbo_nvfp4.safetensors",
    "raw-int8": "diffusion_models/krea2_raw_int8_convrot.safetensors",
}


def remote_files(folder):
    url = f"https://huggingface.co/api/models/{REPOSITORY}/tree/main/{folder}?recursive=false"
    with urllib.request.urlopen(url, timeout=30) as response:
        return {item["path"]: item for item in json.load(response) if item["type"] == "file"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(models_dir, variant, loras):
    paths = [MODELS[variant], "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", "vae/qwen_image_vae.safetensors"]
    paths.extend(f"loras/krea2_{name}.safetensors" for name in loras)
    catalog = {}
    for folder in {path.split("/", 1)[0] for path in paths}:
        catalog.update(remote_files(folder))

    pending = []
    remaining = 0
    for path in paths:
        info = catalog[path]
        destination = os.path.join(models_dir, path)
        if os.path.exists(destination):
            if os.path.getsize(destination) != info["size"] or sha256(destination) != info["lfs"]["oid"]:
                raise ValueError(f"Existing file does not match the published SHA-256: {destination}")
            print(f"Verified {destination}")
            continue
        part = destination + ".part"
        part_size = os.path.getsize(part) if os.path.exists(part) else 0
        if part_size > info["size"]:
            raise ValueError(f"Partial file exceeds the published size: {part}")
        remaining += info["size"] - part_size
        pending.append((path, info, destination, part))

    os.makedirs(models_dir, exist_ok=True)
    free = shutil.disk_usage(models_dir).free
    if free < remaining + 1024**3:
        raise OSError(f"Need {(remaining + 1024**3) / 1024**3:.1f} GiB free; only {free / 1024**3:.1f} GiB available")

    for path, info, destination, part in pending:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        url = f"https://huggingface.co/{REPOSITORY}/resolve/main/{path}"
        if not os.path.exists(part) or os.path.getsize(part) < info["size"]:
            subprocess.run(["curl", "--silent", "--show-error", "--fail", "--location", "--retry", "5", "--continue-at", "-", "--output", part, url], check=True)
        if os.path.getsize(part) != info["size"] or sha256(part) != info["lfs"]["oid"]:
            raise ValueError(f"Download failed SHA-256 verification: {part}")
        os.replace(part, destination)
        print(f"Installed {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    parser.add_argument("--variant", choices=MODELS, default="turbo-int8")
    parser.add_argument("--loras", nargs="*", choices=LORAS, default=LORAS)
    args = parser.parse_args()
    install(os.path.abspath(args.models_dir), args.variant, args.loras)


if __name__ == "__main__":
    main()
