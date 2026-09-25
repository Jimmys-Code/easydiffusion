import atexit
import asyncio
import io
import json
import os
import socket
import struct
import subprocess
import tempfile
import time
import uuid

import aiohttp
import requests
from PIL import Image
from sdkit.utils import img_to_base64_str

from easydiffusion.app import getConfig
from easydiffusion.model_manager import get_model_dirs, resolve_model_to_use
from easydiffusion.utils.model_identifier import identify_model_type

from sdkit_common import create_sdkit_context, filter_images as sdkit_filter_images
from sdkit_common import load_model as sdkit_load_model, unload_model as sdkit_unload_model


ed_info = {"name": "Krea 2 via ComfyUI", "version": (1, 0, 0), "type": "backend"}
image_size_multiple = 16

_process = None
_model_paths_file = None
_url = "http://127.0.0.1:8188"
_filter_models = {"gfpgan", "realesrgan", "codeformer"}


def install_backend():
    pass


def uninstall_backend():
    pass


def is_installed():
    return True


def start_backend():
    global _process, _model_paths_file, _url
    config = getConfig().get("backend_config") or {}
    _url = config.get("comfyui_url", "http://127.0.0.1:8188").rstrip("/")
    comfy_dir = config.get("comfyui_dir") or os.getenv("COMFYUI_DIR")
    if ping(timeout=5):
        return
    if not comfy_dir:
        raise ConnectionError("Set the ComfyUI folder in Easy Diffusion settings or start ComfyUI at " + _url)

    python = os.path.join(comfy_dir, ".venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join(comfy_dir, ".venv", "bin", "python")
    if not os.path.isfile(python):
        raise FileNotFoundError(f"ComfyUI Python environment not found: {python}")
    from urllib.parse import urlparse

    address = urlparse(_url)
    if address.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("Automatic ComfyUI startup requires a local URL")
    try:
        with socket.create_connection((address.hostname, address.port or 8188), timeout=2):
            return
    except OSError:
        pass
    from easydiffusion import app

    paths = {
        "easy_diffusion": {
            "base_path": app.MODELS_DIR,
            "diffusion_models": "stable-diffusion\ndiffusion_models",
            "text_encoders": "text-encoder\ntext_encoders",
            "vae": "vae",
            "loras": "lora\nloras",
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as file:
        json.dump(paths, file)
        _model_paths_file = file.name
    _process = subprocess.Popen(
        [python, "main.py", "--listen", "127.0.0.1", "--port", str(address.port or 8188),
         "--preview-method", "auto", "--disable-all-custom-nodes", "--disable-api-nodes",
         "--disable-metadata", "--extra-model-paths-config", _model_paths_file], cwd=comfy_dir,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    atexit.register(stop_backend)


def stop_backend():
    global _process, _model_paths_file
    if _process is not None and _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _process.kill()
    _process = None
    if _model_paths_file and os.path.exists(_model_paths_file):
        os.unlink(_model_paths_file)
    _model_paths_file = None


def ping(timeout=1):
    try:
        response = requests.get(f"{_url}/system_stats", timeout=timeout)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def create_context():
    context = create_sdkit_context(use_diffusers=False)
    context.models = {}
    context.model_paths = {}
    context.model_configs = {}
    context.vram_optimizations = set()
    context.vram_usage_level = "balanced"
    context.active_prompt_id = None
    context.options = {}
    return context


def load_model(context, model_type, **kwargs):
    path = context.model_paths.get(model_type)
    if path is None:
        return
    if model_type == "stable-diffusion":
        if identify_model_type(path) != "krea2":
            raise ValueError("The Krea 2 backend requires a Krea 2 diffusion model")
    elif model_type in _filter_models:
        sdkit_load_model(context, model_type, **kwargs)
        return
    elif model_type not in ("vae", "text-encoder", "lora"):
        raise ValueError(f"The Krea 2 backend does not support {model_type} models")
    context.models[model_type] = path


def unload_model(context, model_type, **kwargs):
    if model_type in _filter_models and model_type in context.models:
        sdkit_unload_model(context, model_type, **kwargs)
    context.models.pop(model_type, None)


def set_options(context, **kwargs):
    context.options.update(kwargs)


def _model_name(context, model_type, default=None):
    path = context.model_paths.get(model_type)
    if not path and default:
        path = resolve_model_to_use(default, model_type)
    if not path:
        raise FileNotFoundError(f"Select a {model_type} model for Krea 2")
    if isinstance(path, list):
        if len(path) != 1:
            raise ValueError(f"Krea 2 requires one {model_type} model")
        path = path[0]
    for directory in get_model_dirs(model_type):
        if os.path.commonpath((os.path.abspath(path), os.path.abspath(directory))) == os.path.abspath(directory):
            return os.path.relpath(path, directory).replace(os.sep, "/")
    raise ValueError(f"Model is outside the configured {model_type} folders: {path}")


def _request(method, path, **kwargs):
    response = requests.request(method, f"{_url}{path}", timeout=kwargs.pop("timeout", 30), **kwargs)
    response.raise_for_status()
    return response


def _workflow(model, encoder, vae, prompt, negative, seed, width, height, steps, cfg, sampler, scheduler, loras):
    graph = {
        "model": {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": encoder, "type": "krea2", "device": "default"}},
        "positive": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
    }
    model_output = ["model", 0]
    for index, (name, strength) in enumerate(loras):
        node = f"lora_{index}"
        graph[node] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_output, "lora_name": name, "strength_model": strength},
        }
        model_output = [node, 0]
    if cfg > 1:
        graph["negative"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": negative}}
    else:
        graph["negative"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["positive", 0]}}
    graph["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_output, "positive": ["positive", 0], "negative": ["negative", 0],
            "latent_image": ["latent", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": scheduler, "denoise": 1,
        },
    }
    graph["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    graph["save"] = {"class_type": "PreviewImage", "inputs": {"images": ["decode", 0]}}
    return graph


async def _run_prompt(context, graph, callback, client_id, output_type, steps):
    websocket_url = _url.replace("http", "ws", 1) + "/ws"
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(websocket_url, params={"clientId": client_id}) as socket:
            async with session.post(f"{_url}/prompt", json={"prompt": graph, "client_id": client_id}) as response:
                response.raise_for_status()
                submitted = await response.json()
            if "error" in submitted:
                raise RuntimeError(submitted["error"])
            prompt_id = submitted["prompt_id"]
            context.active_prompt_id = prompt_id
            progress = 0
            reported_progress = -1
            last_report = 0
            try:
                while True:
                    now = time.monotonic()
                    if callback and (progress != reported_progress or now - last_report >= 2):
                        callback(None, progress)
                        reported_progress = progress
                        last_report = now
                    if context.stop_processing:
                        return []
                    try:
                        message = await socket.receive(timeout=2)
                    except asyncio.TimeoutError:
                        message = None
                    if message and message.type == aiohttp.WSMsgType.TEXT:
                        event = json.loads(message.data)
                        data = event.get("data") or {}
                        if data.get("prompt_id") == prompt_id:
                            if event.get("type") == "progress" and data.get("node") == "sampler":
                                progress = min(steps, int(data["value"]))
                            elif event.get("type") == "execution_error":
                                raise RuntimeError(data.get("exception_message", "ComfyUI execution failed"))
                    elif message and message.type == aiohttp.WSMsgType.BINARY and context.options.get("stream_image_progress"):
                        if struct.unpack(">I", message.data[:4])[0] == 1:
                            preview = Image.open(io.BytesIO(message.data[8:])).convert("RGB")
                            if callback:
                                callback([preview], progress)
                    elif message and message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        raise ConnectionError("ComfyUI progress connection closed")

                    async with session.get(f"{_url}/history/{prompt_id}") as response:
                        response.raise_for_status()
                        history = (await response.json()).get(prompt_id)
                    if not history:
                        continue
                    if history.get("status", {}).get("status_str") != "success":
                        raise RuntimeError(f"ComfyUI failed: {history.get('status')}")
                    images = []
                    for output in history["outputs"]["save"]["images"]:
                        async with session.get(f"{_url}/view", params=output) as response:
                            response.raise_for_status()
                            data = await response.read()
                        image = Image.open(io.BytesIO(data)).convert("RGB")
                        if output_type == "base64":
                            image = img_to_base64_str(
                                image, context.options.get("output_format", "jpeg"),
                                context.options.get("output_quality", 75), context.options.get("output_lossless", False),
                            )
                        images.append(image)
                    if callback:
                        callback(None, steps)
                    return images
            finally:
                context.active_prompt_id = None


def generate_images(context, callback=None, output_type="pil", **req):
    if any(req.get(key) for key in ("init_image", "init_image_mask", "ref_images", "control_image")):
        raise ValueError("Krea 2 currently supports text-to-image generation only")
    if req["num_inference_steps"] < 1 or req["width"] < 16 or req["height"] < 16:
        raise ValueError("Krea 2 requires positive steps and image dimensions of at least 16 pixels")
    context.stop_processing = False
    model = _model_name(context, "stable-diffusion")
    encoder = _model_name(context, "text-encoder", "qwen3vl_4b_fp8_scaled")
    vae = _model_name(context, "vae", "qwen_image_vae")
    lora_paths = context.model_paths.get("lora") or []
    lora_paths = lora_paths if isinstance(lora_paths, list) else [lora_paths]
    alphas = req.get("lora_alpha") or []
    alphas = alphas if isinstance(alphas, list) else [alphas]
    if lora_paths and len(lora_paths) != len(alphas):
        raise ValueError("LoRA models and strengths must have the same length")
    loras = []
    for path, alpha in zip(lora_paths, alphas):
        for directory in get_model_dirs("lora"):
            if os.path.commonpath((os.path.abspath(path), os.path.abspath(directory))) == os.path.abspath(directory):
                loras.append((os.path.relpath(path, directory).replace(os.sep, "/"), float(alpha)))
                break
        else:
            raise ValueError(f"LoRA is outside the configured folders: {path}")

    sampler_names = {
        "euler_a": "euler_ancestral", "dpm2": "dpm_2", "dpm2_a": "dpm_2_ancestral",
        "dpmpp_2s_a": "dpmpp_2s_ancestral", "heun_pp2": "heunpp2",
    }
    sampler = sampler_names.get(req.get("sampler_name"), req.get("sampler_name") or "euler")
    scheduler_names = {"automatic": "simple", "uniform": "sgm_uniform", "ddim": "ddim_uniform"}
    scheduler = scheduler_names.get(req.get("scheduler_name"), req.get("scheduler_name") or "simple")
    choices = _request("GET", "/object_info/KSampler").json()["KSampler"]["input"]["required"]
    for field, choice in (("sampler_name", sampler), ("scheduler", scheduler)):
        if choice not in choices[field][0]:
            raise ValueError(f"ComfyUI does not support Krea 2 {field}: {choice}")
    images = []
    for index in range(req.get("num_outputs", 1)):
        graph = _workflow(
            model, encoder, vae, req["prompt"], req.get("negative_prompt", ""), req["seed"] + index,
            req["width"], req["height"], req["num_inference_steps"], req["guidance_scale"],
            sampler, scheduler, loras,
        )
        images.extend(asyncio.run(_run_prompt(
            context, graph, callback, str(uuid.uuid4()), output_type, req["num_inference_steps"]
        )))
        if context.stop_processing:
            break
    return images


def stop_rendering(context):
    context.stop_processing = True
    prompt_id = context.active_prompt_id
    if prompt_id:
        queue = _request("GET", "/queue").json()
        _request("POST", "/queue", json={"delete": [prompt_id]})
        if any(item[1] == prompt_id for item in queue.get("queue_running", [])):
            _request("POST", "/interrupt", json={})


def filter_images(context, images, filters, filter_params=None, input_type="pil"):
    return sdkit_filter_images(context, images, filters, filter_params or {}, input_type)


def get_url():
    return _url


def refresh_models():
    pass


def list_controlnet_filters():
    return []
