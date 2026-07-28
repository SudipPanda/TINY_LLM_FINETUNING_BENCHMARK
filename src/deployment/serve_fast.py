import sys
from pathlib import Path
import torch
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.config import load_config
import logging
from contextlib import asynccontextmanager
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



logger = logging.getLogger(__name__)
_state:dict = {}

CONFIG_PATH = os.environ.get("DEPLOYMENT_CONFIG", "configs/deployment_config.yaml")
VARIANT = os.environ.get("MODEL_VARIANT", "base_fp32")

def _load_varient(cfg , varient_name):
    varients = cfg.variants
    if varient_name not in varients:
        raise ValueError(f"Unknown MODEL_VARIANT '{varient_name}'. Options: {list(varients.keys())}")
    
    varient_cfg = varients[varient_name]
    kind = varient_cfg["kind"]
    path = varient_cfg["path"]

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if kind == "hf_transformers":
        dtype = {"float32": torch.float32, "float16": torch.float16}.get(
            varient_cfg.get("dtype", "float32"), torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    

    elif kind == "bitsandbytes":
        from transformers import BitsAndBytesConfig
        manifest_path = os.path.join(path, "int8_quantization_manifest.json")
        import json
        with open(manifest_path) as f:
            manifest = json.load(f)
        bnb_config = BitsAndBytesConfig(**manifest["quantization_config"])
        model = AutoModelForCausalLM.from_pretrained(
            manifest["source_checkpoint"], quantization_config=bnb_config, device_map="auto"
        )


    elif kind == "autoawq":
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(path, fuse_layers=False)


    elif kind == "autogptq":
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(path, device="cuda" if torch.cuda.is_available() else "cpu")
    
    else:
        raise ValueError("unknlwn varient {kind}")
    
    return model , tokenizer


@asynccontextmanager
async def lifespam(app:FastAPI):
    cfg = load_config("/workspaces/TINY_LLM_FINETUNING_BENCHMARK/CONFIG/training_config.yaml")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    model , tokenizer = _load_varient(cfg , VARIANT)

    load_time = time.perf_counter()-load_start
    logger.info("Variant '%s' loaded in %.2fs", VARIANT, load_time)

    _state["cfg"] = cfg
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["variant"] = VARIANT
    _state["load_time"] = load_time
    yield
    _state.clear()
    

app = FastAPI(title="TinyForge Inference Server", lifespan=lifespam)


"""Model Rewuest and Response format here"""

class GenerateRequest(BaseModel):
    prompt : str
    max_new_token : int | None = None

class GenerateResponse(BaseModel):
    variant: str
    generated_text: str
    ttft_ms: float
    total_time_ms: float
    tokens_generated: int
    tokens_per_sec: float



"""Fast API endpoint creation here"""

@app.get("/health")
def health():
    return {"status": "ok", "variant": _state.get("variant"), "load_time_sec": _state.get("load_time")}


@app.post("/generate", response_model=GenerateResponse)
def generate(req:GenerateRequest):
    model = _state.get("model")
    tokenizer = _state.get("tokenizer")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    max_new_token = None
    device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"

    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        first_token_out =  model.generate(
            **inputs, max_new_tokens=1, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    time = time.perf_counter()-start_time


    with torch.no_grad():
        full_out = model.generate(
            **inputs, max_new_tokens=max_new_token, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    total_time = time.perf_counter() - start_time


    generate_text = tokenizer.decode(full_out[0])
    
    ########################
    tokens_generated = full_out.shape[-1] - inputs["input_ids"].shape[-1]
    decode_time = max(total_time - time, 1e-6)
    tokens_per_sec = tokens_generated / decode_time if tokens_generated > 0 else 0.0

    return GenerateResponse(
        variant=_state["variant"],
        generated_text=generate_text,
        ttft_ms=time * 1000,
        total_time_ms=total_time * 1000,
        tokens_generated=tokens_generated,
        tokens_per_sec=tokens_per_sec,
    )












