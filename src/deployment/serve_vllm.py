
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
import sys
from src.utils.config import load_config
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def _build_engine(cfg , varient):
    try:
        from vllm import LLM
    except ImportError:
        raise ImportError("vLLM is not installed. Please install vLLM to use this function.")
    
    varients = cfg.varients
    if varient not in varients:
        raise ValueError(f"Varient {varient} not found in config. Available varients: {list(varients.keys())}")
    
    varient_cfg = varients[varient]
    engine_cfg = cfg.engine

    llm_kwargs = dict(
        model=variant_cfg["path"],
        gpu_memory_utilization=engine_cfg.gpu_memory_utilization,
        max_model_len=engine_cfg.max_model_len,
        tensor_parallel_size=engine_cfg.tensor_parallel_size,
        enforce_eager=engine_cfg.get("enforce_eager", False),
        swap_space=engine_cfg.get("swap_space_gb", 4),
        dtype=variant_cfg.get("dtype", "auto"),
    )

    if variant_cfg.get("quantization"):
        llm_kwargs["quantization"] = variant_cfg["quantization"]
    if variant_cfg.get("load_format"):
        llm_kwargs["load_format"] = variant_cfg["load_format"]

    logger.info("Initializing vLLM engine for variant '%s' with kwargs: %s", variant_name, llm_kwargs)

    engine = LLM(**llm_kwargs)
    return engine


@asynccontextmanager
async def lifespam(app:FastAPI):
    cofig_path = ROOT / "CONFIG" / "deployment_config.yaml"
    cfg = load_config(cofig_path)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    engine = _build_engine(cfg, VARIANT)

    load_time = time.perf_counter()-load_start
    logger.info("Variant '%s' loaded in %.2fs", VARIANT, load_time)

    _state["cfg"] = cfg
    _state["engine"] = engine
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
    engine = _state.get("engine")
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded yet")
    
    from vllm import SamplingParams
    cfg = _state.get("cfg")

    max_tokens = req.max_new_token if req.max_new_token is not None else cfg.engine.get("max_new_tokens_default", 128)

    sampling_params = SamplingParams(
        temperature=cfg.sampling_defaults.temperature,
        top_p=cfg.sampling_defaults.top_p,
        max_tokens=max_new_tokens,
    )
    
    start_time = time.perf_counter()

    output = engine.generate(req.prompt , sampling_params=sampling_params) ## NOtice this line here

    total_time = time.perf_counter()-start_time
    output = output[0]  # Assuming single request, get the first output
    generated_text = output.outputs[0].text
    tokens_generated = len(output.outputs[0].token_ids)
    
    """ NOtice that we are returning the response in the format of GenerateResponse model"""
    return GenerateResponse(
        variant=_state.get("variant"),
        generated_text=generated_text,
        ttft_ms=output.time_to_first_token * 1000,
        total_time_ms=total_time * 1000,
        tokens_generated=tokens_generated,
        tokens_per_sec=tokens_generated / total_time if total_time > 0 else 0
    )



# def main():
#     cofig_path = ROOT / "CONFIG" / "deployment_config.yaml"
#     cfg = load_config(cofig_path)
