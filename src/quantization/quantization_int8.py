import sys
from pathlib import Path
from datasets import load_dataset
from src.utils.config import load_config
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
import json

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

def quantize_int8(cfg):
    src_dir = cfg.paths.merged_fp32_model_dir
    out_dir = ROOT / cfg.int8.output_dir
    out_dir.mkdir(parents = True , exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=cfg.int8.load_in_8bit,
        llm_int8_threshold=cfg.int8.llm_int8_threshold,
        llm_int8_has_fp16_weight=cfg.int8.llm_int8_has_fp16_weight,
    )

    logger.info("loading model in src model" , src_dir)
    model = AutoModelForCausalLM.from_pretrained(
        src_dir, quantization_config=bnb_config, device_map="auto")
    
    tokenizer = AutoTokenizer.from_pretrained(
        src_dir
    )
    
    sample_prompt = "### Instruction:\nWrite a SQL query.\n\n### SQL:\n"
    inputs = tokenizer(sample_prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    logger.info("INT8 sanity generation: %s", tokenizer.decode(output_ids[0], skip_special_tokens=True))

    tokenizer.save_pretrained(out_dir)

    with open(out_dir / "int8_quantization_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "source_checkpoint": str(src_dir),
            "quantization_config": bnb_config.to_dict(),
        }, f, indent=2)
    logger.info("INT8 manifest + tokenizer saved to %s", out_dir)

def main():
    cfg = load_config(ROOT / "TINY_LLM_FINETUNING_BENCHMARK/CONFIG/quantization_config.yaml")
    quantize_int8(cfg)

if __name__ == "__main__":
    main()





