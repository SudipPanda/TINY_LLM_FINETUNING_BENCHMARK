from datasets import load_dataset
from pathlib import Path
from src.utils.config import load_config
import logging
import json


logger = logging.getLogger(__name__)

def _load_calibration_data(cfg , tokenizer):
    n =cfg.gptq.calibration.num_calibration_samples
    max_len =cfg.gptq.calibration.max_calib_seq_len

    ds = load_dataset("json", data_files={"train": "data/processed/train.jsonl"})["train"]
    n = min(len(ds) , n)
    exemples = []

    for i in range(n):
                enc = tokenizer(ds[i]["text"], truncation=True, max_length=max_len, return_tensors="pt")
                exemples.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    
    return exemples

def quantization_gptq(cfg)->None:

    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "AutoGPTQ is not installed. Install with `pip install auto-gptq` "
            "(requires a CUDA-capable GPU)."
        ) from exc
    
    src_dir = cfg.paths.merged_fp32_model_dir
    out_dir = cfg.gptq.output_dir

    out_dir.mkdir(parents = True , exist_ok =True)
    quantize_config = BaseQuantizeConfig(
        bits=cfg.gptq.bits,
        group_size=cfg.gptq.group_size,
        desc_act=cfg.gptq.desc_act,
        damp_percent=cfg.gptq.damp_percent,
    )

    logger.info("Loading model from %s for GPTQ quantization", src_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(src_dir)
    model = AutoGPTQForCausalLM.from_pretrained(src_dir , quantize_config)

    calib_exemple = _load_calibration_data(cfg , tokenizer)
    logger.info(f"the length of the claibration dataset is ",len(calib_exemple))

    model.quantize(calib_exemple)
    
    model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(out_dir)

    with open(out_dir / "gptq_quantization_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "source_checkpoint": str(src_dir),
            "bits": cfg.gptq.bits,
            "group_size": cfg.gptq.group_size,
            "num_calibration_samples": len(calib_exemple),
        }, f, indent=2)
    logger.info("GPTQ-quantized model saved to %s", out_dir)


    logger.info("saving and training the model is done here")

def main():
    cfg = load_config("/workspaces/TINY_LLM_FINETUNING_BENCHMARK/CONFIG/training_config.yaml")
    quantization_gptq(cfg)

if __name__ == "__main__":
     main()


