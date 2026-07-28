import sys
from pathlib import Path
from datasets import load_dataset
from src.utils.config import load_config
import logging
import json

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

def __load_awq_dataset__(src):
    n = src.awq.calibration.num_calibration_samples
    ds = load_dataset("json", data_files={"train": "data/processed/train.jsonl"})["train"]

    n = min(n , len(ds))
    return [ds[i]['text'] for i in range(n)]

def quantization_awq(cfg):
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "AutoAWQ is not installed. Install with `pip install autoawq` "
            "(requires a CUDA-capable GPU)."
        ) from exc
    
    src_dir = cfg.paths.merged_fp32_model_dir
    out_dir = cfg.awq.output_dir
    out_dir.mkdir(parents = True , exist_ok=True)

    quant_config = {
        "zero_point": cfg.awq.zero_point,
        "q_group_size": cfg.awq.q_group_size,
        "w_bit": cfg.awq.w_bit,
        "version": "GEMM",
    }

    logger.info("Loading model from %s for AWQ quantization", src_dir)
    model = AutoAWQForCausalLM.from_pretrained(src_dir)
    tokenizer = AutoTokenizer.from_pretrained(src_dir)

    calib_texts = __load_awq_dataset__(cfg)
    logger.info("Running AWQ calibration on %d samples", len(calib_texts))
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)

    model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(out_dir)

    with open(out_dir / "awq_quantization_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"source_checkpoint": str(src_dir), "quant_config": quant_config,
                    "num_calibration_samples": len(calib_texts)}, f, indent=2)
    logger.info("AWQ-quantized model saved to %s", out_dir)


def main():
    cfg = load_config("/workspaces/TINY_LLM_FINETUNING_BENCHMARK/CONFIG/training_config.yaml")
    quantization_awq(cfg)

if __name__ == "__main__":
    main()




 