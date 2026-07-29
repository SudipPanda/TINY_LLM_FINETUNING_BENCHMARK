import torch
from pathlib import Path
import logging
import json
from datasets import load_dataset
import sys
from sql_metric import is_valid_sql , normalize_exact_match
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from tqdm import tqdm
import csv
import gc

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config 


PROMPT_TEMPLATE = (
    "### Instruction:\nGiven the database schema, write a SQL query to answer the question.\n\n"
    "### Schema:\n{context}\n\n### Question:\n{question}\n\n### SQL:\n"
)

def _resolve_dtype(dtype):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    return mapping[dtype]


def _load_stage_model(stage_cfg):
    kind = stage_cfg["kind"]
    path = stage_cfg["path"]

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if kind == "hf_transformers":
        dtype = _resolve_dtype(stage_cfg.get("dtype", "float32"))
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    elif kind == "bitsandbytes":
        from transformers import BitsAndBytesConfig
        manifest_path = Path(path) / "int8_quantization_manifest.json"
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
        raise ValueError(f"Unknown stage kind: {kind}")

    return model, tokenizer



"""Generate SQL query from the model given the question and context"""
def _generate_sql_query(model, tokenizer, question, context, max_new_tokens):
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_text.split("### SQL:")[-1].strip()  # Extract the SQL part from the output


"""load the evaluation dataset"""
def _load_evaluation_dataset(cfg):
    dataset_path = ROOT / cfg.eval.test_split_path

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")
    
    ds = load_dataset("json", data_files={"test": str(dataset_path)})['test']
    return ds

"""Evaluation Dataset Here"""
def _evaluation_dataset(stage_name: str, stage_cfg, cfg):

    logger.info("=== Evaluating stage: %s ===", stage_name)
    model , tokenizer = _load_stage_model(stage_cfg)

    ds = _load_evaluation_dataset(cfg)
    logger.info(f"Loaded evaluation dataset with {len(ds)} samples.")
    
    per_sample_results = []

    for i , sample in tqdm(enumerate(ds), total=len(ds)):
        question = sample["question"]
        context = sample["context"]
        sql_query = sample["answer"]

        # Generate SQL query using the model
        generated_sql = _generate_sql_query(model, tokenizer, question, context, cfg.eval.max_new_tokens)
        valid = is_valid_sql(generated_sql)
        exact_match = normalize_exact_match(generated_sql, sql_query)

        # Store the generated SQL query in the dataset
        per_sample_results.append({
            "question": question,
            "context": context,
            "reference_sql": sql_query,
            "generated_sql": generated_sql,
            "is_valid": valid,
            "exact_match": exact_match
            })
    
    """clean up the memory here"""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n = len(per_sample_results)
   

    return per_sample_results




def run_benchmark(cfg) -> None:

    all_results = {}
    for stage_name in cfg.pipeline_order:
        stage_cfg = cfg.stages[stage_name]
        try:
            all_results[stage_name] = _evaluation_dataset(stage_name, stage_cfg, cfg)
        except Exception as exc:  # noqa: BLE001 - one missing/broken stage shouldn't kill the whole sweep
            logger.error("Skipping stage '%s' due to error: %s", stage_name, exc)
            continue

    #_write_outputs(cfg, all_results)

"""Write the outputs to JSON and CSV files"""
def _write_outputs(cfg, all_results: dict) -> None:
    json_path = Path(cfg.output.results_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Full per-sample results written to %s", json_path)

    csv_path = Path(cfg.output.results_csv)
    aggregates = [r["aggregate"] for r in all_results.values()]
    if aggregates:
        fieldnames = list(aggregates[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in aggregates:
                writer.writerow(row)
        logger.info("Aggregate results written to %s", csv_path)


if __name__ == "__main__":
    config_path = ROOT / "CONFIG" / "evaluation_config.yaml"
    cfg = load_config(config_path)
    run_benchmark(cfg)
    