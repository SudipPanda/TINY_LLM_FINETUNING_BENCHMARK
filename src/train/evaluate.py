import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils.config import load_config
import logging

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import math
import re
from tqdm import tqdm
import gc

logger = logging.getLogger(__name__)

try:
    import sqlglot

    def _is_valid_sql(sql)->bool:
        try:
            sqlglot.parse_one(sql)
            return True
        except Exception:
            return False

except Exception as e:
    logger.error("problem with importing the sqlglot here")

"""check for normalized sql key here"""
def _normalize_sql(sql: str) -> str:
    """Whitespace/keyword-case normalization for exact-match comparison."""
    sql = re.sub(r"\s+", " ", sql).strip().lower()
    return sql

"""Implement Perplexity here"""
def perplexity(model , tokenizer , text , device:str)->float:
    model.eval()
    loss = []
    with torch.no_grad():
        for text in tqdm(text, desc="Calculating Perplexity"):
            input = tokenizer(text ,return_tensors="pt", truncation=True, max_length=512).to(device)
            output = model(**input , labels=input["input_ids"])
            loss.append(output.loss.item())

    model_loss = sum(loss)/len(loss)
    return math.exp(model_loss)

"""Generate sql here"""
def generate_sql(model , tokenizer , context , question , device, max_new_tokens: int = 128):
     
    prompt = (
        "### Instruction:\nGiven the database schema, write a SQL query to answer the question.\n\n"
        f"### Schema:\n{context}\n\n### Question:\n{question}\n\n### SQL:\n"
     )
    input_token = tokenizer(prompt , return_tensors = 'pt').to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **input_token , 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        print(f"output_ids are {output_ids}")

    full_text = tokenizer.decode(output_ids , skip_special_tokens=True)[0]

    print(f"full_text are {full_text}")
    return full_text.split("### SQL:")[-1].strip()

"""Evaluate Model here"""
def evaluate_model(model , tokenizer , test_row , device)->dict:
    samples = []
    valid_count = 0
    exact_match_count  = 0

    for text in test_row:
        predict = generate_sql(model , tokenizer , text["question"], text["context"], device)
        is_valid = _is_valid_sql(predict) #valid sql or not 
        is_exact = _normalize_sql(predict) == _normalize_sql(text['answer']) #mean exact match here
        
        valid_count += int(is_valid)
        exact_match_count += int(is_exact)
        samples.append(
            {
                'question':text['question'] ,
                'context':text['context'] ,
                'valid_sql':valid_count , 
                'exact_match_count':exact_match_count,
                'predict_sql':predict ,
                'exact_sql_ans':text['answer']
            }
        )
    
    n = len(test_row)
    return {
        "sql_validity_rate": valid_count / n if n else 0.0,
        "normalized_exact_match_rate": exact_match_count / n if n else 0.0,
        "samples": samples,
    }

"""Running Comparision here"""
def run_comparision(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    toeknizer = AutoTokenizer.from_pretrained(cfg.model.base_model_id)
    if toeknizer.pad_token is None:
        toeknizer.pad_token = toeknizer.eos_token
    
    test_row = list(load_dataset("json", data_files={"test": "data/processed/test.jsonl"})["test"])
    total_sample =cfg.evaluation.num_qualitative_samples
    test_row = test_row[:total_sample]

    result = {}
    for model_label , model in [("base", cfg.model.base_model_id),
        ("fine_tuned", cfg.project.output_dir)]:

        logger.info("evaluation % modle from %" , model_label , model)

        """Loading the modle here"""
        model = AutoModelForCausalLM.from_pretrained(model).to(device)

        print(f"columens are {test_row[0]}")

        text_val = [row['text'] for row in test_row]

        perplexity_value = perplexity(model ,toeknizer , text_val, device)

        logger.info("Perplexity for %s model: %.4f", model_label, perplexity_value)

        eval_metrics = evaluate_model(model , toeknizer , test_row ,device)

        result[model_label] = {
            "perplexity":perplexity_value , 
            "sql_validity_rate": eval_metrics["sql_validity_rate"],
            "normalized_exact_match_rate": eval_metrics["normalized_exact_match_rate"],
            "samples": eval_metrics["samples"],
        }

        if device == 'cuda':
            torch.cuda.empty_cache()
    
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_the_output(result , out_dir / "qualitative_samples.md")

"""Write the output here"""
def _write_the_output(result , path:Path):
    lines = ["base mode vs fine tune model here"]
    base_sample = result['base']['samples']
    fine_tune_sample = result['fine_tuned']['samples']

    for i, (b, f) in enumerate(zip(base_sample, fine_tune_sample), start=1):
        lines.append(f"## Sample {i}\n")
        lines.append(f"**Question:** {b['question']}\n")
        lines.append(f"**Reference SQL:** `{b['exact_sql_ans']}`\n")
        lines.append(f"**Base model output:** `{b['predict_sql']}` (valid={b['valid_sql']}, exact_match={b['exact_match_count']})\n")
        lines.append(f"**Fine-tuned model output:** `{f['predict_sql']}` (valid={f['valid_sql']}, exact_match={f['exact_match_count']})\n")
        lines.append("---\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Qualitative comparison written to %s", path)

def main():
    config_path = ROOT / "CONFIG" / "training_config.yaml"
    cfg = load_config(config_path)
    run_comparision(cfg)
    del model 
    gc.collect()


if __name__ == "__main__":
    main()




