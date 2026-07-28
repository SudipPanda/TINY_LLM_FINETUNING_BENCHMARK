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
    while torch.no_grad():
        for text in text:
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
    input_token = tokenizer(prompt , return_tensor = 'pt').to(device)
    with torch.no_grad():
        output_ids = model(
            **input_token , 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    
    full_text = tokenizer.decode(output_ids[0] , skip_special_tokens=True)
    return full_text.split("### SQL:")[-1].strip()

"""Evaluate Model here"""
def evaluate_model(model , tokenizer , test_row , device)->dict:
    samples = []
    valid_count = 0
    exact_match_count  = 0

    for text in test_row:
        predict = generate_sql(model , tokenizer , text["question"], text["context"], device)
        is_valid = _is_valid_sql(predict) #valid sql or not 
        is_exact = _normalize_sql(is_valid) == _normalize_sql(text['answer']) #mean exact match here
        
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
    toeknizer = AutoTokenizer(cfg.model.base_model_id)
    if toeknizer.pad_token is None:
        toeknizer.pad_token = toeknizer.eos_token
    
    test_row = list(load_dataset("json", data_files={"test": "data/processed/test.jsonl"})["test"])
    total_sample =cfg.evaluation.total_test_sample
    test_row = test_row[:total_sample]

    result = {}
    for model_label , model in [("base", cfg.model.base_model_id),
        ("fine_tuned", str(Path(cfg.project.output_dir) / "merged"))]:
        logger.info("evaluation % modle from %" , model_label , model)

        """Loading the modle here"""
        model = AutoModelForCausalLM.from_pretrained(model).to(device)
        text = [row[text] for row in test_row]

        perplexity = perplexity(model ,toeknizer , text , device)
        eval_metrics = evaluate_model(model , toeknizer , test_row ,device)

        result[model_label] = {
            "perplexity":perplexity , 
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
        lines.append(f"**Reference SQL:** `{b['reference_sql']}`\n")
        lines.append(f"**Base model output:** `{b['generated_sql']}` (valid={b['valid_sql']}, exact_match={b['exact_match']})\n")
        lines.append(f"**Fine-tuned model output:** `{f['generated_sql']}` (valid={f['valid_sql']}, exact_match={f['exact_match']})\n")
        lines.append("---\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Qualitative comparison written to %s", path)




