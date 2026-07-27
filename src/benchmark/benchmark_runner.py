import csv
import json
import os
import json
from pathlib import Path
import psutil
import torch
import requests
from src.utils.config import load_config
import logging
import time


logger  =  logging.getLogger(__name__)

def _load_fix_prompt(cfg):
    prompt_path = cfg.benchmark.fixed_prompts_file
    if prompt_path.exists():
        return json.loads(prompt_path.read_text())
    
    return [
        "### Instruction:\nWrite a SQL query to list all employees earning more than 100000.\n\n### SQL:\n",
    ] * cfg.benchmark.num_timed_requests


def _run_time_request(base_url , prompt , warmup):
    result = []
    for i , prompt_val in enumerate(prompt):
        resp = requests.post(f"{base_url}/generate", json={"prompt": prompt}, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if i>=warmup:
            result.append(data)
    return result

"""How much our api can handle concurrent requests here"""
def _concurrent_request(base_url:str , prompt:str , concurrency:int ,n_request:int = 10):
    from concurrent.futures import ThreadPoolExecutor

    def _one_request():
        requests.post(f"{base_url}/generate", json={"prompt": prompt}, timeout=120)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_one_request) for _ in range(n_request)]
        for f in futures:
            f.result()
    elapsed = time.perf_counter() - start
    return n_request / elapsed if elapsed > 0 else 0.0


def benchmark_varient(cfg , varient , port , model_dir)->dict:
    base_url = f"http://127.0.0.1:{port}"

    health = requests.get(f"{base_url}/health", timeout=10).json()
    load_time_sec = health.get("load_time_sec" ,  float("nan"))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    process = psutil.Process()
    cpu_mem_before = process.memory_info().rss #actual ram occupied here before the inference here

    prompts = _load_fix_prompt(cfg)
    n_warmup = cfg.benchmark.num_warmup_requests #warmup reqiest mean first few requiest are not mesured

    """Main benchmark test here"""
    timed_result = _run_time_request(base_url , prompts , n_warmup)

    cpu_memeory_peak = max(process.memory_info().res , cpu_mem_before)/(1024*1024) #how much cpu peak memory
    gpu_mem_peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    )



    avg_ttft_ms = sum(r["ttft_ms"] for r in timed_result) / len(timed_result)
    avg_latency_ms = sum(r["total_time_ms"] for r in timed_result) / len(timed_result)
    avg_tokens_per_sec = sum(r["tokens_per_sec"] for r in timed_result) / len(timed_result)

    throughput_concurrency ={}

    for label in cfg.benchmark.concurrency_levels:
        throughput_concurrency[label] =  _concurrent_request(base_url, prompts[0], label)
    #model_size_mb =_dir_size_mb()

    result = {
          "varient":varient , 
          "peak_gpu_memory":gpu_mem_peak_mb , 
          "peak_cpu_memory":cpu_memeory_peak ,
          "ttft_ms":round(avg_ttft_ms , 2) ,
          "avg_latency_ms":avg_latency_ms ,
          "avg_token_sec":avg_tokens_per_sec,
          "throughput_req_per_sec_batch1": round(throughput_concurrency.get(1, 0.0), 3),
          "throughput_req_per_sec_batch4": round(throughput_concurrency.get(4, 0.0), 3),
          "throughput_req_per_sec_batch8": round(throughput_concurrency.get(8, 0.0), 3),
    }

    return result

def append_to_csv(result:dict , csv_path:str):
    path  = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with open(path , "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    logger.info("Appended result to %s", csv_path)


def main()->None:
    cfg = load_config("/workspaces/TINY_LLM_FINETUNING_BENCHMARK/CONFIG/deployment_config.yaml")
    result = benchmark_varient(cfg , "hf_transformers" , port , None)
    append_to_csv(result , cfg.benchmark.output_csv)

if __name__ == "__main__":
    main()









