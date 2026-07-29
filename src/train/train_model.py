import sys
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import torch
import json
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config

"""Loading teh environment variable here"""
env_path = ROOT / ".env"  # <-- change this
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=False)
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ['HF_TOKEN'] = hf_token


logger =logging.getLogger(__name__)


"""Load the dataset split here"""
def load_process_split(data_dir: Path = ROOT / "data"/"processed"):
    files = {
        "train": str(data_dir / "train.jsonl"),
        "validation": str(data_dir / "valid.jsonl"),
    }
    return load_dataset("json", data_files=files)


"""build the model and tokenizer here"""
def build_model_tokenizer(cfg):
    model_cfg = cfg.model
    
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
      }
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.base_model_id ,
        dtype=dtype_map.get(model_cfg.torch_dtype , torch.float16),
        trust_remote_code=model_cfg.trust_remote_code,
        )
    return tokenizer , model


"""create the lora config here"""
def build_lora_config(cfg) -> LoraConfig:
    lora_cfg = cfg.lora
    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
        target_modules=list(lora_cfg.target_modules),
    )

"""create the sft config here"""

def build_sft_config(cfg) -> SFTConfig:
    t = cfg.training
    return SFTConfig(
        output_dir=cfg.project.output_dir,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_ratio=t.warmup_ratio,
        optim=t.optim,
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        bf16=t.bf16,
        fp16=t.fp16,
        gradient_checkpointing=t.gradient_checkpointing,
        logging_steps=t.logging_steps,
        eval_strategy=t.eval_strategy,
        eval_steps=t.eval_steps,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        report_to=t.report_to,
        max_length=cfg.dataset.max_seq_len,
        dataset_text_field="text",
        seed=cfg.project.seed,
    )


"""Train the model here"""
def train(cfg):
    accelerator = Accelerator()
    set_seed(cfg.project.seed)

    dataset = load_process_split()
    tokenizer , model = build_model_tokenizer(cfg)
    lora_config = build_lora_config(cfg)
    sft_config = build_sft_config(cfg)

    model = get_peft_model(model , lora_config)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    logger.info("starting the training process here....")
    train_result = trainer.train(resume_from_checkpoint= ROOT / "runs/qwen05b-sql-lora/checkpoint-200")

    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        trainer.save_model(cfg.project.output_dir)

    if accelerator.is_main_process:
        """saving all the result here"""
        metrics_path = Path(cfg.project.output_dir) / "train_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(train_result.metrics, f, indent=2)
        logger.info("Training complete. Metrics written to %s", metrics_path)


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Merge LoRA adapter into the base model for a deployable, quantization-ready checkpoint.
        merged_dir = Path(cfg.project.output_dir) / "merged"

        """merge and unload the model with peft parameter here"""
        del trainer
        torch.cuda.empty_cache()
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        logger.info("Merged fine-tuned model saved to %s", merged_dir)


"""The main function here"""
def main():
    config_path = ROOT / "CONFIG" / "training_config.yaml"
    cfg = load_config(config_path)
    train(cfg)

if __name__ == "__main__":
    main()