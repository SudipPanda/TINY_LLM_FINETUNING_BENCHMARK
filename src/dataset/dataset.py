from datasets import load_dataset , Dataset
import json
from src.utils.config import load_config
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
"""Try to check the sql is valid or not"""
try:
    import sqlglot

    def _is_valid(sql:str)->bool:
        try:
            sqlglot.parse_one(sql)
            return True

        except Exception:
            return False

except Exception as e:
    logger.warning("sqlglot is not imported properly here")

"""Load the dataset here"""

def _load_raw_dataset(cfg)->Dataset:
    cfg_dataset = cfg.dataset
    for dataset_id in [cfg_dataset.hf_dataset , cfg_dataset.fallback_df_dataset]:
        if not dataset_id:
            continue
        try:
            logger.info("loading the dataset here......")
            raw = load_dataset(dataset_id)
            return raw

        except Exception as e:
            logger.warning(f"failed to load the dataset are {dataset_id}")


"""NORMALIZED THE DATASET HERE"""
def _normalized_dataset(raw):
    col = set(col.column_names)
    if {"question", "context", "answer"}.issubset(cols):
        return raw
    
    
    rename_map = {
        "sql_prompt": "question",
        "sql_context": "context",
        "sql": "answer",}
    
    applicable = {k:v for k , v in rename_map.items() if k in col}
    if applicable:
        raw.rename_columns(applicable)
    
    missing = {"question", "context", "answer"} - set(raw.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns after normalization: {missing}")

    return raw


"""Filter the row with valid sql answer here"""
def _filter_valid_sql(raw:Dataset)->Dataset:
    mask = [_is_valid(row['answer']) for row in raw]
    kept = sum(mask)

    logger.info("total no of valid mask is " ,kept)
    return raw.select([i for i, ok in enumerate(mask) if ok])

"""APply chat template here for each row"""
def _apply_chat_template(row , chat_template):
    row['text'] = template.format(
        context=row["context"], question=row["question"], answer=row["answer"]
    )
    return row

"""Filter by length here"""
def _filter_by_length(raw , tokenizer , seq_len):
    def _ok(row):
        return len(tokenizer(row["text"], truncation=False)["input_ids"]) <= max_len

    kept = raw.filter(_ok)
    logger.info("Length filter (<=%d tokens): keeping %d / %d rows", max_len, len(kept), len(raw))
    return kept


"""PREPARING THE DATASET HERE"""
def prepare(cfg):
    ds_cfg = cfg.dataset
    raw = _load_raw_dataset(cfg)
    raw = _normalized_dataset(raw)
    raw = _dedup(raw, ds_cfg.dedup_on)

    if ds_cfg['min_sql_validity_check']:
        raw = _filter_valid_sql(raw)

    """applying the chat prompt template here"""
    raw = raw.map(lambda r : _apply_chat_template(r , ds_cfg.prompt_template))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_id)

    raw = _filter_by_length(raw , tokenizer , ds_cfg.max_seq_len)
    raw = raw.shuffle(seed = cfg.project.seed)

    n_train , n_test, n_valid = ds_cfg.train_size , ds_cfg.test_size, ds_cfg.val_size
    
    total_length = n_train+n_test+n_valid

    if total_length<len(raw):
        logger.warning("the main dataset length is way bigger than size needed here")

        """Rescaling the size of train and test and valid dataset here"""
        if ds_cfg.rescale_split:
            scale = len(raw)/total_length
            n_train, n_val, n_test = int(n_train * scale), int(n_val * scale), int(n_test * scale)

    train = raw.select(range(0, n_train))
    val = raw.select(range(n_train, n_train + n_val))
    test = raw.select(range(n_train + n_val, n_train + n_val + n_test))
    
    logger.info("Final split sizes -> train: %d, val: %d, test: %d", len(train), len(val), len(test))
    return {
        'train':train , 
        "test":test , 
        "valid":val
    }


"""SAVE THE SPLIT HERE"""
def save_splits(splits :dict[str , Dataset] , out_dir: str = "data/processed"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for split_name, ds in splits.items():
        file_path = out_path / f"{split_name}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps({"question": row["question"], "context": row["context"],
                                     "answer": row["answer"], "text": row["text"]}) + "\n")

        logger.info("Wrote %s (%d rows)", file_path, len(ds))


"""main file here"""

def main():
    cfg = load_config("/workspaces/TINY_LLM_FINETUNING_BENCHMARK/CONFIG/training_config.yaml")
    splits = prepare(splits)
    save_splits(splits)


if __name__ == "__main__":
    main()