from pathlib import Path
from typing import Any , Dict
import yaml

class Config(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

def load_config(path:str|Path)->Config:
    """Load the yaml file here"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found in {path}")
    
    with open(path , "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    
    return Config(raw)
