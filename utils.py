import os

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def assert_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)[:30]} ...")
