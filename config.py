from dataclasses import dataclass

@dataclass(frozen=True)
class DataConfig:
    raw_csv_path: str = "data/raw/support2.csv"
    time_col: str = "d.time"          # SUPPORT2 follow-up time in days (common)
    event_col: str = "death"          # death indicator (often 0/1)
    treatment_col: str = "dnr"        # DNR as a proxy treatment/exposure if present
    id_col: str = "id"
    min_followup_days: float = 1.0
    test_size: float = 0.2
    random_state: int = 42

@dataclass(frozen=True)
class ModelConfig:
    penalizer: float = 0.01           # Cox regularization
    l1_ratio: float = 0.0
    horizon_days: int = 180           # ITE horizon
