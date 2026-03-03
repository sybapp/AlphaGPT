import torch
import os


class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DB_URL = f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','password')}@{os.getenv('DB_HOST','localhost')}:5432/{os.getenv('DB_NAME','crypto_quant')}"
    BATCH_SIZE = 8192
    TRAIN_STEPS = 1000
    MAX_FORMULA_LEN = 12
    TRADE_SIZE_USD = 1000.0
    MIN_LIQUIDITY = 5000.0 # 低于此流动性视为归零/无法交易
    BASE_FEE = 0.005 # 基础费率 0.5% (Swap + Gas + Jito Tip)
    TRAIN_SPLIT_RATIO = float(os.getenv('TRAIN_SPLIT_RATIO', 0.8))

    FACTOR_MODE = os.getenv("FACTOR_MODE", "basic").strip().lower()
    FACTOR_DIMS = {
        "basic": 6,
        "advanced": 12,
        "albrooks": 10,
        "ictsmc": 12,
    }
    INPUT_DIM = FACTOR_DIMS.get(FACTOR_MODE, FACTOR_DIMS["basic"])
