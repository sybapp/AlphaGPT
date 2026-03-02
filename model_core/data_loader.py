import pandas as pd
import torch
import sqlalchemy

from .config import ModelConfig
from .factors import FeatureEngineer, AdvancedFactorEngineer, AlBrooksFactorEngineer, ICTSMCFactorEngineer


class CryptoDataLoader:
    def __init__(self, factor_mode=None):
        self.engine = sqlalchemy.create_engine(ModelConfig.DB_URL)
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None

        self.factor_mode = (factor_mode or ModelConfig.FACTOR_MODE).strip().lower()
        self._advanced_engineer = AdvancedFactorEngineer()

    def _compute_features(self, raw_data_cache):
        if self.factor_mode == "advanced":
            return self._advanced_engineer.compute_advanced_features(raw_data_cache)
        if self.factor_mode == "albrooks":
            return AlBrooksFactorEngineer.compute_features(raw_data_cache)
        if self.factor_mode == "ictsmc":
            return ICTSMCFactorEngineer.compute_features(raw_data_cache)
        return FeatureEngineer.compute_features(raw_data_cache)

    def load_data(self, limit_tokens=500):
        print("Loading data from SQL...")
        top_query = f"""
        SELECT address FROM tokens
        LIMIT {limit_tokens}
        """
        addrs = pd.read_sql(top_query, self.engine)['address'].tolist()
        if not addrs:
            raise ValueError("No tokens found.")
        addr_str = "'" + "','".join(addrs) + "'"
        data_query = f"""
        SELECT time, address, open, high, low, close, volume, liquidity, fdv
        FROM ohlcv
        WHERE address IN ({addr_str})
        ORDER BY time ASC
        """
        df = pd.read_sql(data_query, self.engine)

        def to_tensor(col):
            pivot = df.pivot(index='time', columns='address', values=col)
            pivot = pivot.fillna(method='ffill').fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'volume': to_tensor('volume'),
            'liquidity': to_tensor('liquidity'),
            'fdv': to_tensor('fdv')
        }
        self.feat_tensor = self._compute_features(self.raw_data_cache)

        op = self.raw_data_cache['open']
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        self.target_ret = torch.log(t2 / (t1 + 1e-9))
        self.target_ret[:, -2:] = 0.0

        if torch.isnan(self.feat_tensor).any() or torch.isinf(self.feat_tensor).any():
            self.feat_tensor = torch.nan_to_num(self.feat_tensor, nan=0.0, posinf=5.0, neginf=-5.0)

        print(f"Data Ready. Mode: {self.factor_mode} | Shape: {self.feat_tensor.shape}")
