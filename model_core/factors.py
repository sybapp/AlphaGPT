import torch
import torch.nn as nn


class RMSNormFactor(nn.Module):
    """RMSNorm for factor normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class MemeIndicators:
    @staticmethod
    def liquidity_health(liquidity, fdv):
        ratio = liquidity / (fdv + 1e-6)
        return torch.clamp(ratio * 4.0, 0.0, 1.0)

    @staticmethod
    def buy_sell_imbalance(close, open_, high, low):
        range_hl = high - low + 1e-9
        body = close - open_
        strength = body / range_hl
        return torch.tanh(strength * 3.0)

    @staticmethod
    def fomo_acceleration(volume, window=5):
        vol_prev = torch.roll(volume, 1, dims=1)
        vol_chg = (volume - vol_prev) / (vol_prev + 1.0)
        acc = vol_chg - torch.roll(vol_chg, 1, dims=1)
        return torch.clamp(acc, -5.0, 5.0)

    @staticmethod
    def pump_deviation(close, window=20):
        pad = torch.zeros((close.shape[0], window-1), device=close.device)
        c_pad = torch.cat([pad, close], dim=1)
        ma = c_pad.unfold(1, window, 1).mean(dim=-1)
        dev = (close - ma) / (ma + 1e-9)
        return dev

    @staticmethod
    def volatility_clustering(close, window=10):
        """Detect volatility clustering patterns"""
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        ret_sq = ret ** 2
        
        pad = torch.zeros((ret_sq.shape[0], window-1), device=close.device)
        ret_sq_pad = torch.cat([pad, ret_sq], dim=1)
        vol_ma = ret_sq_pad.unfold(1, window, 1).mean(dim=-1)
        
        return torch.sqrt(vol_ma + 1e-9)

    @staticmethod
    def momentum_reversal(close, window=5):
        """Capture momentum reversal signals"""
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        
        pad = torch.zeros((ret.shape[0], window-1), device=close.device)
        ret_pad = torch.cat([pad, ret], dim=1)
        mom = ret_pad.unfold(1, window, 1).sum(dim=-1)
        
        # Detect reversals
        mom_prev = torch.roll(mom, 1, dims=1)
        reversal = (mom * mom_prev < 0).float()
        
        return reversal

    @staticmethod
    def relative_strength(close, high, low, window=14):
        """RSI-like indicator for strength detection"""
        ret = close - torch.roll(close, 1, dims=1)
        
        gains = torch.relu(ret)
        losses = torch.relu(-ret)
        
        pad = torch.zeros((gains.shape[0], window-1), device=close.device)
        gains_pad = torch.cat([pad, gains], dim=1)
        losses_pad = torch.cat([pad, losses], dim=1)
        
        avg_gain = gains_pad.unfold(1, window, 1).mean(dim=-1)
        avg_loss = losses_pad.unfold(1, window, 1).mean(dim=-1)
        
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        return (rsi - 50) / 50  # Normalize


class AdvancedFactorEngineer:
    """Advanced feature engineering with multiple factor types"""
    def __init__(self):
        self.rms_norm = RMSNormFactor(1)
    
    def robust_norm(self, t):
        """Robust normalization using median absolute deviation"""
        median = torch.nanmedian(t, dim=1, keepdim=True)[0]
        mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
        norm = (t - median) / mad
        return torch.clamp(norm, -5.0, 5.0)
    
    def compute_advanced_features(self, raw_dict):
        """Compute 12-dimensional feature space with advanced factors"""
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        liq = raw_dict['liquidity']
        fdv = raw_dict['fdv']
        
        # Basic factors
        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))
        liq_score = MemeIndicators.liquidity_health(liq, fdv)
        pressure = MemeIndicators.buy_sell_imbalance(c, o, h, l)
        fomo = MemeIndicators.fomo_acceleration(v)
        dev = MemeIndicators.pump_deviation(c)
        log_vol = torch.log1p(v)
        
        # Advanced factors
        vol_cluster = MemeIndicators.volatility_clustering(c)
        momentum_rev = MemeIndicators.momentum_reversal(c)
        rel_strength = MemeIndicators.relative_strength(c, h, l)
        
        # High-low range
        hl_range = (h - l) / (c + 1e-9)
        
        # Close position in range
        close_pos = (c - l) / (h - l + 1e-9)
        
        # Volume trend
        vol_prev = torch.roll(v, 1, dims=1)
        vol_trend = (v - vol_prev) / (vol_prev + 1.0)
        
        features = torch.stack([
            self.robust_norm(ret),
            liq_score,
            pressure,
            self.robust_norm(fomo),
            self.robust_norm(dev),
            self.robust_norm(log_vol),
            self.robust_norm(vol_cluster),
            momentum_rev,
            self.robust_norm(rel_strength),
            self.robust_norm(hl_range),
            close_pos,
            self.robust_norm(vol_trend)
        ], dim=1)
        
        return features


class AlBrooksFactorEngineer:
    """Al Brooks inspired price-action factor engineering"""
    INPUT_DIM = 10

    @staticmethod
    def robust_norm(t):
        median = torch.nanmedian(t, dim=1, keepdim=True)[0]
        mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
        norm = (t - median) / mad
        return torch.clamp(norm, -5.0, 5.0)

    @staticmethod
    def _rolling_prev_window(x, window):
        pad = x[:, :1].repeat(1, window)
        x_prev = torch.cat([pad, x[:, :-1]], dim=1)
        return x_prev.unfold(1, window, 1)

    @staticmethod
    def _rolling_window(x, window):
        pad = x[:, :1].repeat(1, window - 1)
        x_pad = torch.cat([pad, x], dim=1)
        return x_pad.unfold(1, window, 1)

    @staticmethod
    def _ema(x, span):
        alpha = 2.0 / (span + 1.0)
        ema = torch.zeros_like(x)
        ema[:, 0] = x[:, 0]
        for t in range(1, x.shape[1]):
            ema[:, t] = alpha * x[:, t] + (1.0 - alpha) * ema[:, t - 1]
        return ema

    @classmethod
    def compute_features(cls, raw_dict):
        eps = 1e-9

        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']

        prev5_low = cls._rolling_prev_window(l, 5).min(dim=-1).values
        prev5_high = cls._rolling_prev_window(h, 5).max(dim=-1).values
        pb = (c - prev5_low) / (prev5_high - prev5_low + eps)

        prev_close = torch.roll(c, 1, dims=1)
        prev_close[:, 0] = c[:, 0]
        ret = torch.log(c / (prev_close + eps))

        tc = torch.sign(cls._ema(ret, 3) * cls._ema(ret, 8))

        max_high_prev20 = cls._rolling_prev_window(h, 20).max(dim=-1).values
        bo = (c - max_high_prev20) / (max_high_prev20 + eps)
        bo = torch.clamp(bo, -5.0, 5.0)

        bar_range = h - l + eps
        rb = (c - o) / bar_range

        upper_wick = h - torch.maximum(o, c)
        lower_wick = torch.minimum(o, c) - l
        wb = (upper_wick - lower_wick) / bar_range

        prev_high = torch.roll(h, 1, dims=1)
        prev_low = torch.roll(l, 1, dims=1)
        prev_high[:, 0] = h[:, 0]
        prev_low[:, 0] = l[:, 0]
        ib = ((h <= prev_high) & (l >= prev_low)).float()

        ret_window10 = cls._rolling_window(ret, 10)
        ret_std10 = ret_window10.std(dim=-1)
        ema_ret10 = cls._ema(ret, 10)
        rv = ret_std10 / (torch.abs(ema_ret10) + 1e-4)

        vol_prev10 = cls._rolling_prev_window(v, 10)
        vol_median10 = torch.median(vol_prev10, dim=-1).values
        vol_mad10 = torch.median(torch.abs(vol_prev10 - vol_median10.unsqueeze(-1)), dim=-1).values
        vo = (v - vol_median10) / (vol_mad10 + 1e-6)
        vo = torch.clamp(vo, -5.0, 5.0)

        fd = ((bo > 0) & (c < o) & (c < max_high_prev20 * 0.995)).float()

        tr = torch.maximum(h - l, torch.maximum(torch.abs(h - prev_close), torch.abs(l - prev_close)))
        atr20 = cls._rolling_window(tr, 20).mean(dim=-1)
        ema20 = cls._ema(c, 20)
        cb = (c - ema20) / (atr20 + 1e-6)

        features = torch.stack([
            cls.robust_norm(pb),
            tc,
            cls.robust_norm(bo),
            rb,
            wb,
            ib,
            cls.robust_norm(rv),
            vo,
            fd,
            cls.robust_norm(cb)
        ], dim=1)

        return torch.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)



class FeatureEngineer:
    INPUT_DIM = 6

    @staticmethod
    def compute_features(raw_dict):
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        liq = raw_dict['liquidity']
        fdv = raw_dict['fdv']

        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))
        liq_score = MemeIndicators.liquidity_health(liq, fdv)
        pressure = MemeIndicators.buy_sell_imbalance(c, o, h, l)
        fomo = MemeIndicators.fomo_acceleration(v)
        dev = MemeIndicators.pump_deviation(c)
        log_vol = torch.log1p(v)

        def robust_norm(t):
            median = torch.nanmedian(t, dim=1, keepdim=True)[0]
            mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
            norm = (t - median) / mad
            return torch.clamp(norm, -5.0, 5.0)

        features = torch.stack([
            robust_norm(ret),
            liq_score,
            pressure,
            robust_norm(fomo),
            robust_norm(dev),
            robust_norm(log_vol)
        ], dim=1)

        return features
