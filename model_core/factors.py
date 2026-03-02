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


class ICTSMCFactorEngineer:
    """ICT/SMC inspired tensor factor engineering (12D)."""
    INPUT_DIM = 12

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

    @classmethod
    def compute_features(cls, raw_dict):
        eps = 1e-9

        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']

        # Core volatility proxies
        prev_close = torch.roll(c, 1, dims=1)
        prev_close[:, 0] = c[:, 0]
        tr = torch.maximum(h - l, torch.maximum(torch.abs(h - prev_close), torch.abs(l - prev_close)))
        atr14 = cls._rolling_window(tr, 14).mean(dim=-1)

        # 1) LS: liquidity sweep strength (-1 sweep-high, +1 sweep-low)
        prev10_high = cls._rolling_prev_window(h, 10).max(dim=-1).values
        prev10_low = cls._rolling_prev_window(l, 10).min(dim=-1).values
        sweep_high = ((h > prev10_high) & (c < prev10_high)).float()
        sweep_low = ((l < prev10_low) & (c > prev10_low)).float()
        ls = sweep_low - sweep_high

        # 2) BOS: signed structure break (+1 up, -1 down)
        swing_k = 20
        swing_high_prev = cls._rolling_prev_window(h, swing_k).max(dim=-1).values
        swing_low_prev = cls._rolling_prev_window(l, swing_k).min(dim=-1).values
        bos_up = (c > swing_high_prev).float()
        bos_dn = (c < swing_low_prev).float()
        bos = bos_up - bos_dn

        # 3) CHOCH: signed change-of-character proxy
        prev_bos = torch.roll(bos, 1, dims=1)
        prev_bos[:, 0] = 0.0
        choch = torch.where((prev_bos * bos) < 0.0, bos, torch.zeros_like(bos))

        # 4) FVG_GAP: signed fair-value-gap amplitude
        high_t2 = torch.roll(h, 2, dims=1)
        low_t2 = torch.roll(l, 2, dims=1)
        high_t2[:, :2] = h[:, :2]
        low_t2[:, :2] = l[:, :2]
        bull_gap = torch.relu(l - high_t2)
        bear_gap = torch.relu(low_t2 - h)
        fvg_gap = torch.clamp((bull_gap - bear_gap) / (atr14 + eps), -5.0, 5.0)

        # 5) FVG_FILL: fill ratio against latest active FVG zone [0, 1]
        n_tokens, n_time = c.shape
        fvg_fill = torch.zeros_like(c)
        last_fvg_low = torch.zeros(n_tokens, device=c.device)
        last_fvg_high = torch.zeros(n_tokens, device=c.device)
        last_fvg_dir = torch.zeros(n_tokens, device=c.device)

        for t in range(n_time):
            bull_mask = bull_gap[:, t] > 0
            bear_mask = bear_gap[:, t] > 0

            if bull_mask.any():
                last_fvg_low[bull_mask] = high_t2[bull_mask, t]
                last_fvg_high[bull_mask] = l[bull_mask, t]
                last_fvg_dir[bull_mask] = 1.0
            if bear_mask.any():
                last_fvg_low[bear_mask] = h[bear_mask, t]
                last_fvg_high[bear_mask] = low_t2[bear_mask, t]
                last_fvg_dir[bear_mask] = -1.0

            zone_w = torch.clamp(last_fvg_high - last_fvg_low, min=eps)
            bull_fill = torch.clamp((last_fvg_high - c[:, t]) / zone_w, 0.0, 1.0)
            bear_fill = torch.clamp((c[:, t] - last_fvg_low) / zone_w, 0.0, 1.0)
            fvg_fill[:, t] = torch.where(
                last_fvg_dir > 0,
                bull_fill,
                torch.where(last_fvg_dir < 0, bear_fill, torch.zeros_like(bull_fill))
            )

        # 6) OB_PROX + 7) BRK_OB: order-block proxy and breaker proxy
        body = c - o
        body_abs = torch.abs(body)
        vol_z = cls.robust_norm(torch.log1p(v))
        disp_raw = body_abs / (atr14 + eps)
        disp_event = disp_raw > 1.5

        prev_o = torch.roll(o, 1, dims=1)
        prev_c = torch.roll(c, 1, dims=1)
        prev_o[:, 0] = o[:, 0]
        prev_c[:, 0] = c[:, 0]

        prev_body_low = torch.minimum(prev_o, prev_c)
        prev_body_high = torch.maximum(prev_o, prev_c)

        ob_prox = torch.zeros_like(c)
        brk_ob = torch.zeros_like(c)

        last_ob_low = torch.zeros(n_tokens, device=c.device)
        last_ob_high = torch.zeros(n_tokens, device=c.device)
        last_ob_dir = torch.zeros(n_tokens, device=c.device)

        for t in range(n_time):
            bull_disp = disp_event[:, t] & (body[:, t] > 0)
            bear_disp = disp_event[:, t] & (body[:, t] < 0)

            if bull_disp.any():
                last_ob_low[bull_disp] = prev_body_low[bull_disp, t]
                last_ob_high[bull_disp] = prev_body_high[bull_disp, t]
                last_ob_dir[bull_disp] = 1.0
            if bear_disp.any():
                last_ob_low[bear_disp] = prev_body_low[bear_disp, t]
                last_ob_high[bear_disp] = prev_body_high[bear_disp, t]
                last_ob_dir[bear_disp] = -1.0

            ob_mid = (last_ob_low + last_ob_high) / 2.0
            ob_half = torch.clamp((last_ob_high - last_ob_low) / 2.0, min=eps)
            ob_dist = torch.abs(c[:, t] - ob_mid) / (ob_half + atr14[:, t] + eps)
            ob_prox[:, t] = torch.clamp(1.0 - ob_dist, 0.0, 1.0)

            in_zone = (c[:, t] >= last_ob_low) & (c[:, t] <= last_ob_high)
            invalid_bull = (last_ob_dir > 0) & (c[:, t] < last_ob_low)
            invalid_bear = (last_ob_dir < 0) & (c[:, t] > last_ob_high)
            retest_bull = (last_ob_dir > 0) & in_zone & (c[:, t] < ob_mid)
            retest_bear = (last_ob_dir < 0) & in_zone & (c[:, t] > ob_mid)
            brk_ob[:, t] = torch.where(
                invalid_bull | retest_bull,
                -torch.ones_like(c[:, t]),
                torch.where(invalid_bear | retest_bear, torch.ones_like(c[:, t]), torch.zeros_like(c[:, t]))
            )

        # 8) DISP: displacement strength (bounded)
        disp = torch.clamp(disp_raw * (1.0 + torch.relu(vol_z)), 0.0, 5.0)

        # 9) EQHL: equal high/low liquidity pool proximity (signed)
        pool_m = 8
        prev_high_m = cls._rolling_prev_window(h, pool_m)
        prev_low_m = cls._rolling_prev_window(l, pool_m)
        high_spread = prev_high_m.max(dim=-1).values - prev_high_m.min(dim=-1).values
        low_spread = prev_low_m.max(dim=-1).values - prev_low_m.min(dim=-1).values
        eq_high = (high_spread / (atr14 + eps) < 0.4).float()
        eq_low = (low_spread / (atr14 + eps) < 0.4).float()

        high_pool = prev_high_m.mean(dim=-1)
        low_pool = prev_low_m.mean(dim=-1)
        prox_high = torch.exp(-torch.abs(c - high_pool) / (atr14 + eps)) * eq_high
        prox_low = torch.exp(-torch.abs(c - low_pool) / (atr14 + eps)) * eq_low
        eqhl = torch.clamp(prox_low - prox_high, -1.0, 1.0)

        # 10) PD_LOC: premium/discount location in recent dealing range
        dr_high = cls._rolling_prev_window(h, 20).max(dim=-1).values
        dr_low = cls._rolling_prev_window(l, 20).min(dim=-1).values
        dr_mid = (dr_high + dr_low) / 2.0
        dr_half = torch.clamp((dr_high - dr_low) / 2.0, min=eps)
        pd_loc = torch.clamp((c - dr_mid) / dr_half, -3.0, 3.0)

        # 11) MIT: post-BOS mitigation/retest confirmation (signed)
        mit = torch.zeros_like(c)
        last_bos_dir = torch.zeros(n_tokens, device=c.device)
        last_bos_level = torch.zeros(n_tokens, device=c.device)

        for t in range(n_time):
            up_mask = bos_up[:, t] > 0
            dn_mask = bos_dn[:, t] > 0

            if up_mask.any():
                last_bos_dir[up_mask] = 1.0
                last_bos_level[up_mask] = swing_high_prev[up_mask, t]
            if dn_mask.any():
                last_bos_dir[dn_mask] = -1.0
                last_bos_level[dn_mask] = swing_low_prev[dn_mask, t]

            mit_up = (last_bos_dir > 0) & (l[:, t] <= last_bos_level) & (c[:, t] > last_bos_level)
            mit_dn = (last_bos_dir < 0) & (h[:, t] >= last_bos_level) & (c[:, t] < last_bos_level)
            mit[:, t] = mit_up.float() - mit_dn.float()

        # 12) VOL_IMB: volume/displacement imbalance (signed, robust)
        signed_disp = torch.tanh(body / (atr14 + eps))
        vol_imb = torch.clamp(torch.tanh(vol_z) * signed_disp, -1.0, 1.0)

        features = torch.stack([
            ls,
            bos,
            choch,
            fvg_gap,
            fvg_fill,
            ob_prox,
            brk_ob,
            disp,
            eqhl,
            pd_loc,
            mit,
            vol_imb,
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
