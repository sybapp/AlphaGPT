import torch

# ========== Basic Time Series Operators ==========

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    """Delay/Lag operator: shift series by d periods"""
    if d == 0:
        return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)


@torch.jit.script
def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    """Delta/Difference: x[t] - x[t-d]"""
    return x - _ts_delay(x, d)


@torch.jit.script
def _ts_mean(x: torch.Tensor, d: int) -> torch.Tensor:
    """Rolling mean over d periods"""
    if d <= 1:
        return x
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.mean(dim=-1)


@torch.jit.script
def _ts_std(x: torch.Tensor, d: int) -> torch.Tensor:
    """Rolling standard deviation over d periods"""
    if d <= 1:
        return torch.zeros_like(x)
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.std(dim=-1)


@torch.jit.script
def _ts_zscore(x: torch.Tensor, d: int) -> torch.Tensor:
    """Rolling z-score normalization"""
    if d <= 1:
        return torch.zeros_like(x)
    mean = _ts_mean(x, d)
    std = _ts_std(x, d) + 1e-6
    return (x - mean) / std


@torch.jit.script
def _ts_decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    """Linear weighted moving average (more recent = higher weight)"""
    if d <= 1:
        return x
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    weights = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()
    return (windows * weights).sum(dim=-1)


# ========== Advanced Statistical Operators ==========

@torch.jit.script
def _ts_rank(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Rolling rank percentile: position of current value in window
    Returns value in [0, 1] where 1 = highest in window
    """
    if d <= 1:
        return torch.ones_like(x) * 0.5
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)  # [B, T, d]
    current = x.unsqueeze(-1)  # [B, T, 1]
    rank = (windows < current).float().sum(dim=-1) / d
    return rank


@torch.jit.script
def _ts_argmax(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Position of maximum value in window (normalized to [0, 1])
    0 = max at beginning, 1 = max at end
    """
    if d <= 1:
        return torch.ones_like(x) * 0.5
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.argmax(dim=-1).float() / (d - 1)


@torch.jit.script
def _ts_argmin(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Position of minimum value in window (normalized to [0, 1])
    """
    if d <= 1:
        return torch.ones_like(x) * 0.5
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)
    return windows.argmin(dim=-1).float() / (d - 1)


@torch.jit.script
def _ts_skew(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Rolling skewness: measure of distribution asymmetry
    Positive = right-skewed, Negative = left-skewed
    """
    if d <= 2:
        return torch.zeros_like(x)
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)

    mean = windows.mean(dim=-1, keepdim=True)
    std = windows.std(dim=-1, keepdim=True) + 1e-6
    z = (windows - mean) / std
    return (z ** 3).mean(dim=-1)


@torch.jit.script
def _ts_kurt(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Rolling kurtosis: measure of tail heaviness
    High kurtosis = fat tails (outliers)
    """
    if d <= 2:
        return torch.zeros_like(x)
    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    windows = x_pad.unfold(1, d, 1)

    mean = windows.mean(dim=-1, keepdim=True)
    std = windows.std(dim=-1, keepdim=True) + 1e-6
    z = (windows - mean) / std
    return (z ** 4).mean(dim=-1) - 3.0  # Excess kurtosis


@torch.jit.script
def _ts_corr(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """
    Rolling correlation between two series
    Returns value in [-1, 1]
    """
    if d <= 2:
        return torch.zeros_like(x)

    pad = torch.zeros((x.shape[0], d - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    y_pad = torch.cat([pad, y], dim=1)

    x_win = x_pad.unfold(1, d, 1)
    y_win = y_pad.unfold(1, d, 1)

    x_mean = x_win.mean(dim=-1, keepdim=True)
    y_mean = y_win.mean(dim=-1, keepdim=True)

    cov = ((x_win - x_mean) * (y_win - y_mean)).mean(dim=-1)
    std_x = (x_win - x_mean).std(dim=-1) + 1e-6
    std_y = (y_win - y_mean).std(dim=-1) + 1e-6

    return cov / (std_x * std_y)


# ========== Technical Analysis Operators ==========

@torch.jit.script
def _ts_rsi(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Relative Strength Index
    Returns value in [0, 100], normalized to [-1, 1]
    """
    if d <= 1:
        return torch.zeros_like(x)

    delta = x - _ts_delay(x, 1)

    gains = torch.relu(delta)
    losses = torch.relu(-delta)

    avg_gain = _ts_mean(gains, d)
    avg_loss = _ts_mean(losses, d)

    rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return (rsi - 50.0) / 50.0  # Normalize to [-1, 1]


@torch.jit.script
def _ts_macd(x: torch.Tensor) -> torch.Tensor:
    """
    MACD: Moving Average Convergence Divergence
    fast_ema(12) - slow_ema(26)
    """
    fast_ema = _ts_decay_linear(x, 12)
    slow_ema = _ts_decay_linear(x, 26)
    return fast_ema - slow_ema


@torch.jit.script
def _ts_bollinger_upper(x: torch.Tensor, d: int) -> torch.Tensor:
    """Bollinger Band Upper: mean + 2*std"""
    mean = _ts_mean(x, d)
    std = _ts_std(x, d)
    return mean + 2.0 * std


@torch.jit.script
def _ts_bollinger_lower(x: torch.Tensor, d: int) -> torch.Tensor:
    """Bollinger Band Lower: mean - 2*std"""
    mean = _ts_mean(x, d)
    std = _ts_std(x, d)
    return mean - 2.0 * std


# ========== Advanced Logic Operators ==========

@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Conditional gate: if condition > 0 then x else y"""
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y


@torch.jit.script
def _op_jump(x: torch.Tensor) -> torch.Tensor:
    """Detect jumps/outliers: values > 3 std from mean"""
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)


@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    """Exponential decay weighted sum"""
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)


# ========== Operator Configuration ==========

OPS_CONFIG = [
    # ===== Binary Arithmetic =====
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6 * torch.sign(y)), 2),

    # ===== Unary Math =====
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('LOG', lambda x: torch.log(torch.abs(x) + 1e-6), 1),
    ('SQRT', lambda x: torch.sqrt(torch.abs(x)), 1),
    ('SQUARE', lambda x: x ** 2, 1),
    ('TANH', torch.tanh, 1),

    # ===== Time Series Basics =====
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('DELAY5', lambda x: _ts_delay(x, 5), 1),
    ('DELTA5', lambda x: _ts_delta(x, 5), 1),
    ('DELTA10', lambda x: _ts_delta(x, 10), 1),

    # ===== Moving Averages =====
    ('MA5', lambda x: _ts_mean(x, 5), 1),
    ('MA10', lambda x: _ts_mean(x, 10), 1),
    ('MA20', lambda x: _ts_mean(x, 20), 1),
    ('WMA10', lambda x: _ts_decay_linear(x, 10), 1),  # Weighted MA

    # ===== Statistical =====
    ('STD5', lambda x: _ts_std(x, 5), 1),
    ('STD10', lambda x: _ts_std(x, 10), 1),
    ('STD20', lambda x: _ts_std(x, 20), 1),
    ('ZSCORE10', lambda x: _ts_zscore(x, 10), 1),
    ('ZSCORE20', lambda x: _ts_zscore(x, 20), 1),

    # ===== Ranking & Positioning =====
    ('RANK5', lambda x: _ts_rank(x, 5), 1),
    ('RANK10', lambda x: _ts_rank(x, 10), 1),
    ('RANK20', lambda x: _ts_rank(x, 20), 1),
    ('ARGMAX10', lambda x: _ts_argmax(x, 10), 1),
    ('ARGMIN10', lambda x: _ts_argmin(x, 10), 1),

    # ===== Advanced Statistics =====
    ('SKEW10', lambda x: _ts_skew(x, 10), 1),
    ('SKEW20', lambda x: _ts_skew(x, 20), 1),
    ('KURT10', lambda x: _ts_kurt(x, 10), 1),
    ('CORR10', lambda x, y: _ts_corr(x, y, 10), 2),
    ('CORR20', lambda x, y: _ts_corr(x, y, 20), 2),

    # ===== Technical Indicators =====
    ('RSI14', lambda x: _ts_rsi(x, 14), 1),
    ('MACD', _ts_macd, 1),
    ('BOLL_UP20', lambda x: _ts_bollinger_upper(x, 20), 1),
    ('BOLL_LOW20', lambda x: _ts_bollinger_lower(x, 20), 1),

    # ===== Logic & Conditional =====
    ('GATE', _op_gate, 3),
    ('JUMP', _op_jump, 1),
    ('DECAY', _op_decay, 1),
    ('MAX', torch.max, 2),
    ('MIN', torch.min, 2),
    ('CLAMP', lambda x: torch.clamp(x, -3, 3), 1),
]


# ========== Summary Statistics ==========

def get_ops_summary():
    """Get summary of available operators"""
    summary = {
        'total_ops': len(OPS_CONFIG),
        'unary_ops': sum(1 for _, _, arity in OPS_CONFIG if arity == 1),
        'binary_ops': sum(1 for _, _, arity in OPS_CONFIG if arity == 2),
        'ternary_ops': sum(1 for _, _, arity in OPS_CONFIG if arity == 3),
    }

    categories = {
        'Arithmetic': ['ADD', 'SUB', 'MUL', 'DIV'],
        'Unary Math': ['NEG', 'ABS', 'SIGN', 'LOG', 'SQRT', 'SQUARE', 'TANH'],
        'Time Series': ['DELAY1', 'DELAY5', 'DELTA5', 'DELTA10'],
        'Moving Average': ['MA5', 'MA10', 'MA20', 'WMA10'],
        'Statistical': ['STD5', 'STD10', 'STD20', 'ZSCORE10', 'ZSCORE20', 'SKEW10', 'SKEW20', 'KURT10'],
        'Ranking': ['RANK5', 'RANK10', 'RANK20', 'ARGMAX10', 'ARGMIN10'],
        'Correlation': ['CORR10', 'CORR20'],
        'Technical': ['RSI14', 'MACD', 'BOLL_UP20', 'BOLL_LOW20'],
        'Logic': ['GATE', 'JUMP', 'DECAY', 'MAX', 'MIN', 'CLAMP']
    }

    return summary, categories


if __name__ == '__main__':
    summary, categories = get_ops_summary()
    print("=" * 60)
    print("AlphaGPT Operator Library")
    print("=" * 60)
    print(f"Total Operators: {summary['total_ops']}")
    print(f"  Unary (1 input):   {summary['unary_ops']}")
    print(f"  Binary (2 inputs): {summary['binary_ops']}")
    print(f"  Ternary (3 inputs): {summary['ternary_ops']}")
    print("\nCategories:")
    for cat, ops in categories.items():
        print(f"  {cat:15s}: {len(ops):2d} ops - {', '.join(ops[:3])}{'...' if len(ops) > 3 else ''}")
