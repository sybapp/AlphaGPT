import torch
import numpy as np


class MemeBacktest:
    """
    Enhanced Backtest Engine with Multi-Metric Reward Function

    Improvements:
    - Sharpe Ratio: Risk-adjusted return metric
    - Calmar Ratio: Return / Max Drawdown
    - Sortino Ratio: Downside risk adjusted return
    - Win Rate: Percentage of profitable periods
    - Soft activity penalty: Replace hard threshold with smooth penalty
    """

    def __init__(self):
        self.trade_size = 1000.0
        self.min_liq = 500000.0
        self.base_fee = 0.0060

        # Reward function weights
        self.weights = {
            'sharpe': 0.35,
            'calmar': 0.25,
            'sortino': 0.20,
            'win_rate': 0.10,
            'turnover_penalty': 0.10
        }

    def evaluate(self, factors, raw_data, target_ret):
        """
        Evaluate trading strategy with comprehensive metrics

        Args:
            factors: [B, T] Factor values for each token
            raw_data: Dict containing liquidity, fdv, etc.
            target_ret: [B, T] Target returns

        Returns:
            final_score: Scalar fitness score
            metrics_dict: Dictionary of detailed metrics
        """
        liquidity = raw_data['liquidity']

        # Generate trading signals
        signal = torch.sigmoid(factors)
        is_safe = (liquidity > self.min_liq).float()
        position = (signal > 0.85).float() * is_safe

        # Transaction cost modeling
        impact_slippage = self.trade_size / (liquidity + 1e-9)
        impact_slippage = torch.clamp(impact_slippage, 0.0, 0.05)
        total_slippage_one_way = self.base_fee + impact_slippage

        # Calculate turnover
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way

        # PnL calculation
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost

        # ========== Multi-Metric Evaluation ==========

        # 1. Sharpe Ratio (annualized)
        daily_ret = net_pnl
        mean_ret = daily_ret.mean(dim=1)
        std_ret = daily_ret.std(dim=1) + 1e-6
        sharpe = mean_ret / std_ret * torch.sqrt(torch.tensor(252.0, device=daily_ret.device))

        # 2. Maximum Drawdown & Calmar Ratio
        cum_nav = (1 + daily_ret).cumprod(dim=1)
        running_max = torch.cummax(cum_nav, dim=1)[0]
        drawdown = (cum_nav - running_max) / (running_max + 1e-9)
        max_dd = drawdown.min(dim=1)[0].abs()

        # Annualized return
        T = daily_ret.shape[1]
        annual_ret = daily_ret.sum(dim=1) * (252 / T)
        calmar = annual_ret / (max_dd + 1e-3)  # Avoid division by zero

        # 3. Sortino Ratio (downside risk)
        downside_mask = daily_ret < 0
        downside_returns = daily_ret * downside_mask.float()
        downside_std = torch.sqrt((downside_returns ** 2).sum(dim=1) / (downside_mask.sum(dim=1) + 1)) + 1e-6
        sortino = mean_ret / downside_std * torch.sqrt(torch.tensor(252.0, device=daily_ret.device))

        # 4. Win Rate
        win_rate = (daily_ret > 0).float().mean(dim=1)

        # 5. Activity Penalty (soft constraint)
        activity = position.sum(dim=1)
        activity_ratio = activity / T
        # Penalize if activity < 10% (too inactive) or > 80% (overtrading)
        activity_penalty = torch.relu(0.1 - activity_ratio) * 5.0 + torch.relu(activity_ratio - 0.8) * 3.0

        # 6. Turnover Penalty (penalize excessive trading)
        avg_turnover = turnover.mean(dim=1)
        turnover_penalty = torch.clamp(avg_turnover - 0.3, min=0) * 3.0

        # ========== Weighted Composite Score ==========
        score = (
            self.weights['sharpe'] * sharpe +
            self.weights['calmar'] * calmar +
            self.weights['sortino'] * sortino +
            self.weights['win_rate'] * (win_rate - 0.5) * 10.0 +  # Normalize win_rate
            -self.weights['turnover_penalty'] * turnover_penalty -
            activity_penalty
        )

        # Handle edge cases
        score = torch.where(torch.isnan(score) | torch.isinf(score),
                           torch.tensor(-10.0, device=score.device),
                           score)

        # Clip extreme values
        score = torch.clamp(score, -10.0, 10.0)

        # Use median for robustness
        final_fitness = torch.median(score)

        # Return detailed metrics for logging
        metrics = {
            'sharpe': sharpe.mean().item(),
            'calmar': calmar.mean().item(),
            'sortino': sortino.mean().item(),
            'win_rate': win_rate.mean().item(),
            'max_dd': max_dd.mean().item(),
            'annual_ret': annual_ret.mean().item(),
            'avg_turnover': avg_turnover.mean().item(),
            'cum_ret': net_pnl.sum(dim=1).mean().item()
        }

        return final_fitness, metrics


class A_ShareBacktest:
    """
    Backtest engine for A-Share market (code/main.py)
    Enhanced version with improved reward function
    """

    def __init__(self, cost_rate=0.0005):
        self.cost_rate = cost_rate

    def evaluate(self, factor, target_ret, split_idx=None):
        """
        Evaluate A-Share strategy

        Args:
            factor: [T] Factor values
            target_ret: [T] Target returns (open-to-open)
            split_idx: Train/test split index

        Returns:
            score: Fitness score (Sortino ratio)
            metrics: Detailed metrics dict
        """
        if split_idx is not None:
            factor = factor[:split_idx]
            target_ret = target_ret[:split_idx]

        # Handle invalid factors
        if torch.isnan(factor).all() or (factor == 0).all() or factor.numel() == 0:
            return torch.tensor(-5.0), {}

        # Generate signals
        signal = torch.tanh(factor)
        position = torch.sign(signal)

        # Calculate turnover
        turnover = torch.abs(position - torch.roll(position, 1))
        if turnover.numel() > 0:
            turnover[0] = 0.0
        else:
            return torch.tensor(-5.0), {}

        # Net PnL
        gross_pnl = position * target_ret
        tx_cost = turnover * self.cost_rate
        net_pnl = gross_pnl - tx_cost

        if net_pnl.numel() < 10:
            return torch.tensor(-5.0), {}

        # ========== Metrics ==========

        # 1. Sortino Ratio (primary metric)
        mean_ret = net_pnl.mean()
        downside_returns = net_pnl[net_pnl < 0]

        if downside_returns.numel() > 5:
            down_std = downside_returns.std() + 1e-6
            sortino = mean_ret / down_std * 15.87  # Annualized
        else:
            std_ret = net_pnl.std() + 1e-6
            sortino = mean_ret / std_ret * 15.87

        # 2. Sharpe Ratio
        sharpe = mean_ret / (net_pnl.std() + 1e-6) * 15.87

        # 3. Maximum Drawdown
        cum_nav = (1 + net_pnl).cumprod(dim=0)
        running_max = torch.cummax(cum_nav, dim=0)[0]
        drawdown = (cum_nav - running_max) / (running_max + 1e-9)
        max_dd = drawdown.min().abs()

        # 4. Win Rate
        win_rate = (net_pnl > 0).float().mean()

        # 5. Calmar Ratio
        annual_ret = net_pnl.sum() * (252 / len(net_pnl))
        calmar = annual_ret / (max_dd + 1e-3)

        # ========== Penalties ==========

        # Penalty for negative returns
        if mean_ret < 0:
            sortino = torch.tensor(-5.0)

        # Penalty for excessive trading (>50% turnover)
        if turnover.mean() > 0.5:
            sortino = sortino - 1.5

        # Penalty for no trading
        if (position == 0).all():
            sortino = torch.tensor(-5.0)

        # Penalty for extreme drawdown
        if max_dd > 0.3:  # >30% drawdown
            sortino = sortino - (max_dd - 0.3) * 10.0

        score = torch.clamp(sortino, -5.0, 10.0)

        metrics = {
            'sortino': sortino.item(),
            'sharpe': sharpe.item(),
            'calmar': calmar.item(),
            'max_dd': max_dd.item(),
            'win_rate': win_rate.item(),
            'annual_ret': annual_ret.item(),
            'avg_turnover': turnover.mean().item()
        }

        return score, metrics
