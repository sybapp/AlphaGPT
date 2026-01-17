import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


class OverfitDetector:
    """
    Overfitting Detection and Early Stopping

    Features:
    - Train/Val performance divergence detection
    - Early stopping with patience
    - Training stability monitoring
    - Automatic checkpoint saving
    """

    def __init__(self, patience=50, min_delta=0.01, mode='max'):
        """
        Args:
            patience: Number of steps without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' or 'min' - whether higher or lower is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        # History tracking
        self.train_history = []
        self.val_history = []
        self.step_history = []

        # Early stopping state
        self.best_val = float('-inf') if mode == 'max' else float('inf')
        self.best_step = 0
        self.counter = 0
        self.should_stop = False

        # Overfitting detection
        self.overfit_signals = []

    def update(self, step: int, train_score: float, val_score: float) -> Tuple[bool, Optional[str]]:
        """
        Update detector with new scores

        Args:
            step: Current training step
            train_score: Training set performance
            val_score: Validation set performance

        Returns:
            (should_stop, reason): Whether to stop training and reason
        """
        self.step_history.append(step)
        self.train_history.append(train_score)
        self.val_history.append(val_score)

        # 1. Check for overfitting signal
        overfit_detected, overfit_msg = self._detect_overfitting()
        if overfit_detected:
            self.overfit_signals.append((step, overfit_msg))

        # 2. Early stopping logic
        is_better = self._is_improvement(val_score)

        if is_better:
            self.best_val = val_score
            self.best_step = step
            self.counter = 0
        else:
            self.counter += 1

        # 3. Decide whether to stop
        if self.counter >= self.patience:
            self.should_stop = True
            reason = f"Early stopping: no improvement for {self.patience} steps (best at step {self.best_step})"
            return True, reason

        # 4. Check for severe overfitting
        if len(self.overfit_signals) >= 3:
            last_3_signals = self.overfit_signals[-3:]
            if all(step >= self.step_history[-10] for step, _ in last_3_signals):
                self.should_stop = True
                reason = f"Severe overfitting detected: {len(self.overfit_signals)} signals"
                return True, reason

        return False, None

    def _is_improvement(self, val_score: float) -> bool:
        """Check if validation score improved"""
        if self.mode == 'max':
            return val_score > self.best_val + self.min_delta
        else:
            return val_score < self.best_val - self.min_delta

    def _detect_overfitting(self) -> Tuple[bool, Optional[str]]:
        """
        Detect overfitting signals

        Returns:
            (is_overfitting, message)
        """
        if len(self.train_history) < 20:
            return False, None

        # Signal 1: Train improving, Val degrading
        train_recent = np.mean(self.train_history[-10:])
        train_prev = np.mean(self.train_history[-20:-10])
        val_recent = np.mean(self.val_history[-10:])
        val_prev = np.mean(self.val_history[-20:-10])

        train_trend = train_recent - train_prev
        val_trend = val_recent - val_prev

        if self.mode == 'max':
            # Train increasing, Val decreasing
            if train_trend > self.min_delta and val_trend < -self.min_delta:
                return True, f"Train↑{train_trend:.3f} Val↓{val_trend:.3f}"
        else:
            # Train decreasing (improving), Val increasing (degrading)
            if train_trend < -self.min_delta and val_trend > self.min_delta:
                return True, f"Train↓{train_trend:.3f} Val↑{val_trend:.3f}"

        # Signal 2: Train-Val gap widening
        gap_recent = abs(train_recent - val_recent)
        gap_prev = abs(train_prev - val_prev)

        if gap_recent > gap_prev + 0.1 and gap_recent > 0.5:
            return True, f"Train-Val gap widening: {gap_prev:.3f} → {gap_recent:.3f}"

        # Signal 3: Val performance plateau while train improves
        val_std_recent = np.std(self.val_history[-10:])
        if val_std_recent < 0.01 and abs(train_trend) > self.min_delta:
            return True, f"Val plateau (std={val_std_recent:.4f}) while train moves"

        return False, None

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if len(self.train_history) == 0:
            return {}

        return {
            'best_val': self.best_val,
            'best_step': self.best_step,
            'current_patience': self.counter,
            'overfit_signals': len(self.overfit_signals),
            'train_mean_recent': np.mean(self.train_history[-10:]) if len(self.train_history) >= 10 else np.mean(self.train_history),
            'val_mean_recent': np.mean(self.val_history[-10:]) if len(self.val_history) >= 10 else np.mean(self.val_history),
            'train_val_gap': abs(np.mean(self.train_history[-10:]) - np.mean(self.val_history[-10:])) if len(self.train_history) >= 10 else 0,
        }


class TrainingStabilityMonitor:
    """
    Monitor training stability and detect anomalies

    Features:
    - Loss explosion detection
    - NaN/Inf detection
    - Gradient norm monitoring
    - Reward distribution tracking
    """

    def __init__(self, loss_explosion_threshold=10.0, grad_norm_threshold=100.0):
        self.loss_explosion_threshold = loss_explosion_threshold
        self.grad_norm_threshold = grad_norm_threshold

        self.loss_history = []
        self.grad_norm_history = []
        self.reward_history = []

        self.anomalies = []

    def update(self, step: int, loss: float, grad_norm: float, rewards: torch.Tensor):
        """Update monitor with training metrics"""
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        self.reward_history.append(rewards.mean().item())

        # Check for anomalies
        anomaly_detected, anomaly_msg = self._check_anomalies(step, loss, grad_norm, rewards)
        if anomaly_detected:
            self.anomalies.append((step, anomaly_msg))
            return True, anomaly_msg

        return False, None

    def _check_anomalies(self, step: int, loss: float, grad_norm: float, rewards: torch.Tensor) -> Tuple[bool, Optional[str]]:
        """Check for training anomalies"""

        # 1. NaN/Inf detection
        if np.isnan(loss) or np.isinf(loss):
            return True, f"Loss is {loss} at step {step}"

        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            return True, f"Rewards contain NaN/Inf at step {step}"

        # 2. Loss explosion
        if len(self.loss_history) >= 10:
            recent_loss = np.mean(self.loss_history[-10:])
            if loss > recent_loss * self.loss_explosion_threshold:
                return True, f"Loss explosion: {loss:.4f} >> {recent_loss:.4f}"

        # 3. Gradient explosion
        if grad_norm > self.grad_norm_threshold:
            return True, f"Gradient explosion: {grad_norm:.2f} > {self.grad_norm_threshold}"

        # 4. Reward collapse (all rewards are identical)
        if rewards.std() < 1e-6:
            return True, f"Reward collapse: std={rewards.std():.2e}"

        # 5. Extreme gradient vanishing
        if grad_norm < 1e-8 and step > 100:
            return True, f"Gradient vanishing: {grad_norm:.2e}"

        return False, None

    def get_summary(self) -> Dict:
        """Get stability summary"""
        if len(self.loss_history) == 0:
            return {}

        return {
            'loss_mean': np.mean(self.loss_history[-100:]),
            'loss_std': np.std(self.loss_history[-100:]),
            'grad_norm_mean': np.mean(self.grad_norm_history[-100:]),
            'grad_norm_max': np.max(self.grad_norm_history[-100:]),
            'reward_mean': np.mean(self.reward_history[-100:]),
            'reward_std': np.std(self.reward_history[-100:]),
            'anomalies_count': len(self.anomalies)
        }


def test_detector():
    """Test overfitting detector"""
    detector = OverfitDetector(patience=10, min_delta=0.01, mode='max')

    # Simulate training
    print("Simulating training...")
    for step in range(100):
        # Simulate normal training (both improving)
        if step < 50:
            train_score = 0.5 + step * 0.01 + np.random.randn() * 0.02
            val_score = 0.5 + step * 0.008 + np.random.randn() * 0.03
        # Simulate overfitting (train improving, val degrading)
        else:
            train_score = 1.0 + (step - 50) * 0.01 + np.random.randn() * 0.02
            val_score = 0.9 - (step - 50) * 0.005 + np.random.randn() * 0.03

        should_stop, reason = detector.update(step, train_score, val_score)

        if step % 10 == 0:
            summary = detector.get_summary()
            print(f"Step {step}: Train={train_score:.3f}, Val={val_score:.3f}, Signals={summary.get('overfit_signals', 0)}")

        if should_stop:
            print(f"\n⚠️  Training stopped: {reason}")
            break

    print(f"\nFinal Summary: {detector.get_summary()}")
    print(f"Overfit signals: {detector.overfit_signals}")


if __name__ == '__main__':
    test_detector()
