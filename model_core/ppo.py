import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) Trainer

    PPO improves upon vanilla policy gradient by:
    1. Clipped surrogate objective (prevents large policy updates)
    2. Multiple epochs of optimization per batch
    3. Better sample efficiency

    Reference: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        clip_ratio: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        target_kl: float = 0.01
    ):
        """
        Args:
            model: Policy network (AlphaGPT)
            optimizer: Optimizer
            clip_ratio: PPO clipping parameter (epsilon)
            vf_coef: Value function loss coefficient
            ent_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping threshold
            ppo_epochs: Number of optimization epochs per batch
            target_kl: Target KL divergence for early stopping
        """
        self.model = model
        self.optimizer = optimizer
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.target_kl = target_kl

        # Statistics tracking
        self.stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clipfrac': [],
            'explained_var': []
        }

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation (GAE)

        Args:
            rewards: [B, T] rewards
            values: [B, T] value predictions
            gamma: discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            advantages: [B, T] advantage estimates
            returns: [B, T] value targets
        """
        B, T = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        # Compute advantages using reverse iteration
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + gamma * next_value - values[:, t]
            advantages[:, t] = last_gae = delta + gamma * gae_lambda * last_gae

        returns = advantages + values
        return advantages, returns

    def ppo_update(
        self,
        sequences: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform PPO update

        Args:
            sequences: [B, T] token sequences
            old_log_probs: [B, T] old log probabilities
            advantages: [B, T] advantage estimates
            returns: [B, T] value targets

        Returns:
            stats: Dictionary of training statistics
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        epoch_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clipfrac': []
        }

        for epoch in range(self.ppo_epochs):
            # Forward pass with current policy
            B, T = sequences.shape
            inp = torch.cat([torch.zeros((B, 1), dtype=torch.long, device=sequences.device),
                           sequences], dim=1)

            new_log_probs_list = []
            new_values_list = []
            entropies_list = []

            for t in range(T):
                logits, value, _ = self.model(inp[:, :t+1])
                from torch.distributions import Categorical
                dist = Categorical(logits=logits)

                # Get log prob for the action that was taken
                action = sequences[:, t]
                new_log_prob = dist.log_prob(action)
                new_log_probs_list.append(new_log_prob)
                new_values_list.append(value.squeeze(-1))
                entropies_list.append(dist.entropy())

            new_log_probs = torch.stack(new_log_probs_list, dim=1)  # [B, T]
            new_values = torch.stack(new_values_list, dim=1)  # [B, T]
            entropy = torch.stack(entropies_list, dim=1).mean()  # scalar

            # ========== PPO Clipped Objective ==========

            # Ratio between new and old policy
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Surrogate objectives
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            # Policy loss (take minimum)
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE)
            value_loss = F.mse_loss(new_values, returns)

            # Total loss
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

            # ========== Optimization Step ==========

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # ========== Statistics ==========

            with torch.no_grad():
                # Approximate KL divergence
                approx_kl = (old_log_probs - new_log_probs).mean().item()

                # Fraction of samples clipped
                clipfrac = (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean().item()

                epoch_stats['policy_loss'].append(policy_loss.item())
                epoch_stats['value_loss'].append(value_loss.item())
                epoch_stats['entropy'].append(entropy.item())
                epoch_stats['approx_kl'].append(approx_kl)
                epoch_stats['clipfrac'].append(clipfrac)

            # Early stopping if KL divergence is too large
            if approx_kl > 1.5 * self.target_kl:
                print(f"  Early stopping at epoch {epoch+1}/{self.ppo_epochs} due to high KL: {approx_kl:.4f}")
                break

        # Average statistics across epochs
        final_stats = {k: sum(v) / len(v) if v else 0.0 for k, v in epoch_stats.items()}

        # Explained variance (how well value function predicts returns)
        with torch.no_grad():
            var_y = returns.var().item()
            explained_var = 1 - ((returns - new_values) ** 2).mean().item() / (var_y + 1e-8)
            final_stats['explained_var'] = explained_var

        # Update global stats
        for k, v in final_stats.items():
            self.stats[k].append(v)

        return final_stats


class MultiObjectiveOptimizer:
    """
    Multi-Objective Optimization using Weighted Sum

    Combines multiple objectives (e.g., Sharpe, Calmar, Win Rate)
    with adaptive weights based on performance.
    """

    def __init__(
        self,
        objectives: List[str],
        initial_weights: Dict[str, float] = None,
        adapt_weights: bool = True,
        adapt_rate: float = 0.05
    ):
        """
        Args:
            objectives: List of objective names
            initial_weights: Initial weights for each objective
            adapt_weights: Whether to adapt weights during training
            adapt_rate: Learning rate for weight adaptation
        """
        self.objectives = objectives
        self.adapt_weights = adapt_weights
        self.adapt_rate = adapt_rate

        # Initialize weights
        if initial_weights is None:
            # Equal weights
            self.weights = {obj: 1.0 / len(objectives) for obj in objectives}
        else:
            self.weights = initial_weights

        # Performance history
        self.performance_history = {obj: [] for obj in objectives}

    def compute_weighted_score(self, metrics: Dict[str, float]) -> float:
        """
        Compute weighted combination of objectives

        Args:
            metrics: Dictionary of metric values

        Returns:
            weighted_score: Weighted sum of objectives
        """
        score = 0.0
        for obj in self.objectives:
            if obj in metrics:
                score += self.weights[obj] * metrics[obj]

        return score

    def update_weights(self, metrics: Dict[str, float]):
        """
        Adapt weights based on recent performance

        Strategy: Increase weight of objectives that are improving,
                 decrease weight of objectives that are degrading
        """
        if not self.adapt_weights:
            return

        for obj in self.objectives:
            if obj not in metrics:
                continue

            # Record performance
            self.performance_history[obj].append(metrics[obj])

            # Compute improvement trend
            if len(self.performance_history[obj]) >= 10:
                recent = self.performance_history[obj][-10:]
                prev = self.performance_history[obj][-20:-10] if len(self.performance_history[obj]) >= 20 else recent

                trend = (sum(recent) / len(recent)) - (sum(prev) / len(prev))

                # Increase weight if improving, decrease if degrading
                if trend > 0:
                    self.weights[obj] = min(1.0, self.weights[obj] * (1 + self.adapt_rate))
                else:
                    self.weights[obj] = max(0.1, self.weights[obj] * (1 - self.adapt_rate))

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def get_summary(self) -> Dict:
        """Get current weights and performance summary"""
        return {
            'weights': self.weights.copy(),
            'performance_means': {
                obj: sum(hist[-100:]) / min(100, len(hist)) if hist else 0
                for obj, hist in self.performance_history.items()
            }
        }


def test_ppo():
    """Test PPO implementation"""
    print("Testing PPO Trainer...")

    # Dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            logits = self.fc(torch.randn(x.shape[0], 10))
            value = torch.randn(x.shape[0], 1)
            return logits, value, None

    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ppo = PPOTrainer(model, optimizer)

    # Dummy data
    B, T = 32, 8
    sequences = torch.randint(0, 5, (B, T))
    old_log_probs = torch.randn(B, T)
    rewards = torch.randn(B, T)
    values = torch.randn(B, T)

    # Compute advantages
    advantages, returns = ppo.compute_gae(rewards, values)

    # PPO update
    stats = ppo.ppo_update(sequences, old_log_probs, advantages, returns)

    print(f"PPO Update Stats: {stats}")
    print("✅ PPO test passed!")


if __name__ == '__main__':
    test_ppo()
