import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json

from .config import ModelConfig
from .data_loader import CryptoDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import MemeBacktest

class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5,
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=1.0):
        """
        Initialize AlphaGPT training engine with enhanced training stability.

        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
            entropy_coef: Entropy regularization coefficient (default: 0.01)
            value_coef: Value loss coefficient for critic baseline (default: 0.5)
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        """
        self.loader = CryptoDataLoader()
        self.loader.load_data()

        self.model = AlphaGPT().to(ModelConfig.DEVICE)

        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        # Training hyperparameters
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # Low-Rank Decay regularizer
        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = StackVM()
        self.bt = MemeBacktest()

        self.best_score = -float('inf')
        self.best_formula = None
        self.best_metrics = {}
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'stable_rank': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'grad_norm': []
        }

    def train(self):
        print("🚀 Starting Meme Alpha Mining with Enhanced Training...")
        print(f"   Entropy Regularization: {self.entropy_coef}")
        print(f"   Value Loss Weight: {self.value_coef}")
        print(f"   Gradient Clipping: {self.max_grad_norm}")
        if self.use_lord:
            print(f"   LoRD Regularization: Enabled")
            print(f"   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))

        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)

            log_probs = []
            entropies = []
            values = []
            tokens_list = []

            # Sample trajectories
            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, value, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()

                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                values.append(value.squeeze(-1))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

            seqs = torch.stack(tokens_list, dim=1)

            # Evaluate rewards
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            metrics_batch = []

            for i in range(bs):
                formula = seqs[i].tolist()

                res = self.vm.execute(formula, self.loader.feat_tensor)

                if res is None:
                    rewards[i] = -5.0
                    continue

                if res.std() < 1e-4:
                    rewards[i] = -2.0
                    continue

                score, metrics = self.bt.evaluate(res, self.loader.raw_data_cache, self.loader.target_ret)
                rewards[i] = score
                metrics_batch.append(metrics)

                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula
                    self.best_metrics = metrics
                    tqdm.write(
                        f"[!] New King: Score {score:.2f} | "
                        f"Sharpe {metrics.get('sharpe', 0):.2f} | "
                        f"Calmar {metrics.get('calmar', 0):.2f} | "
                        f"MaxDD {metrics.get('max_dd', 0):.2%} | "
                        f"WinRate {metrics.get('win_rate', 0):.2%}"
                    )

            # ========== Enhanced Loss Computation ==========

            # 1. Use critic as baseline (reduce variance)
            baseline = torch.stack(values, dim=1).mean(dim=1)  # Average value over sequence
            advantages = rewards - baseline.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            # 2. Policy loss
            log_probs_stacked = torch.stack(log_probs, dim=1)  # [B, T]
            policy_loss = -(log_probs_stacked * advantages.unsqueeze(1)).mean()

            # 3. Value loss (MSE between predicted value and actual reward)
            value_loss = torch.nn.functional.mse_loss(baseline, rewards)

            # 4. Entropy bonus (encourage exploration)
            entropies_stacked = torch.stack(entropies, dim=1)  # [B, T]
            entropy = entropies_stacked.mean()
            entropy_loss = -self.entropy_coef * entropy  # Negative because we want to maximize entropy

            # 5. Total loss
            loss = policy_loss + self.value_coef * value_loss + entropy_loss

            # ========== Gradient Optimization ==========

            self.opt.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.opt.step()

            # Apply Low-Rank Decay regularization
            if self.use_lord:
                self.lord_opt.step()

            # ========== Logging ==========

            avg_reward = rewards.mean().item()
            postfix_dict = {
                'AvgRew': f"{avg_reward:.3f}",
                'BestScore': f"{self.best_score:.3f}",
                'PL': f"{policy_loss.item():.3f}",
                'VL': f"{value_loss.item():.3f}",
                'Ent': f"{entropy.item():.3f}",
                'GN': f"{grad_norm.item():.2f}"
            }

            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)

            # Record history
            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)
            self.training_history['policy_loss'].append(policy_loss.item())
            self.training_history['value_loss'].append(value_loss.item())
            self.training_history['entropy'].append(entropy.item())
            self.training_history['grad_norm'].append(grad_norm.item())

            pbar.set_postfix(postfix_dict)

        # Save best formula
        with open("best_meme_strategy.json", "w") as f:
            json.dump({
                'formula': self.best_formula,
                'score': self.best_score,
                'metrics': self.best_metrics
            }, f, indent=2)

        # Save training history
        with open("training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)

        print(f"\n✓ Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")
        print(f"\n  Detailed Metrics:")
        for k, v in self.best_metrics.items():
            print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()