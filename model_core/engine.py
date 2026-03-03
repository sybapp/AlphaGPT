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
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5, factor_mode=None):
        """
        Initialize AlphaGPT training engine.
        
        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
        """
        self.factor_mode = (factor_mode or ModelConfig.FACTOR_MODE)
        self.loader = CryptoDataLoader(factor_mode=self.factor_mode)
        self.loader.load_data()
        
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        
        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
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
        self.best_train_score = -float('inf')
        self.best_formula = None
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'avg_test_score': [],
            'avg_generalization_gap': [],
            'best_score': [],
            'best_train_score': [],
            'stable_rank': []
        }

    def train(self):
        print("🚀 Starting Meme Alpha Mining with LoRD Regularization..." if self.use_lord else "🚀 Starting Meme Alpha Mining...")
        if self.use_lord:
            print(f"   LoRD Regularization enabled")
            print(f"   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        
        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            
            log_probs = []
            tokens_list = []
            
            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()
                
                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
            
            seqs = torch.stack(tokens_list, dim=1)
            
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            test_scores = torch.zeros(bs, device=ModelConfig.DEVICE)
            generalization_gaps = torch.zeros(bs, device=ModelConfig.DEVICE)

            for i in range(bs):
                formula = seqs[i].tolist()

                res_train = self.vm.execute(formula, self.loader.feat_tensor_train)

                if res_train is None:
                    rewards[i] = -5.0
                    test_scores[i] = -5.0
                    generalization_gaps[i] = 0.0
                    continue

                if res_train.std() < 1e-4:
                    rewards[i] = -2.0
                    test_scores[i] = -2.0
                    generalization_gaps[i] = 0.0
                    continue

                res_test = self.vm.execute(formula, self.loader.feat_tensor_test)
                if res_test is None or res_test.std() < 1e-4:
                    rewards[i] = -2.0
                    test_scores[i] = -2.0
                    generalization_gaps[i] = 0.0
                    continue

                train_score, train_ret = self.bt.evaluate(res_train, self.loader.raw_data_train, self.loader.target_ret_train)
                test_score, test_ret = self.bt.evaluate(res_test, self.loader.raw_data_test, self.loader.target_ret_test)

                rewards[i] = train_score
                test_scores[i] = test_score
                gap = train_score - test_score
                generalization_gaps[i] = gap

                combined_score = test_score - 0.1 * torch.abs(gap)
                if combined_score.item() > self.best_score:
                    self.best_score = combined_score.item()
                    self.best_train_score = train_score.item()
                    self.best_formula = formula
                    tqdm.write(
                        f"[!] New King: Combined {combined_score:.2f} | "
                        f"Train {train_score:.2f} ({train_ret:.2%}) | "
                        f"Test {test_score:.2f} ({test_ret:.2%}) | Gap {gap:.2f} | "
                        f"Formula {formula}"
                    )
            
            # Normalize rewards
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            loss = 0
            for t in range(len(log_probs)):
                loss += -log_probs[t] * adv
            
            loss = loss.mean()
            
            # Gradient step
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            # Apply Low-Rank Decay regularization
            if self.use_lord:
                self.lord_opt.step()
            
            # Logging
            avg_reward = rewards.mean().item()
            avg_test_score = test_scores.mean().item()
            avg_gap = generalization_gaps.mean().item()
            postfix_dict = {
                'TrainAvg': f"{avg_reward:.3f}",
                'TestAvg': f"{avg_test_score:.3f}",
                'BestCombo': f"{self.best_score:.3f}"
            }
            
            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
            
            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['avg_test_score'].append(avg_test_score)
            self.training_history['avg_generalization_gap'].append(avg_gap)
            self.training_history['best_score'].append(self.best_score)
            self.training_history['best_train_score'].append(self.best_train_score)
            
            pbar.set_postfix(postfix_dict)

        # Save best formula
        with open("best_meme_strategy.json", "w") as f:
            json.dump(self.best_formula, f)
        
        # Save training history
        import json as js
        with open("training_history.json", "w") as f:
            js.dump(self.training_history, f)
        
        print(f"\n✓ Training completed!")
        print(f"  Best combined score: {self.best_score:.4f}")
        print(f"  Best train score: {self.best_train_score:.4f}")
        print(f"  Best formula: {self.best_formula}")


if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()