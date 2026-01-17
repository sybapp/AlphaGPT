# AlphaGPT 训练指南

## 📚 目录
1. [训练原理](#训练原理)
2. [方式一：A股回测训练](#方式一a股回测训练)
3. [方式二：加密货币训练](#方式二加密货币训练)
4. [训练技巧与调优](#训练技巧与调优)
5. [常见问题](#常见问题)

---

## 🧠 训练原理

AlphaGPT 使用**符号回归 + 强化学习**自动挖掘交易因子：

1. **模型架构**: Transformer 编码器 + Actor-Critic 网络
2. **训练方法**: Policy Gradient (策略梯度)
3. **奖励函数**: 夏普比率 / 信息系数
4. **正则化**: Newton-Schulz Low-Rank Decay (LoRD) 防止过拟合

**训练流程**:
```
初始化模型 → 生成因子公式 → 回测评估 → 计算奖励 → 更新策略 → 循环迭代
```

---

## 方式一：A股回测训练

### 🎯 适用场景
- 研究A股/港股/商品期货策略
- 简单快速的策略验证
- 无需数据库和复杂环境

### 第一步：准备环境

```bash
# 1. 安装依赖
pip install torch numpy pandas tushare matplotlib tqdm

# 2. 获取 Tushare Token
# 访问 https://tushare.pro/ 注册并获取免费Token
```

### 第二步：配置参数

编辑 `code/main.py`:

```python
# ========== 数据配置 ==========
TS_TOKEN = 'your_tushare_token_here'  # 填入你的Token
INDEX_CODE = '511260.SH'               # 交易标的
START_DATE = '20150101'                # 训练开始日期
END_DATE = '20240101'                  # 训练结束日期
TEST_END_DATE = '20250101'             # 测试结束日期

# ========== 训练配置 ==========
BATCH_SIZE = 1024                      # 每次生成1024个公式
TRAIN_ITERATIONS = 400                 # 训练400轮
MAX_SEQ_LEN = 8                        # 公式最大长度8个token
COST_RATE = 0.0005                     # 交易成本万五
```

**推荐标的**:
- `511260.SH` - 十年国债ETF（低波动，适合新手）
- `000905.SH` - 中证500指数（中等波动）
- `000852.SH` - 中证1000指数（高波动，高收益潜力）

### 第三步：运行训练

```bash
python code/main.py
```

**训练过程输出**:
```
🌐 Fetching 511260.SH...
✅ Data Loaded: 2000 days
🚀 Starting Training...

100%|████████| 400/400 [05:23<00:00,  1.24it/s, Loss=-0.156, AvgRew=0.234, Best=2.13]

🎯 Training Complete!
💾 Saved: best_formula_*.txt

📊 Backtesting Best Formula...
✅ Train Sharpe: 1.85
✅ Test Sharpe: 1.72
📈 Annualized Return: 24.3%
📉 Max Drawdown: -8.7%
```

### 第四步：查看结果

训练完成后会生成：

1. **best_formula_*.txt** - 最优因子公式
   ```
   Token序列: [2, 15, 1, 16, 7, ...]
   中缀表达式: MA20(SUB(RET5, DELTA5(VOL_CHG)))
   ```

2. **backtest.png** - 净值曲线图
   - 蓝线：策略净值
   - 橙线：基准净值

3. **data_cache_final.parquet** - 缓存数据（下次训练更快）

### 第五步：参数调优

#### 提高收益（可能增加过拟合风险）
```python
MAX_SEQ_LEN = 12          # 增加公式复杂度
TRAIN_ITERATIONS = 1000   # 更多训练轮次
```

#### 降低过拟合
```python
MAX_SEQ_LEN = 6           # 简化公式
COST_RATE = 0.001         # 提高交易成本惩罚
```

#### GPU加速
```python
DEVICE = torch.device("cuda")  # 自动启用GPU
```

---

## 方式二：加密货币训练

### 🎯 适用场景
- 实盘交易加密货币
- 大规模多标的训练
- 需要实时数据更新

### 第一步：安装依赖

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt
```

### 第二步：配置数据库

```bash
# 安装 PostgreSQL
sudo apt install postgresql postgresql-contrib  # Ubuntu
brew install postgresql                          # macOS

# 启动数据库
sudo systemctl start postgresql  # Linux
brew services start postgresql   # macOS

# 创建数据库
createdb crypto_quant
```

### 第三步：配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑配置
nano .env
```

**必填配置**:
```bash
# 数据库
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_NAME=crypto_quant

# Birdeye API（需付费）
BIRDEYE_API_KEY=your_api_key_here
```

**获取 Birdeye API**:
1. 访问 https://birdeye.so/
2. 注册账号并选择付费套餐（需要1分钟K线数据）
3. 复制API密钥

### 第四步：数据采集

```bash
python -m data_pipeline.run_pipeline
```

**过程说明**:
```
🚀 Starting Data Pipeline...
📡 Fetching token list from Birdeye...
✅ Found 1,234 tokens (filtered by liquidity > $500K)

⏬ Downloading historical data (7 days, 1min candles)...
100%|████████| 1234/1234 [12:34<00:00,  1.64it/s]

💾 Saved to PostgreSQL: crypto_quant.ohlcv
```

**筛选条件** (可在 `data_pipeline/config.py` 修改):
- 流动性 ≥ $500,000
- FDV ≥ $10,000,000
- 链: Solana
- 时间粒度: 1分钟

### 第五步：训练模型

```bash
python -m model_core.engine
```

**训练输出**:
```
🚀 Starting Meme Alpha Mining with LoRD Regularization...
   LoRD Regularization enabled
   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']

📊 Loaded 1,234 tokens, 10,080 candles per token
🔀 Train/Test split: 80% / 20%

Training: 100%|████████| 1000/1000 [23:45<00:00,  1.43s/it]
[!] New King: Score 2.45 | Ret 18.3% | Formula [3, 12, 1, 8, 15, ...]

✅ Training Complete!
💾 Saved: best_meme_strategy.json
💾 Saved: training_history.json
```

### 第六步：可视化分析

```bash
streamlit run dashboard/app.py
```

访问 `http://localhost:8501`，你将看到：

1. **净值曲线** - 实时回测表现
2. **因子表达式** - 最优公式解析
3. **持仓分析** - Top 10 代币
4. **风险指标** - 夏普/最大回撤/胜率

### 第七步：调优参数

编辑 `model_core/config.py`:

```python
# ========== 训练参数 ==========
BATCH_SIZE = 8192          # 增加批次 → 更稳定但更慢
TRAIN_STEPS = 2000         # 更多步数 → 更好收敛
MAX_FORMULA_LEN = 12       # 更复杂公式 → 可能过拟合

# ========== 回测参数 ==========
TRADE_SIZE_USD = 1000.0    # 单笔交易金额
MIN_LIQUIDITY = 5000.0     # 最小流动性（过滤归零币）
BASE_FEE = 0.005           # 交易费率 0.5%

# ========== LoRD正则化 ==========
# 编辑 model_core/engine.py
engine = AlphaEngine(
    use_lord_regularization=True,
    lord_decay_rate=1e-3,      # 增加 → 更强正则化
    lord_num_iterations=5       # 迭代次数
)
```

---

## 🎓 训练技巧与调优

### 1. 防止过拟合

**症状**: 训练集表现好，测试集差
**解决方案**:
```python
# 方法1: 简化公式
MAX_FORMULA_LEN = 6  # 减少复杂度

# 方法2: 启用LoRD正则化
use_lord_regularization=True
lord_decay_rate=5e-3  # 增加正则化强度

# 方法3: 增加交易成本
COST_RATE = 0.001  # 提高费率惩罚

# 方法4: 扩大数据集
START_DATE = '20100101'  # 更长历史
```

### 2. 加速训练

**方法1: GPU加速**
```bash
# 检查CUDA
nvidia-smi

# 自动使用GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**方法2: 减小搜索空间**
```python
BATCH_SIZE = 512      # 减小批次
MAX_FORMULA_LEN = 6   # 简化公式
TRAIN_ITERATIONS = 200  # 减少轮次
```

**方法3: 数据缓存**
```python
# 首次运行会下载数据并缓存到 .parquet 文件
# 后续训练直接读取缓存，速度提升10倍+
```

### 3. 提高夏普比率

**策略1: 优化奖励函数**
```python
# 编辑 model_core/backtest.py
# 当前: 奖励 = 信息系数(IC)
# 可改为: 奖励 = 夏普比率 * 收益率
```

**策略2: 多因子融合**
```bash
# 训练多个模型，选择Top 3公式
python code/main.py --model_id 1
python code/main.py --model_id 2
python code/main.py --model_id 3

# 手动组合因子（等权或优化权重）
```

**策略3: 时间窗口优化**
```python
# A股: 使用近5年数据（避免市场风格切换）
START_DATE = '20190101'

# 加密: 缩短历史（高波动市场）
HISTORY_DAYS = 3
```

### 4. 监控训练过程

**关键指标**:
- `AvgRew`: 平均奖励（应逐渐上升）
- `BestScore`: 最优得分（应持续刷新）
- `Rank`: 稳定秩（LoRD启用时，应下降）

**异常情况**:
```
# 1. AvgRew 不上升 → 学习率太低
self.opt = torch.optim.AdamW(self.model.parameters(), lr=5e-3)

# 2. Loss=NaN → 梯度爆炸
torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

# 3. 所有公式都 -5.0 → 公式构建失败
# 检查算子定义是否有bug
```

---

## ❓ 常见问题

### Q1: 训练多久能完成？

**A股系统**:
- CPU: ~5-10分钟 (400轮)
- GPU: ~2-3分钟

**加密货币系统**:
- 数据采集: 10-30分钟（取决于代币数量）
- 模型训练: 20-60分钟 (1000轮)

### Q2: 需要多少内存？

- A股系统: 2GB+
- 加密系统: 8GB+ (推荐16GB)

### Q3: 训练出来的策略有效期多久？

- **A股**: 通常3-6个月后需要重训练
- **加密**: 1-2周后需要重训练（市场变化快）

建议: **每周重新训练一次**

### Q4: 如何判断训练成功？

**成功标志**:
1. 测试集夏普 > 1.0
2. 年化收益 > 10%
3. 最大回撤 < 20%
4. 训练/测试集表现相近（无过拟合）

**失败标志**:
1. 测试集夏普 < 0.5
2. 最大回撤 > 50%
3. 训练集好、测试集差（严重过拟合）

### Q5: Tushare 积分不够怎么办？

```bash
# 方法1: 参与Tushare社区活动获取积分
# 方法2: 付费订阅（¥99/年）
# 方法3: 使用其他数据源（AKShare/yfinance）
```

### Q6: Birdeye API太贵了？

**免费替代方案**:
```python
# 编辑 data_pipeline/config.py
USE_DEXSCREENER = True  # 使用免费的DexScreener API
TIMEFRAME = "15min"      # 但只能获取15分钟K线
```

---

## 🚀 进阶训练

### 多标的训练（加密货币）

```bash
# 同时训练多个策略
python -m model_core.engine --strategy long_only
python -m model_core.engine --strategy short_only
python -m model_core.engine --strategy market_neutral
```

### 自定义因子库

编辑 `model_core/factors.py` 或 `code/main.py`:

```python
# 添加自定义算子
OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),

    # 新增: 动量反转
    ('MOMENTUM', lambda x: _ts_delta(x, 20), 1),

    # 新增: 布林带
    ('BOLL_UP', lambda x: x + 2 * _ts_zscore(x, 20), 1),
    ('BOLL_DOWN', lambda x: x - 2 * _ts_zscore(x, 20), 1),
]
```

### 分布式训练

```bash
# 多机训练（需配置Ray或Horovod）
# 暂不支持，未来版本会加入
```

---

## 📞 获取帮助

训练遇到问题？

1. **查看日志**: 检查终端输出的错误信息
2. **检查配置**: 确认 .env 和参数设置正确
3. **社群求助**: QQ群 838641831
4. **提交Issue**: https://github.com/imbue-bit/AlphaGPT/issues

---

## ⚠️ 重要提醒

1. **训练≠盈利**: 回测好不代表实盘能赚钱
2. **定期重训**: 市场环境变化，策略需要更新
3. **风险控制**: 设置止损，控制仓位
4. **小资金测试**: 实盘前先用小钱验证

**Good luck! 祝你挖到Alpha! 🎯**
