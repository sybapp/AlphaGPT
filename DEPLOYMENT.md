# AlphaGPT 部署文档

## 📖 目录
1. [快速开始](#快速开始)
2. [详细部署步骤](#详细部署步骤)
3. [系统架构](#系统架构)
4. [使用指南](#使用指南)
5. [常见问题](#常见问题)

---

## 🚀 快速开始

### 一键部署（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/imbue-bit/AlphaGPT.git
cd AlphaGPT

# 2. 运行部署脚本
chmod +x deploy.sh
./deploy.sh

# 3. 配置环境变量
cp .env.example .env
nano .env  # 填入你的API密钥

# 4. 运行（选择一种模式）
# A股回测模式
python code/main.py

# 加密货币模式
python -m data_pipeline.run_pipeline
```

---

## 📦 详细部署步骤

### 系统 A: A股量化回测系统

适用于研究中国A股市场策略。

#### 1. 安装依赖

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# 安装依赖
pip install torch numpy pandas tushare matplotlib tqdm
```

#### 2. 配置 Tushare Token

```bash
# 访问 https://tushare.pro/ 注册并获取Token
# 编辑 code/main.py 第13行，填入你的Token
TS_TOKEN = 'your_tushare_token_here'
```

#### 3. 运行回测

```bash
python code/main.py
```

**输出说明:**
- `best_formula_*.txt`: 最优因子公式
- `backtest.png`: 回测净值曲线
- `data_cache_final.parquet`: 缓存的行情数据

#### 4. 参数调整

编辑 `code/main.py` 修改以下参数:

```python
INDEX_CODE = '511260.SH'      # 交易标的代码
START_DATE = '20150101'        # 回测开始日期
END_DATE = '20240101'          # 训练截止日期
TEST_END_DATE = '20250101'     # 测试截止日期
BATCH_SIZE = 1024              # 批次大小
TRAIN_ITERATIONS = 400         # 训练迭代次数
MAX_SEQ_LEN = 8                # 公式最大长度
COST_RATE = 0.0005             # 交易费率
```

---

### 系统 B: 加密货币量化交易系统

完整的实盘交易系统，包含数据采集、模型训练、回测、执行和可视化。

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 配置 PostgreSQL

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# macOS
brew install postgresql
brew services start postgresql

# 创建数据库
createdb crypto_quant

# 或使用SQL
psql -U postgres
CREATE DATABASE crypto_quant;
\q
```

#### 3. 配置环境变量

```bash
cp .env.example .env
nano .env
```

**必填项:**
- `BIRDEYE_API_KEY`: 从 https://birdeye.so/ 获取（需付费订阅）
- `DB_PASSWORD`: PostgreSQL 数据库密码

**可选项:**
- `SOLANA_RPC_URL`: Solana RPC节点（实盘交易需要）
- `SOLANA_PRIVATE_KEY`: 钱包私钥（实盘交易需要）

#### 4. 运行系统

##### 步骤1: 数据采集

```bash
python -m data_pipeline.run_pipeline
```

这将:
- 从 Birdeye API 获取 Solana 链上的代币数据
- 筛选符合条件的代币（流动性 > $500K，FDV > $10M）
- 下载1分钟K线数据（最近7天）
- 存储到 PostgreSQL 数据库

##### 步骤2: 训练模型

```bash
python -m model_core.engine
```

这将:
- 从数据库加载历史数据
- 使用强化学习训练 AlphaGPT 模型
- 自动生成交易因子
- 保存最优模型到 `best_model.pth`

##### 步骤3: 启动可视化Dashboard

```bash
streamlit run dashboard/app.py
```

访问 `http://localhost:8501` 查看:
- 实时净值曲线
- 持仓分析
- 因子表达式
- 风险指标

##### 步骤4: 实盘交易（可选）

```bash
# 配置钱包私钥后运行
python -m execution.trader
```

**⚠️ 警告**: 实盘交易有风险，建议先用小资金测试！

---

## 🏗️ 系统架构

```
AlphaGPT/
│
├── code/                    # A股回测系统（独立）
│   └── main.py             # 完整的A股策略代码
│
├── data_pipeline/          # 数据采集模块
│   ├── config.py           # 配置
│   ├── data_manager.py     # 数据管理器
│   ├── db_manager.py       # 数据库操作
│   ├── fetcher.py          # 数据抓取
│   ├── processor.py        # 数据处理
│   ├── providers/          # 数据源
│   │   ├── birdeye.py      # Birdeye API
│   │   └── dexscreener.py  # DexScreener API
│   └── run_pipeline.py     # 入口脚本
│
├── model_core/             # 模型训练模块
│   ├── alphagpt.py         # AlphaGPT模型 + LoRD正则化
│   ├── config.py           # 配置
│   ├── data_loader.py      # 数据加载器
│   ├── engine.py           # 训练引擎
│   ├── backtest.py         # 回测引擎
│   ├── factors.py          # 因子库
│   ├── ops.py              # 算子定义
│   └── vm.py               # 虚拟机（执行因子公式）
│
├── execution/              # 交易执行模块
│   ├── config.py           # 配置
│   ├── jupiter.py          # Jupiter 聚合器
│   ├── rpc_handler.py      # Solana RPC交互
│   ├── trader.py           # 交易执行器
│   └── utils.py            # 工具函数
│
├── dashboard/              # 可视化Dashboard
│   ├── app.py              # Streamlit 主应用
│   ├── data_service.py     # 数据服务
│   └── visualizer.py       # 图表组件
│
├── lord/                   # LoRD正则化实验
│   └── experiment.py       # 低秩衰减实验
│
└── times.py                # 研究脚本（A股实验）
```

---

## 📚 使用指南

### A股回测系统

#### 修改交易标的

```python
# 编辑 code/main.py
INDEX_CODE = '000905.SH'  # 中证500
# INDEX_CODE = '000300.SH'  # 沪深300
# INDEX_CODE = '511260.SH'  # 十年国债ETF
```

#### 调整策略参数

```python
MAX_SEQ_LEN = 12           # 增加公式复杂度
TRAIN_ITERATIONS = 1000    # 更多训练轮次
BATCH_SIZE = 2048          # 增加批次大小（需要更多内存）
```

#### 解读结果

```
🎯 最终测试集夏普: 1.85
📈 年化收益率: 24.3%
📊 最大回撤: -8.7%
```

---

### 加密货币系统

#### 筛选代币条件

编辑 `data_pipeline/config.py`:

```python
MIN_LIQUIDITY_USD = 500000.0   # 最小流动性
MIN_FDV = 10000000.0           # 最小市值
MAX_FDV = 100000000.0          # 最大市值（避免蓝筹）
HISTORY_DAYS = 7               # 历史数据天数
```

#### 模型训练参数

编辑 `model_core/config.py`:

```python
BATCH_SIZE = 8192              # 批次大小
TRAIN_STEPS = 1000             # 训练步数
MAX_FORMULA_LEN = 12           # 公式长度
BASE_FEE = 0.005               # 交易费率 0.5%
```

#### 启用 LoRD 正则化

```python
# 编辑 model_core/engine.py
engine = AlphaEngine(
    use_lord_regularization=True,
    lord_decay_rate=1e-3,         # 正则化强度
    lord_num_iterations=5          # Newton-Schulz迭代次数
)
```

---

## ❓ 常见问题

### Q1: 如何获取 Birdeye API Key?

访问 https://birdeye.so/ 注册账号，选择付费套餐。**免费版不支持1分钟K线数据**。

### Q2: 数据库连接失败？

```bash
# 检查 PostgreSQL 是否运行
sudo systemctl status postgresql

# 测试连接
psql -U postgres -h localhost -p 5432 -d crypto_quant
```

### Q3: GPU 加速？

```bash
# 检查 CUDA
nvidia-smi

# 安装 PyTorch GPU 版本
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Q4: 回测效果不好？

可能原因:
- 训练轮次不足（增加 `TRAIN_STEPS`）
- 数据质量问题（检查数据完整性）
- 过拟合（减少 `MAX_FORMULA_LEN`）
- 市场环境变化（策略需要定期重训练）

### Q5: Tushare 积分不足？

访问 https://tushare.pro/ 参与社区活动获取积分，或考虑付费订阅。

### Q6: 实盘交易如何防止滑点？

- 使用付费RPC节点（Helius/QuickNode）
- 增加 Jito MEV 保护
- 减小单笔交易金额
- 避免流动性不足的代币

---

## 🔐 安全提示

1. **永远不要上传 `.env` 文件到 GitHub**
2. **私钥使用硬件钱包或冷钱包**
3. **实盘前先用小资金测试**
4. **定期监控账户余额和持仓**
5. **设置止损机制**

---

## 📞 联系支持

- Email: imbue2025@outlook.com
- QQ 群: 838641831
- GitHub Issues: https://github.com/imbue-bit/AlphaGPT/issues

---

## ⚖️ 免责声明

本项目仅用于研究和教育目的。使用本代码进行实盘交易的风险由使用者自行承担。作者不承担任何直接或间接的损失责任。

量化策略一旦公开，其有效性会快速衰减。建议在原有基础上进行创新和改进。
