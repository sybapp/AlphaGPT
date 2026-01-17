#!/bin/bash
# AlphaGPT 一键部署脚本
# 用途：自动化环境配置和依赖安装

set -e  # 遇到错误立即退出

echo "🚀 开始部署 AlphaGPT..."

# 检查 Python 版本
echo "📌 检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ 错误: 需要 Python 3.10+，当前版本: $python_version"
    exit 1
fi
echo "✅ Python 版本检查通过: $python_version"

# 创建虚拟环境
echo "📦 创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ 虚拟环境创建成功"
else
    echo "⚠️  虚拟环境已存在，跳过创建"
fi

# 激活虚拟环境
source venv/bin/activate

# 升级 pip
echo "⬆️  升级 pip..."
pip install --upgrade pip

# 安装核心依赖
echo "📥 安装核心依赖..."
pip install -r requirements.txt

# 询问是否安装可选依赖
read -p "是否安装可选依赖 (用于A股回测和实验)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 安装可选依赖..."
    pip install -r requirements-optional.txt
fi

# 创建 .env 文件
if [ ! -f ".env" ]; then
    echo "📝 创建 .env 配置文件..."
    cp .env.example .env
    echo "⚠️  请编辑 .env 文件，填入你的API密钥和数据库配置"
else
    echo "⚠️  .env 文件已存在，请手动检查配置"
fi

# 检查 PostgreSQL
echo "🔍 检查 PostgreSQL..."
if command -v psql &> /dev/null; then
    echo "✅ PostgreSQL 已安装"
    read -p "是否需要初始化数据库? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "请输入数据库名称 [crypto_quant]: " db_name
        db_name=${db_name:-crypto_quant}

        echo "正在创建数据库: $db_name"
        createdb $db_name || echo "⚠️  数据库可能已存在"

        echo "✅ 数据库初始化完成"
    fi
else
    echo "⚠️  未检测到 PostgreSQL，请手动安装"
    echo "   Ubuntu/Debian: sudo apt install postgresql postgresql-contrib"
    echo "   macOS: brew install postgresql"
fi

echo ""
echo "🎉 部署完成！"
echo ""
echo "📋 下一步操作："
echo "1. 编辑 .env 文件，填入API密钥"
echo "2. 选择运行模式："
echo ""
echo "   【A股回测模式】"
echo "   python code/main.py"
echo ""
echo "   【加密货币模式】"
echo "   # 步骤1: 采集数据"
echo "   python -m data_pipeline.run_pipeline"
echo "   "
echo "   # 步骤2: 训练模型"
echo "   python -m model_core.engine"
echo "   "
echo "   # 步骤3: 启动Dashboard"
echo "   streamlit run dashboard/app.py"
echo ""
echo "3. 查看文档: cat DEPLOYMENT.md"
echo ""
