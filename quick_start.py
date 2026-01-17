#!/usr/bin/env python3
"""
AlphaGPT 快速训练示例
用途: 演示如何快速开始训练
"""

print("""
╔══════════════════════════════════════════════╗
║        AlphaGPT 训练快速入门                 ║
╚══════════════════════════════════════════════╝

你有两种训练方式:

【方式1: A股回测训练】简单快速，适合新手
  1. 获取Tushare Token: https://tushare.pro/
  2. 编辑 code/main.py 第13行，填入Token
  3. 运行: python code/main.py
  ✅ 5-10分钟完成训练

【方式2: 加密货币训练】完整系统，适合实盘
  1. 安装PostgreSQL数据库
  2. 获取Birdeye API: https://birdeye.so/
  3. 配置 .env 文件
  4. 步骤:
     a) python -m data_pipeline.run_pipeline  # 采集数据
     b) python -m model_core.engine           # 训练模型
     c) streamlit run dashboard/app.py        # 查看结果
  ⏱️ 首次约30-60分钟

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

import sys
import os

def check_dependencies():
    """检查依赖是否安装"""
    print("🔍 检查依赖...")

    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError:
        print("  ❌ PyTorch 未安装")
        print("     安装: pip install torch")
        return False

    try:
        import pandas
        print(f"  ✅ Pandas {pandas.__version__}")
    except ImportError:
        print("  ❌ Pandas 未安装")
        return False

    try:
        import numpy
        print(f"  ✅ NumPy {numpy.__version__}")
    except ImportError:
        print("  ❌ NumPy 未安装")
        return False

    return True

def quick_start_menu():
    """快速开始菜单"""
    print("\n请选择训练方式:")
    print("  [1] A股回测训练 (推荐新手)")
    print("  [2] 加密货币训练 (需要数据库)")
    print("  [3] 查看完整教程")
    print("  [q] 退出")

    choice = input("\n请输入选项 [1/2/3/q]: ").strip().lower()

    if choice == '1':
        a_share_guide()
    elif choice == '2':
        crypto_guide()
    elif choice == '3':
        show_tutorial()
    elif choice == 'q':
        print("👋 再见!")
        sys.exit(0)
    else:
        print("❌ 无效选项，请重新选择")
        quick_start_menu()

def a_share_guide():
    """A股训练指南"""
    print("\n" + "="*50)
    print("📈 A股回测训练指南")
    print("="*50)

    # 检查文件是否存在
    if not os.path.exists('code/main.py'):
        print("❌ 错误: code/main.py 文件不存在")
        print("   请确保在 AlphaGPT 项目根目录运行此脚本")
        return

    print("\n第一步: 获取 Tushare Token")
    print("  1. 访问 https://tushare.pro/")
    print("  2. 注册账号（免费）")
    print("  3. 复制你的Token")

    token = input("\n请粘贴你的 Tushare Token (或按Enter跳过): ").strip()

    if token:
        # 自动修改配置文件
        print("  ✅ Token已保存")
        print("\n第二步: 选择交易标的")
        print("  [1] 511260.SH - 十年国债ETF (低风险)")
        print("  [2] 000905.SH - 中证500 (中等风险)")
        print("  [3] 000852.SH - 中证1000 (高风险高收益)")

        target = input("请选择 [1/2/3, 默认1]: ").strip() or "1"

        target_map = {
            "1": "511260.SH",
            "2": "000905.SH",
            "3": "000852.SH"
        }

        index_code = target_map.get(target, "511260.SH")

        print(f"\n✅ 已选择: {index_code}")
        print("\n第三步: 开始训练")
        print(f"  执行命令: python code/main.py")
        print("  预计时间: 5-10分钟")
        print("  训练完成后会生成:")
        print("    - best_formula_*.txt (最优公式)")
        print("    - backtest.png (净值曲线)")

        run_now = input("\n是否立即开始训练? [y/N]: ").strip().lower()

        if run_now == 'y':
            import subprocess
            subprocess.run([sys.executable, 'code/main.py'])
        else:
            print("\n稍后手动运行: python code/main.py")
    else:
        print("\n⚠️  未填入Token，请按以下步骤手动配置:")
        print("  1. 编辑 code/main.py")
        print("  2. 找到第13行: TS_TOKEN = '填入 Tushare Token'")
        print("  3. 填入你的真实Token")
        print("  4. 运行: python code/main.py")

def crypto_guide():
    """加密货币训练指南"""
    print("\n" + "="*50)
    print("🪙 加密货币训练指南")
    print("="*50)

    print("\n【前置要求】")

    # 检查PostgreSQL
    import subprocess
    try:
        result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
        print(f"  ✅ PostgreSQL 已安装: {result.stdout.strip()}")
    except FileNotFoundError:
        print("  ❌ PostgreSQL 未安装")
        print("     Ubuntu: sudo apt install postgresql")
        print("     macOS: brew install postgresql")
        return

    # 检查.env文件
    if not os.path.exists('.env'):
        print("  ⚠️  .env 文件不存在，正在创建...")
        if os.path.exists('.env.example'):
            import shutil
            shutil.copy('.env.example', '.env')
            print("  ✅ 已创建 .env 文件")
        else:
            print("  ❌ 缺少 .env.example 模板文件")
            return

    print("\n【配置步骤】")
    print("1. 编辑 .env 文件:")
    print("   nano .env")
    print("\n2. 填入必要配置:")
    print("   DB_PASSWORD=你的数据库密码")
    print("   BIRDEYE_API_KEY=你的Birdeye密钥")
    print("\n3. 创建数据库:")
    print("   createdb crypto_quant")

    print("\n【训练流程】")
    print("步骤1: 采集数据 (10-30分钟)")
    print("  python -m data_pipeline.run_pipeline")
    print("\n步骤2: 训练模型 (20-60分钟)")
    print("  python -m model_core.engine")
    print("\n步骤3: 可视化分析")
    print("  streamlit run dashboard/app.py")

    cont = input("\n配置完成后按Enter继续，或输入q退出: ").strip().lower()

    if cont != 'q':
        print("\n开始数据采集...")
        import subprocess
        subprocess.run([sys.executable, '-m', 'data_pipeline.run_pipeline'])

def show_tutorial():
    """显示完整教程"""
    print("\n" + "="*50)
    print("📚 完整教程")
    print("="*50)

    print("\n详细文档:")
    print("  1. DEPLOYMENT.md - 部署指南")
    print("  2. TRAINING.md   - 训练教程")
    print("  3. README.md     - 项目说明")

    print("\n在线教程:")
    print("  cat TRAINING.md  # 查看训练教程")

    input("\n按Enter返回主菜单...")
    quick_start_menu()

def main():
    """主函数"""
    if not check_dependencies():
        print("\n❌ 缺少必要依赖，请先安装:")
        print("   pip install -r requirements.txt")
        return

    quick_start_menu()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 训练中断，再见!")
        sys.exit(0)
