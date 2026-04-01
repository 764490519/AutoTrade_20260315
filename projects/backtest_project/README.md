# Backtest Project

独立回测项目入口（仅回测与优化）。

## 运行

```bash
cd projects/backtest_project
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

## 目录

- `strategy_files/`：策略文件（可独立迭代）
- `reports/`：回测、优化、UI状态等输出
- `config/apis.toml`：可选配置（通常回测不需要）
