# Live Trading Project

独立实盘项目入口（仅策略自动交易）。

## 运行

```bash
cd projects/live_project
pip install -r requirements.txt
streamlit run app.py --server.port 8502
```

## 目录

- `strategy_files/`：策略文件（可独立迭代）
- `reports/`：操作日志、UI状态等输出
- `config/apis.toml`：OKX / Email 等配置
