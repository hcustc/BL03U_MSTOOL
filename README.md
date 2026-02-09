# BL03U_MassSpectrumTool

BL03U 质谱数据处理工具（PyQt 桌面 GUI），支持单谱/累加谱展示、拟合、寻峰和导出。

## 技术栈

- Python 3.10+
- PyQt6 + pyqtgraph
- NumPy / Pandas / SciPy / scikit-learn
- PyWavelets

## 当前目录结构（核心）

```text
.
├── main.py                         # 根入口（python main.py）
├── src/
│   └── bl03u_massspec/
│       ├── app_main.py             # GUI 启动
│       ├── massspec_func.py        # 主业务逻辑
│       ├── massspec.py             # UI 定义
│       ├── FindPeak.py             # 寻峰逻辑
│       └── data_processor.py       # 数据处理
├── data/
│   └── raw/
│       └── C6F11O2H/
│           └── 11.5eV/
│               └── C23072502-0000.txt  # 最小示例数据
├── tests/
├── requirements.txt
├── requirements-dev.txt
└── pytest.ini
```

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python main.py
```

## 数据约定

- 输入文件默认 `.txt`，按现有逻辑跳过前 10 行头信息。
- 仓库仅保留最小示例数据：`data/raw/C6F11O2H/11.5eV/C23072502-0000.txt`。
- 完整原始数据建议存放在仓库外（共享盘/对象存储/本地数据盘）。
- 生成物与缓存（`__pycache__`、临时图、IDE 文件等）通过 `.gitignore` 排除。

## 如何使用（GUI）

1. 启动程序：

```bash
python main.py
```

2. 查看单谱：
- 切到“单谱”页签。
- 在“文件地址”输入单个数据文件路径，例如：  
  `data/raw/C6F11O2H/11.5eV/C23072502-0000.txt`
- 点击“查看单谱”。

3. 查看累加谱：
- 切到“累加谱”页签。
- 在“文件夹地址”输入目录路径，例如：  
  `data/raw/C6F11O2H/11.5eV`
- 点击“查看累加谱”。

4. 自动寻峰与导出：
- 点击“自动寻峰”生成峰表。
- 可在表格中调整峰信息后导出。
- 图和表可按界面提示保存到本地文件。

## 命令行使用（可选）

对同一目录数据进行脚本处理：

```bash
python -m src.bl03u_massspec.FindPeak data/raw/C6F11O2H/11.5eV
python -m src.bl03u_massspec.caculate data/raw/C6F11O2H/11.5eV
```

## 开发检查（可选）

```bash
python -m compileall -q .
pytest -q
```

CI 会执行相同检查：`.github/workflows/ci.yml`。
