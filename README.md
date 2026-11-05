# RICE-N：RLlib CPU 独立版

本项目模拟多个地区的储蓄、减排、贸易与气候谈判，并使用共享策略的多智能体强化学习训练决策。仅保留 CPU 路径，支持 Windows PowerShell 与 uv。

## 与原版的区别

- 保留 `rice.py`、`rice_helpers.py` 中的模型方程、地区参数、地区字典排序和效用奖励。
- Gym 改为 Gymnasium；适配新版 RLlib 的 reset/step 接口。
- 原版 Ray 1.0/A2C 改为固定 Ray 2.40/PPO，网络采用两层 128 单元 Tanh，保留动作掩码。因此本版不是原版 A2C 训练结果的直接复现。
- 所有地区共享一个策略网络，各地区观察中包含地区标识。
- 删除 CUDA、WarpDrive、Docker、旧 Notebook、竞赛提交与远程测试下载流程。

## 一键跑完整流程

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -RunDir runs/quickstart -Iterations 1
```

依次安装环境、测试、短训练、从磁盘加载评估、绘图。正式训练可将 `-Iterations` 改为 100 或更多。输出目录必须尚不存在。

## 1. 创建本地环境

在项目根目录运行（目录有空格，请使用引号）：

```powershell
Set-Location -LiteralPath 'F:\BaiduSyncdisk-mu\Project ITP\tool-for-scientific-research\1_RICE_N_original\climate-cooperation_rice'
uv sync --locked
```

uv 自动选择 Python 3.10，并创建项目内 `.venv`。Torch 从官方 CPU 源安装；无需 CUDA。无需手动激活环境，后续命令使用 `uv run`。首次安装需要联网。

## 2. 测试模型

```powershell
uv run pytest -q
```

测试分别跑完关闭谈判的 20 步与开启谈判的 60 步，并检查动作约束、观察空间、奖励有限性与结束年份。

## 3. 训练

先做一次短训练，确认流程：

```powershell
uv run python -m rllib_cpu.train --iterations 1 --output runs/demo
```

正式训练：

```powershell
uv run python -m rllib_cpu.train --config configs/cpu.yaml --output runs/experiment01
```

每次使用新的输出目录，避免覆盖已有实验。默认 100 次 PPO 迭代；`--iterations` 可覆盖配置。所有环境采样在本地进程运行，GPU 数量固定为 0。

## 4. 重新加载与评估

训练结束后，可关闭终端，再在新终端进入项目执行：

```powershell
uv run python -m rllib_cpu.evaluate --run runs/demo --episodes 1
```

这会从磁盘重新加载完整 RLlib checkpoint，再运行环境；不会继续训练。默认选择确定性动作。多次确定性评估通常得到相同轨迹；需要策略采样差异时使用：

```powershell
uv run python -m rllib_cpu.evaluate --run runs/demo --episodes 5 --explore
```

重复评估会覆盖该目录的评估表和 summary。checkpoint 采用运行目录下的相对位置，整个运行文件夹可移动。仅加载自己信任的模型文件。

## 5. 绘图分析

```powershell
uv run python -m rllib_cpu.plot --run runs/demo
```

输出：

| 文件 | 内容 |
|---|---|
| config.yaml | 本次训练的实际配置 |
| checkpoint/ | 完整 RLlib 模型与优化器检查点 |
| checkpoint_path.txt | 保存时的检查点位置，仅供参考 |
| training.csv | 每次训练的环境步数与平均回合奖励 |
| evaluation.csv | 各年份全球气温、碳、产出、消费、减排、福利 |
| regions.csv | 各地区逐年经济、人口、资本、效用、减排 |
| trajectory_000.npz | 包含谈判步骤和初始状态的原始数组 |
| summary.json | 评估摘要与内部地区 ID 对应的参数文件顺序 |
| analysis.png | 全球气候经济六面板图 |
| regional_mitigation.png | 各地区减排率曲线 |
| training.png | 训练奖励曲线 |

评估表只保留初始状态与真正推进经济年份的步骤，避免把谈判步骤当作新增年份。初始时刻的产出、消费等流量由原模型初始化为零。温度表示模型温度异常值；经济量保留模型单位，不自行转换为现实货币。

短训练只证明软件流程可运行，不证明策略已收敛。比较谈判制度时，应分别训练、使用多个随机种子，并保持训练预算一致；不得仅凭单次奖励曲线得出合作机制有效的结论。

## 配置

编辑 `configs/cpu.yaml`：

- `seed`：训练随机种子。
- `iterations`：训练迭代数。
- `num_cpus`：Ray CPU 资源和 Torch 线程数。
- `env.negotiation_on`：是否开启“提案→接受/拒绝→经济气候模拟”。
- `env.num_discrete_action_levels`：动作离散档位。
- `training.train_batch_size`：每次训练的环境步数，所有地区共享策略。
- `training.minibatch_size`、`num_epochs`、`lr`、`gamma`：PPO 批量、更新轮数、学习率和折扣率。

`region_yamls/default.yml` 设置起始年份 2015、每经济步骤 5 年、20 个经济步骤。地区 `.yml` 覆盖默认参数。地区文件按字符串排序加载，因此内部 ID 不等于文件编号，不要重命名文件来重新排序。

## 文件结构与逻辑

```text
configs/cpu.yaml       训练与环境配置
region_yamls/          全球与地区校准参数
rice.py               原始气候经济与谈判模拟
rice_helpers.py       经济气候公式与参数读取
rllib_cpu/
  env.py              Gymnasium 多智能体环境适配
  model.py            共享 actor/critic 与动作掩码
  runtime.py          CPU Ray 运行时与 PPO 配置
  train.py            训练、指标记录与检查点保存
  evaluate.py         模型重新加载、轨迹导出与评估摘要
  plot.py             CSV 结果可视化
 tests/test_env.py     环境和谈判约束测试
run_all.ps1           Windows 一键完整运行
pyproject.toml        直接依赖与 CPU Torch 源
uv.lock               精确依赖锁定
```

调用流程：参数 → Rice.reset → 地区观察 → 共享策略选择动作 → Rice.step → 更新经济/贸易/排放/气候 → 地区效用奖励 → PPO 更新。实际奖励为当期地区效用；折现社会福利另行记录。

## 来源与许可证

基于 MILA / Salesforce 的 RICE-N climate-cooperation-competition。原始许可证及引用信息保存在 `LICENSE.txt` 与 `CITATION.cff`。原版文件可通过 Git 的原分支查看；当前分支是 CPU 整理版。

本机验证记录见 [docs/verification.md](docs/verification.md)。
