# ACC / CRIL workflow

独立可运行的 GitHub 代码文件夹。包含正常场景及 BA、DPA、MA、PA、FA 五种攻击；不依赖上级项目的数据文件。全部目录、代码文件和输出文件均使用英文名称，数据标签按要求使用“正常”“异常”。

## Run

建议使用 Python 3.11。在本文件夹打开终端：

```bash
python -m pip install -r requirements.txt
python run.py
```

默认从仿真开始依次运行五个场景，各独立训练修复五次，五次全部计入均值与标准差，不筛选。运行单个场景：

```bash
python run.py --scenario fa
```

仅运行到异常标记和训练样本导出：

```bash
python run.py --prepare-only
```

每次生成新的 `outputs/<timestamp>/`，不会覆盖以前的结果。所有输入由入口自动传递，无需手动修改脚本中的路径。`--output <new_directory>` 可指定尚不存在的结果目录。

## Ordered stages

| Order | Code | Behavior |
|---|---|---|
| 01 | `stages/step01_simulate.py`, `simulators/` | 正常场景导出数组编号1的ACC及编号0的前车；攻击场景仅保存ACC。保留原数值仿真，删除绘图和与导出无关的车辆位置计算。 |
| 02 | `stages/step02_trim.py` | 丢弃Time 0–1999，保留2000–4799，共2800步；Time不重新编号。 |
| 03 | `stages/step03_beta.py` | 计算正常及攻击的三个beta；正常区间取正常截取数据全部2800行的min/max，不再二次截取。 |
| 04 | `stages/step04_label.py` | **只用beta1、beta3**：任一小于下界或大于上界即异常，边界值为正常。beta2仅计算存档，不参与标记。 |
| 05 | `stages/step05_repair_core.py`, `run.py` | 仅以该攻击文件中Time<3600且标签正常的行训练；全部异常行作为修复输入；独立训练五次并分别保存。 |
| 06 | `stages/step06_metrics.py` | 以Time逐行匹配正常ACC，计算速度、间距MAE/RMSE/MAPE及五次均值、总体标准差（ddof=0）。 |

## Parameters and preserved behavior

- 仿真原有 `np.random.seed(42)` 保留，每个场景开始时设置。车辆数组编号0是前车，编号1是ACC；6辆车，4800步，dt=0.1秒。攻击窗口为 `3600 <= Time < 3900`。
- ACC与beta参数一致：v0=39.12、tau=0.96、s0=10、delta=20、a=1.18、b=8.9。人驾车辆参数保留原仿真 `[30, 1.26, 3.4, 4, 1.06, 2]`。
- 预处理beta3保留此前指定的 **tau - Relative_speed/(2*sqrt(a*b))**，没有采用之前试验过的加号版。
- 原始轨迹先保留3位小数再进入后续处理，beta保留6位小数，边界基于这些beta计算。
- 核心修复类、损失、网络结构、随机噪声、两组GAIL训练与轨迹递推函数，与 `代码整理/6-2. CRIL五次，每次都保存.py`（对应指定的“1.1 CH-GAIL五次，每次都保存”）保持相同的函数语法树，仅移除未调用的Generator、旧路径、绘图及无后续用途的修复后beta检查。
- 每次修复中，两组GAIL各训练60轮，最终使用带状态噪声的第二组policy；每个场景重新初始化五次。默认 `config.json` 的 `training_seed` 为null，保持原训练脚本不显式固定种子的逻辑。需要复现某次指定种子时可填写整数；种子在该场景五次训练开始前只设置一次，不在每次修复时重复设置。这里没有沿用旧流程筛选出的场景种子。

**原修复递推仍有以下约定，整理时未改算法：** 前车输入为正常场景裁剪后的前车速度，修复函数按异常行序号取前车速度，而不是按异常Time查找；不连续异常行也按一个序列连续递推；输出相对速度使用前车减自身，而仿真输入相对速度为自身减前车。这些是原代码行为，指标阶段则始终按真实Time匹配。若要改成按Time取前车或分段修复，应作为单独的方法改动，不能视为纯整理。

## Outputs

正常目录 `normal/`：

- `01_target_raw.txt`, `01_leader_raw.txt`：4800步原始数据。
- `02_target_trimmed.txt`, `02_leader_trimmed.txt`：裁剪后的2800步数据。
- `03_target_beta.txt`, `04_beta_bounds.txt`：三个beta及其正常范围。

每个攻击目录 `ba/`, `dpa/`, `ma/`, `pa/`, `fa/`：

- `01_target_raw.txt` → `02_target_trimmed.txt` → `03_target_beta.txt` → `04_target_labeled.txt`。
- `05_training_normal_before_attack.txt`：真正进入训练的行。
- `05_abnormal_input.txt`：真正进入修复的异常行。
- `06_repaired_run_01.txt` 至 `06_repaired_run_05.txt`：分别保存全部异常行的原值、修复值和原始Time；对应 `06_policy_run_*.pt` 保存最终网络权重。
- `07_metrics_each_run.txt`：五次各自指标；`08_metrics_mean_std.txt`：五次汇总。
- `sample_counts.json`：训练、异常和评价样本数。

总目录还有 `config_used.json` 和 `09_all_metrics_mean_std.txt`。

## Evaluation interval

沿用原指标脚本：**Time 3600–3900，两端包含，仅评价该区间内实际输出的异常修复行**，不是强制300行或301行。正常参考来自本次仿真的正常ACC，不使用其他历史文件。MAPE分母为正常值+1e-6，结果为百分数。无评价行时不会生成虚构的误差值。若某次训练输出非有限值，逐次指标明确记录invalid及原因，继续其余运行；五次不是全部有效时，汇总标记incomplete_five_runs，均值和标准差留空，不剔除失败运行后冒充五次均值。

本次按要求改为仅裁剪一次、训练限于攻击前正常样本，因此不应将旧流程的数值直接当作本版结果。

## Verification

整理时检查了六个场景的速度、相对速度、间距和前车速度，与整理前仿真逐元素一致；核对有效修复类和递推函数的语法树一致。检查beta1/beta3联合判断、beta2不参与、攻击前训练筛选及乱序参考数据的Time对齐。`python run.py --scenario fa --smoke` 可用1轮训练走完五次修复和指标输出，仅供流程检查，不能当作正式实验指标。正式运行不加 `--smoke`。

## Optional fixed 300-step evaluation

不重新训练，使用已有五次修复结果，在连续300步内把异常行替换为修复值，其他行保留攻击轨迹原值，所有行均与正常参考按Time对齐。默认区间3600–3899，不改变入口原有的异常行评价方式：

```bash
python evaluate_fixed_window.py outputs/<timestamp> --scenario dpa
```

结果写入该场景的 `fixed_window_3600_3899/`，包括五份实际参与评价的300步轨迹和指标。目标目录必须不存在，避免覆盖已有实验。

## Use collected TXT files instead of simulation

可用三份未裁剪、Time从0开始的文件替代仿真输入：

```bash
python run.py --scenario fa --normal-target /path/normal_target.txt --normal-leader /path/normal_leader.txt --attack-target /path/fa_target.txt
```

必须同时提供三个文件并指定单个场景。正常前车与跟车的Time需一致，正常数据需覆盖攻击数据的Time；长度可以不同。分别删除前2000步，正常范围使用正常文件全部剩余数据，不默认截短到攻击文件长度。记录原始文件路径及SHA256。无异常行时跳过训练修复，仅生成空修复文件；此时固定窗口指标反映未修复攻击轨迹，五份一致结果不代表五次训练。
