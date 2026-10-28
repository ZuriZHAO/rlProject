# Multi-Taxi RL 项目文档

## 项目概览

本项目实现了一个**网约车调度强化学习环境**，当前版本为**多车多客基线**，使用 **n-Step SARSA + 势能奖励整形（PBRS）** 算法进行训练。

当前实现已经支持：
- **多辆出租车 + 多个乘客** 的任务场景
- 每辆车返回**独立局部观测**
- 环境 `step()` 接收 `action_dict`，并返回多智能体的 `rewards / dones / infos`
- 训练采用 **Independent Learners**：每辆车各自维护一张 Q-table 进行学习

队友接手扩展时，主要工作是系统调参并分析规律。

---

## 文件结构

```
multi_taxi_rl/
├── multi_taxi_env.py   # 核心环境类（GridWorld + 三大机制）
├── train.py            # 训练脚本（n-Step SARSA + PBRS）
├── test_env.py         # 环境快速验证脚本
└── qtables.pkl         # 训练后保存的 Q-table（运行后生成）
```

---

## 快速开始

```bash
# 安装依赖（仅需 numpy）
pip install numpy

# 验证环境是否正常
python test_env.py

# 训练（默认 20000 回合）
python train.py

# 自定义回合数
python train.py --episodes 6000

# 自定义车辆数与乘客数
python train.py --n_taxis 2 --n_passengers 3

# 开启渲染（每 5% 回合打印地图）
python train.py --episodes 1000 --render

# 自定义最大步数
python train.py --max_steps 300

# 自定义车辆重叠惩罚
python train.py --overlap_penalty 4.0

# 固定随机种子，便于复现实验
python train.py --seed 42
```

---

## 一、环境文件 `multi_taxi_env.py`

### 1.1 地图设计（7×7 Grid World）

```
列:  0  1  2  3  4  5  6
行0 [★  ═  ═  ═  ═  ═  ★]   ← 顶部主干道 + CBD
行1 [·  ·  █  ·  ·  █  ·]
行2 [·  ·  █  ·  ·  █  ·]
行3 [·  ·  ·  ★  ·  ·  ·]   ← 中心 CBD
行4 [·  ·  █  ·  ·  █  ·]
行5 [·  ·  █  ·  ·  █  ·]
行6 [★  ═  ═  ═  ═  ═  ★]   ← 底部主干道 + CBD
```

| 符号 | 常量           | 含义           |
|------|----------------|----------------|
| `·`  | `CELL_ROAD=0`  | 普通道路，移动 -1 奖励 |
| `█`  | `CELL_WALL=1`  | 不可通行墙壁   |
| `═`  | `CELL_ARTERIAL=2` | 主干道，有拥堵风险 |
| `★`  | `CELL_HOTSPOT=3`  | CBD 繁华区，乘客高频出没 |

**设计原因**：地图包含墙壁将地图分为左右两个走廊，强迫智能体学会绕路。CBD 区域模拟现实中乘客集中在商圈的规律。

---

### 1.2 动作空间

| 动作编号 | 常量            | 含义     |
|----------|-----------------|----------|
| 0        | `ACTION_SOUTH`  | 向南移动 |
| 1        | `ACTION_NORTH`  | 向北移动 |
| 2        | `ACTION_EAST`   | 向东移动 |
| 3        | `ACTION_WEST`   | 向西移动 |
| 4        | `ACTION_PICKUP` | 接客     |
| 5        | `ACTION_DROPOFF`| 下客     |

---

### 1.3 三大核心机制

#### 机制 A：主干道拥堵博弈（Risk-Aware Routing）

```python
# 主干道移动时：
if bool(self.rng.binomial(1, self.congestion_prob)):
    reward = -10.0   # 堵车！位置不变，重罚
else:
    self.taxi_pos = (nr, nc)
    reward = -2.0    # 通过主干道，固定风险成本
```

- 主干道有 30%（`congestion_prob=0.3`）概率堵死
- 堵车时位置不变，扣 -10 分
- 即使通过，也比普通路多扣 1 分（-2 vs -1）
- **设计意图**：迫使智能体在「走捷径主干道但有风险」与「绕路普通道路但安全」之间做出决策

#### 机制 B：里程补贴动态奖励（Distance-Based Reward）

```python
# 接客时（ACTION_PICKUP 成功）：
self.passenger_dist = self._get_shortest_path_length(pickup_cell, dest_cell)
reward = 5.0 + self.reward_alpha * self.passenger_dist

# 送达时（ACTION_DROPOFF 成功）：
reward = self.base_reward   # 固定 20.0
```

- 接客时立即用 BFS 计算接客点到目的地的最短路径距离，距离越远奖励越高
- `reward_alpha=2.0`：每单位距离补贴 2 分
- **重要设计**：里程奖励在**接客时一次性发放**，而不是在送达时发放。这是为了保持**马尔可夫性**——送达奖励仅依赖当前状态（是否到达目的地），不依赖历史信息（当初接客在哪里）

#### 机制 C：繁华度加权生成（Hotspot Spawning）

```python
HOTSPOT_WEIGHT_FACTOR: float = 5.0

def _build_spawn_pool(self):
    # CBD 格子权重是普通格子的 5 倍
    raw = [5.0] * len(hotspot_cells) + [1.0] * len(normal_cells)
    return pool, raw / raw.sum()
```

- 乘客出现位置按加权概率采样，CBD 格子概率是普通格子的 5 倍
- **设计意图**：模拟现实中乘客集中在商圈的规律，增加任务难度和现实性

---

### 1.4 类构造函数 `__init__`

```python
MultiTaxiEnv(
    n_taxis=1,              # 环境类默认 1；训练脚本默认会传入 2
    n_passengers=1,         # 环境类默认 1；训练脚本默认会传入 3
    congestion_prob=0.3,    # 主干道堵车概率
    reward_alpha=2.0,       # 里程补贴系数
    base_reward=20.0,       # 单个乘客送达基础奖励
    max_steps=200,          # 每回合最大步数，超时结束
    overlap_penalty=4.0,    # 多车重叠惩罚，鼓励车辆分散
    seed=None,              # 随机种子
)
```

初始化时会自动：
1. 构建地图 `self.grid`（7×7 NumPy 数组）
2. 提取所有可通行格子 `self._walkable`（列表）和反查字典 `self._walkable_idx`
3. 提取主干道格子集合 `self._arterial_cells`（set，O(1) 查询）
4. 构建乘客生成池 `self._spawn_pool` 和权重 `self._spawn_weights`
5. 预计算所有可通行格子之间的 BFS 最短路距离表 `self._dist_table`
6. 初始化多车重叠惩罚参数 `self.overlap_penalty`
---

### 1.5 关键内部数据结构

| 属性 | 类型 | 含义 |
|------|------|------|
| `self._walkable` | `List[Tuple[int,int]]` | 所有可通行坐标的有序列表 |
| `self._walkable_idx` | `Dict[Tuple,int]` | 坐标 → 索引的反查字典，O(1) |
| `self.taxi_positions` | `Dict[str, Tuple[int,int]]` | 每辆车当前所在位置 |
| `self.passenger_locs` | `List[int]` | 每个乘客当前位置：`>=0` 为 walkable 索引，`-1` 表示在车内，`-2` 表示已送达 |
| `self.passenger_dests` | `List[int]` | 每个乘客的目的地索引 |
| `self.passenger_dists` | `List[int]` | 每个乘客在接客时记录的 BFS 距离，用于里程补贴 |
| `self.passenger_owners` | `List[Optional[str]]` | 当前由哪辆车负责该乘客；未被接走时为 `None` |
| `self.taxi_carrying` | `Dict[str, Optional[int]]` | 每辆车当前载的是哪个乘客；空车为 `None` |
| `self.delivered_count` | `int` | 已送达乘客数量 |

---

### 1.6 函数详解

#### `reset() → Dict`

重置一个新回合。每次调用：
- `step_count` 归零
- 所有出租车随机放置在可通行格上，且初始位置不重复
- 多个乘客按加权概率生成，上车点尽量不重复
- 每个乘客的目的地随机采样，且保证不等于自己的上车点

**返回格式**：
```python
{
    "taxi_1": (
        taxi_row, taxi_col,
        goal_row, goal_col,
        has_passenger,
        need_diversion,
        demand_high
    ),
    "taxi_2": (...),
    ...
}
```

#### `_get_obs() → Tuple`

生成当前每辆车的观测（7 维元组）：
- `(taxi_row, taxi_col, goal_row, goal_col, has_passenger, need_diversion, demand_high)`

各字段含义如下：
- `taxi_row, taxi_col`：当前出租车自己的位置
- `goal_row, goal_col`
  - 空车时：当前联合分配给该车的待接乘客位置
  - 载客时：当前所载乘客的目的地
  - 没有可接乘客或全部完成后：退化为出租车自身位置
- `has_passenger`：当前车上是否有乘客
- `need_diversion`：最近乘客不是当前联合分配给自己的乘客，说明该车应该避开局部最近目标，减少多车抢同一乘客
- `demand_high`：当前待接乘客数量是否多于空车数量
这样设计的目的是在保持观测紧凑的前提下，引入少量协调信息，帮助多车场景下减少多辆车反复抢同一目标。 

#### `_build_distance_table() → np.ndarray`

预计算所有可通行格子之间的最短路径距离表：

```python
dist[i, j] = 从 _walkable[i] 到 _walkable[j] 的最短路长度
```

#### `step(action_dict) → (next_obs, rewards, dones, infos)`

执行一步，返回四个字典：

```python
next_obs = {"taxi_1": tuple, "taxi_2": tuple, ...}
rewards  = {"taxi_1": float, "taxi_2": float, ...}
dones    = {"taxi_1": bool, "taxi_2": bool, ..., "__all__": bool}
infos    = {"taxi_1": {...}, "taxi_2": {...}, ...}
```

**奖励结构(单车视角)**：

| 事件 | 奖励 |
|------|------|
| 普通移动成功 | -1.0 |
| 撞墙 / 越界 | -3.0（位置不变） |
| 主干道通过 | -2.0 |
| 主干道堵车 | -10.0（位置不变） |
| 接客成功 | +5 + 2×BFS距离 |
| 接客失败（位置错/已有乘客） | -2.0 |
| 送达成功 | +20.0 |
| 错误地点下客 | -10.0（乘客仍留在当前车上，智能体需要继续移动到正确目的地） |
| 多车重叠 | 每辆重叠车辆额外扣 `overlap_penalty`，默认 -4.0 |
| 超时惩罚（叠加在最后一步） | -5.0 |


**`dones["__all__"]`** 为 True 的条件：所有乘客都已送达，或超过 `max_steps` 步。

#### `render()`

打印当前地图到终端。当前实现使用带编号标记：
- `T1`, `T2`, ...：空车出租车
- `C1`, `C2`, ...：载客中的出租车
- `P1`, `P2`, ...：未上车乘客
- `D1`, `D2`, ...：尚未完成乘客的目的地

### 补充：多车多客版本新增的辅助函数

相比原来的单车单客版本，当前环境为了支持多车多客决策，新增了若干辅助函数，主要分为三类：

1. **距离查询相关**
   - `_build_distance_table()`
   - `_get_shortest_path_idx()`
   - `_get_shortest_path_length()`

2. **乘客分配相关**
   - `_get_waiting_passenger_ids()`
   - `_get_idle_taxi_ids()`
   - `_assign_idle_taxis_to_waiting_passengers()`
   - `_get_assigned_waiting_passenger()`
   - `_get_nearest_waiting_passenger()`

3. **轻量协作观测相关**
   - `_get_coordination_bits()`

其中 `_assign_idle_taxis_to_waiting_passengers()` 是当前协调逻辑的核心：它在空车和待接乘客之间做贪心最短路匹配，让每辆空车优先关注一个不同乘客，从而减少多车扎堆抢同一单。

这些函数共同支撑了当前版本的多车多客局部决策逻辑，也是后续继续调参和扩展协作机制时最值得重点阅读的部分。

---

### 1.7 状态空间分析

当前训练仍然使用 **Q-table + 稀疏存储**，并且采用 **Independent Learners** 的方式：每辆车只根据自己的局部观测进行学习。

因此，Q-table 的 key 不再是“全局联合状态”，而是每辆车自己的观测 tuple：

```python
(
    taxi_row, taxi_col,
    goal_row, goal_col,
    has_passenger,
    need_diversion,
    demand_high
)
```
---

## 二、训练文件 `train.py`

### 2.1 超参数说明

```python
DEFAULT_EPISODES        = 20000   # 训练总回合数
DEFAULT_N_STEPS         = 4       # n-Step SARSA 的 n
GAMMA                   = 0.97    # 折扣因子
ALPHA                   = 0.25    # 学习率
EPS_START               = 1.0     # 初始探索率
EPS_END                 = 0.10    # 最终探索率
EPS_DECAY               = 0.99985 # 每回合探索率衰减
PBRS_ETA                = 0.45    # PBRS 势能权重
DEFAULT_OVERLAP_PENALTY = 4.0     # 多车重叠惩罚
DEFAULT_MAX_STEPS       = 200     # 每回合最大步数
DEFAULT_N_TAXIS         = 2       # 默认车辆数
DEFAULT_N_PASSENGERS    = 3       # 默认乘客数
```

**η 的理论约束**（重要）：

PBRS 要保证「原地不动」不能获得正整形奖励，需要：

```
η < 1 / ((1-γ) × d_max)
```

本环境中当前使用 `GAMMA=0.97`，因此：

- `1-γ = 0.03`
- 如果估计 `d_max ≈ 20`
- 上界约为 `1 / (0.03 × 20) ≈ 1.67`

所以当前 `PBRS_ETA=0.45` 是比较保守的设置，能够避免势能奖励过强导致智能体通过无效行为获得异常正收益。

---

### 2.2 函数详解

#### `make_qtable() → defaultdict`

```python
return defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float64))
```

创建 Q-table：未见过的状态自动初始化为全 0 数组（6 个动作）。使用 `defaultdict` 的好处是不需要提前枚举所有状态，只有访问过的状态才占内存。

#### `legal_actions(env, agent_id) → List[int]`

返回当前智能体在当前状态下允许选择的动作集合。

当前合法动作规则：

- 移动动作：只有目标格子在地图范围内，且不是墙壁时才合法
- `PICKUP`：只有空车，并且当前位置有等待中的乘客时才合法
- `DROPOFF`：只有车上有乘客，并且当前位置是该乘客目的地时才合法

这个函数用于训练阶段的 action masking。也就是说，探索和利用时都不会再从全部 6 个动作里盲目选择，而是只从当前合法动作集合中选择。

#### `epsilon_greedy(q_table, state, epsilon, rng, legal) → int`

ε-greedy 策略，但加入了合法动作 masking：

- 以概率 ε 从 `legal` 合法动作集合中随机选动作
- 以概率 1-ε 只在合法动作中选择 Q 值最大的动作
- 非法动作的 Q 值会被临时设为 `-inf`，不会被选中
- 平局时随机选择，避免总是选同一个动作

这个改动可以减少多车多客环境中的无效探索，例如空车乱下客、非乘客位置乱接客、非目的地乱下客等。  

#### `compute_potential(env, agent_id, eta) → float`

PBRS 势能函数 Φ(s)：

```python
# 空车阶段：目标是联合分配给该车的待接乘客
Φ = -η × (dist_to_pickup + dist_to_dropoff)

# 载客阶段：目标是当前所载乘客的目的地
Φ = -η × dist_to_dest

# 全部送达后：势能 = 0
```

**为什么用两段距离之和？**
接客时，势能从「到接客点 + 接客点到目的地」切换到「到目的地」。如果只用「到接客点」，接客瞬间势能会从 -η×0 跳变到 -η×dist_to_dest，造成势能不连续，整形奖励产生巨大负脉冲，破坏 PBRS 的理论保证。

整形奖励公式：`F = γ·Φ(s') - Φ(s)`

靠近目标时，`Φ(s')` 绝对值更小（负数更小），所以 `F > 0`，给出正向引导。

**评估时传入 `eta=0.0`**：早返回 `0.0`，跳过 BFS 计算，加速评估。

#### `nstep_update(q_table, buffer, next_state, next_action, gamma, alpha, is_done)`

n-Step SARSA 的核心更新：

```
G = Σ_{i=0}^{n-1} γ^i × r̃_{t+i+1}  +  γ^n × Q(s_{t+n}, a_{t+n})
Q(s_t, a_t) ← Q(s_t, a_t) + α × [G - Q(s_t, a_t)]
```

参数：
- `buffer`：长度为 n 的 deque，存储 `(state, action, shaped_reward)` 三元组
- `is_done=True` 时，`G` 不加 Q 估计（回合已结束，没有未来）
- 每次调用从 buffer 取出最旧的一条进行更新（`buffer.popleft()`）

#### `run_episode(env, q_tables, epsilon, n, gamma, alpha, eta, rng, render) → (float, int, bool)`

执行一个完整回合，返回 `(累计原始奖励, 使用步数, 是否成功送达全部乘客)`。

**核心循环逻辑**：

```
reset → 选 action
循环:
    执行 action → 得到 raw_r、is_done
    计算整形奖励 shaped_r = raw_r + γ·Φ(s') - Φ(s)
    buffer 追加 (state, action, shaped_r)
    如果 buffer 满 n 步 或 is_done → 触发 nstep_update
    state ← next_state
    action ← next_action
    如果 is_done → 清空 buffer 剩余项 → break
```

**回合结束后清空缓冲区**：

```python
if is_done:
    while buffer:
        nstep_update(q_table, buffer, state, action, gamma, alpha, True)
```

此时传入的 `state/action` 是占位符，`is_done=True` 让函数内部不使用它们。

#### `train(args)`

训练主循环：
1. 创建多车多客环境
2. 为每辆车创建一张独立 Q-table
3. 每回合调用 `run_episode`
4. 每 5% 回合打印一次统计信息：
   - 当前 `ε`
   - 平均步数
   - 成功率
   - 平均回报
   - Q-table 总大小
5. 训练完成后，将所有车辆的 Q-table 一起保存到 `qtables.pkl`
6. 最后进行贪心评估并输出整体表现

**评估与训练隔离**：评估用独立的 env 和 rng，不影响训练随机序列，结果可复现。

---

### 2.3 训练命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--episodes` | 20000 | 训练总回合数 |
| `--n_steps` | 4 | n-Step SARSA 的 n |
| `--n_taxis` | 2 | 训练时使用的车辆数 |
| `--n_passengers` | 3 | 训练时使用的乘客数 |
| `--max_steps` | 200 | 每回合最大步数 |
| `--overlap_penalty` | 4.0 | 多车重叠时每辆车额外扣分 |
| `--seed` | 42 | 训练随机种子 |
| `--render` | 关闭 | 开启后每 5% 回合打印地图 |

---

## 三、验证文件 `test_env.py`

用于快速验证环境逻辑是否正常：

```python
env = MultiTaxiEnv(n_taxis=2, n_passengers=3, seed=42)
states = env.reset()
actions = [
    {"taxi_1": 0, "taxi_2": 1},
    {"taxi_1": 2, "taxi_2": 3},
    {"taxi_1": 4, "taxi_2": 5},
    ...
]
```

运行 `python test_env.py` 可以检查：
- 接客/下客逻辑是否触发
- 奖励数值是否正确
- `dones["__all__"]` 是否在正确时机为 True

---

## 四、算法原理总结

### 4.1 n-Step SARSA

标准 SARSA 每步更新，n-Step SARSA 积累 n 步后再更新，减少偏差、加快传播。
- n=1：等价于普通 SARSA
- n=∞：等价于 Monte Carlo
- n=4：在偏差-方差之间取平衡

### 4.2 PBRS（Potential-Based Reward Shaping）

论文：Ng et al. (1999) *Policy Invariance Under Reward Transformations*

核心结论：只要整形奖励满足 `F(s,a,s') = γ·Φ(s') - Φ(s)`，最优策略不变。

本项目的 Φ(s)：
- 用 BFS 距离而非曼哈顿距离（正确处理墙壁）
- 接客前后用连续的两段距离之和（保持势能连续性）
- η=0.5 满足理论约束（η < 1.0）

### 4.3 奖励马尔可夫性

**问题**：如果送达时发放「里程补贴 + 基础奖励」，那送达时的奖励取决于当时接客在哪里——这是隐藏的历史信息，违反马尔可夫性，Q-table 无法正确收敛。

**解决方案**：里程补贴在接客时一次性发放，送达奖励固定为 20，只依赖当前状态（是否在目的地）。

---

## 五、当前实现总结与后续调参建议

当前版本已经完成了从单车单客到**多车多客**的扩展，具体包括：

### 5.1 已完成的扩展

- 环境已支持多个出租车与多个乘客
- `step()` 已支持接收多智能体动作字典 `action_dict`
- 每辆车返回独立观测，并可获得各自的 `reward / info`
- 训练采用 **Independent Learners**，即每辆车各自维护一张 Q-table
- 空车目标改为基于最短路的联合分配乘客，减少多车抢同一单
- 观测中加入轻量协调信息：`need_diversion` 和 `demand_high`
- PBRS 势能函数已经适配多车多客，并与联合分配目标保持一致
- 训练阶段加入合法动作 masking，探索和利用时都只从合法动作中选择
- 环境加入多车重叠惩罚 `overlap_penalty`，鼓励车辆分散行动

### 5.2 下一位同学的主要任务

下一位同学的重点是做**系统调参与实验分析**。建议优先研究以下参数：

- `n-step` 的 `n`
- 学习率 `ALPHA`
- 折扣因子 `GAMMA`
- 探索率相关参数：`EPS_END`、`EPS_DECAY`
- PBRS 权重 `PBRS_ETA`
- 每回合最大步数 `max_steps`
- 多车重叠惩罚 `overlap_penalty`
- 不同 `n_taxis / n_passengers` 组合下的训练表现

### 5.3 建议重点观察的指标

训练和评估时，建议重点记录：

- 成功率（是否完成全部乘客送达）
- 平均步数
- 平均回报
- Q-table 大小
- 在不同车数 / 客数配置下的收敛速度

### 5.4 建议实验组织方式

建议一次只调整一个变量，其余参数固定。可以按下面顺序逐步实验：

1. 固定 `n_taxis=2, n_passengers=3`，先调 `n-step`
2. 在较优 `n-step` 下再调学习率 `ALPHA`
3. 再调探索率衰减参数 `EPS_END / EPS_DECAY`
4. 最后比较不同 `n_taxis / n_passengers` 配置下的表现差异

这样更容易看出每个参数对成功率、平均回报和训练稳定性的影响。

## 5.5 备注

- 训练结束后会将所有车辆的 Q-table 一起保存到 `qtables.pkl`
- 日志中会输出成功率、平均步数、平均回报和 Q-table 大小，便于后续调参比较
- 当前默认训练配置为 `2` 车 `3` 客，是后续实验的推荐起点
---

## 六、已知的设计权衡

| 设计 | 原因 |
|------|------|
| 错误地点下客时乘客仍留在车上 | 有意为之，避免一次错误下客就把任务状态错误地重置为空车 |
| 接客失败返回 -2（不是 -1） | 比普通移动更重的惩罚，迫使智能体精确导航 |
| 主干道失败位置不变 | 现实中堵死就是动不了，状态变化正确反映物理含义 |
| 评估独立 env/rng | 训练和评估完全隔离，防止随机序列相互干扰，结果可复现 |

---

## 七、依赖

```
Python >= 3.8
numpy
```

无需安装 gym 或其他 RL 框架，环境完全自定义实现。
