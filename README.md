# nb2reg — 计数测量的 NB2 负二项过度离散拟合（离线、纯标准库）

一个零第三方依赖的可复用 Python 库，实现 **NB2 负二项回归**：

```
mu_i    = exp(x_i @ beta + offset_i)
y_i | mu_i, alpha ~ NB2,   Var(y_i) = mu_i + alpha * mu_i^2,   alpha > 0
```

联合最大似然估计 `(beta, log theta)`，其中 `theta = 1/alpha`（NB2 的
target/size 参数）。对数似然用 `math.lgamma`；`digamma`/`trigamma`
标准库没有，按递推式 + 渐近展开自行实现；对 `theta >> y`（近 Poisson
区）的灾难性抵消，使用整数 `y` 的精确连乘/调和和与解析差式规避。

## 运行环境

- **Python 3.14.7**（代码不使用 3.14 之外的特性，3.10+ 亦可运行）
- **仅标准库**（`math`、`random` 等）。装好 Python 后无需 `pip install`
  任何东西；核心、测试、演示全程离线，不访问网络、不需要账号/密钥/
  数据库/中间件/云 API/Docker。

## 命令

在仓库根目录执行：

```bash
python -m unittest discover -s tests -v   # 必要测试，约 1 秒
python demo.py                             # 固定输入演示，约 0.1 秒
```

两条命令都不联网、不 `sleep`、不读取预录结果；`demo.py` 真实计算，
打印一个**正常收敛结果**和两个**实际触发的边界**
（证明无解异常、溢出拒绝步/步失败）。

## 包结构

```
nb2reg/
  __init__.py    公开接口
  core.py        似然/梯度/Hessian、PMF、联合与固定 alpha 拟合
  special.py     lgamma 差分稳定化、自写 digamma/trigamma
  linalg.py      Cholesky 求解、设计矩阵秩诊断
demo.py          固定输入演示
tests/
  reference.py   独立参考（不 import 被测核心）
  test_special.py test_likelihood.py test_fit.py
```

## 公开接口

### 拟合

```python
from nb2reg import fit_nb2

res = fit_nb2(y, X=None, *, offset=None, alpha=None,
              max_iter=100, gtol=1e-7, step_tol=1e-6)
```

- `y`：长度 `n` 的非负整数计数列表，`1 <= n <= 100`。
- `X`：`n x p` 设计矩阵（行列表），`1 <= p <= 5`，必须**列满秩**；
  `X=None` 表示仅截距模型。`beta` 从原点起步。
- `offset`：可选长度 `n` 的暴露量，加入线性预测子。
- `alpha=None`（默认）：**联合**优化 `(beta, log theta)`。
  传入正数则为**固定 alpha 模式**，只优化 `beta`，便于独立核验。
- 整数控制参数（`max_iter`）传 `bool` 一律拒绝；`gtol/step_tol`
  必须为有限正数。

返回 `NB2Result`：

| 字段 | 含义 |
|---|---|
| `beta`, `alpha`, `theta` | 参数估计，`theta = 1/alpha` |
| `loglik`, `gradient`, `mu` | 对数似然、终止点梯度（联合模式末位为 `d/d log theta`）、拟合均值 |
| `converged`, `status`, `message` | 是否收敛、状态码与人类可读诊断 |
| `iterations`, `rejected_steps` | 迭代数、被拒绝（溢出/Armijo 失败）的候选步数 |
| `near_poisson` | 近 Poisson 边界标记（`theta >= 1e6`，即 `alpha <= 1e-6`） |
| `alpha_at_boundary` | 是否顶在 `theta` 上计算箱 |
| `fixed_alpha` | 是否为固定 alpha 模式 |
| `hessian`, `vcov` | 观测 Hessian 及其逆（参数协方差），固定模式为 beta 维 |

`status` 取值：

- `converged`：自由梯度与（预测）步长均进入容差；
- `converged_boundary`：beta 收敛且 `log theta` 顶在上计算箱
  （近 Poisson 边界），按约束最优点报告，**不冒充内点收敛**；
- `boundary_theta_min`：顶在下计算箱，未认证无约束 MLE；
- `max_iter`：**搜索预算耗尽**，不主张无解（可用
  `res.raise_if_not_converged()` 转 `ConvergenceBudgetError`）；
- `step_failure`：所有候选步都被拒绝（溢出/无有限 Armijo 增量），
  参数点保持不变（`StepFailureError`）。

**预算耗尽与证明无解严格分开**：全零计数且设计含截距时，
`y=0` 的质量关于 `mu` 严格递减、似然沿截距方向趋于一个有限点不可达的
上确界，直接抛 `NoFiniteMLEError`（证明无解）；它与 `max_iter`
（只是没算完）是两个不同的类型。

### 似然 / 梯度 / 概率

```python
from nb2reg import nb2_loglik, nb2_gradient, log_pmf, pmf
from nb2reg import range_probability, conditional_pmf

nb2_loglik(y, X, *, offset=None, beta=None,
           alpha=a)                       # 固定 alpha
nb2_loglik(..., theta=t)                   # 等价，theta = 1/alpha
nb2_loglik(..., log_theta=z)               # 给定 log theta 点
nb2_gradient(..., alpha=a)                 # beta 梯度（p 个）
nb2_gradient(..., log_theta=z)             # 联合梯度 [beta..., d/d log theta]

log_pmf(k, mu, alpha)                      # 单点对数质量
pmf(k, mu, alpha)
range_probability(lo, hi, mu, alpha)       # [lo, hi] 求和；hi=None 无上界
conditional_pmf(k, lo, hi, mu, alpha)      # 截断条件质量 P(k | lo<=Y<=hi)
```

离散度参数在 `alpha / theta / log_theta` 中**恰好给一个**；
不给或给多个都是 `NB2InputError`。

## 输入上限（接口边界）

- 样本数 `n <= 100`，设计列数 `p <= 5`；超限拒绝。
- 线性预测子可表示范围 `eta <= 700`（`mu < ~1e304`）；`theta` 计算箱
  `[1e-8, 1e8]`（即 `alpha ∈ [1e-8, 1e8]`），近 Poisson 标记阈值
  `theta = 1e6`。这些常数集中在 `nb2reg.LIMITS`。
- 这些是**接口边界**，不是演示规模；演示用小数据，不穷举上限。

## 错误语义

- 所有数值输入拒绝 `NaN`/`+Inf`/`-Inf`；`y` 必须是非负整数
  （负数、小数、`bool` 拒绝）；`X` 必须矩形、有限、列满秩；
  `offset` 长度必须等于 `n`。
- 秩亏设计抛 `RankDeficientDesignError`（`NB2InputError` 子类），
  消息与 `.dependent_columns` 给出可定位的列索引（改进 Gram–Schmidt
  选主元报告），不静默删列。
- 错误消息带位置，如 `y[1] must be finite (no NaN/Infinity)`、
  `X[3][2] ...`，不静默纠正。
- 拟合是只读操作：候选步被拒绝时参数点不变，不存在“部分变更”。

## 算法与设计取舍

1. **参数化**：用 `log theta`（而非 `log alpha`）与 `beta` 联合优化，
   牛顿法解观测 Hessian；链式法则给出 `d/d log theta = theta · d/d theta`、
   `d²/d(log theta)² = theta² d²/dtheta² + theta d/dtheta`。
2. **溢出处理是“拒绝步”而非裁剪数据**：候选点使 `eta > eta_max`
   或目标非有限时，该步被丢弃、线搜索缩短（Armijo 回溯），
   `rejected_steps` 计数；任何计数都不会被截断。导数用
   `r = mu/(mu+theta)`、`y/(mu+theta)` 等有界量书写，避免
   `mu*y`、`(mu+theta)²` 这类中间量先溢出成 `inf` 再产生 `NaN`。
3. **线搜索**：远离最优点（自由梯度范数很大）时，在满足 Armijo 的
   前提下倍增步长（至多 13 次），并在候选点方向导数变号时停止，
   以覆盖长距离行军；靠近最优点严格取全牛顿步 `t=1`，保留二次收敛。
   `t=1` 不可行时做经典折半回溯；Hessian 非正定则用 Levenberg 阻尼的
   Cholesky 求解，仍失败则 `step_failure`。
4. **近 Poisson 稳定性**：朴素 `lgamma(theta+y) - lgamma(theta)` 在
   `theta ~ 1e7` 时绝对舍入约 `1e-7`，完全淹没量级 `1e-13` 的 score。
   对整数 `y` 改用无抵消形式
   `y·log x + Σ_{j=1}^{y-1} log1p(j/x)`、`Σ 1/(x+j)`、
   `-Σ 1/(x+j)²`；`y` 很大时用两端点渐近展开的**解析差式**，
   小下端点回退到递推/`math.lgamma`。这使 theta 上箱处仍能得到
   干净的梯度与正确的边界判定。
5. **近边界活动集**：顶在 `log theta` 计算箱且梯度指向箱外时，把该
   坐标从牛顿子问题中固定，按自由坐标判 KKT 收敛，避免在箱上反复
   震荡或伪报失败。
6. **独立核验**：`tests/reference.py` 不 import 被测包，用单独写的
   PMF/对数似然与中心差分给期望值；往返（参数化互换）一致只作补充。

## 浮点结果的误差口径

- `digamma/trigamma`：与 `lgamma` 中心差分（`eps=1e-6`）相比，相对误差
  ≤ `5e-8`（该量级主要是大参数处 `lgamma` 差分参考自身的舍入极限，
  非本实现误差）；与已知常数 `psi(1)=-γ`、`psi'(1)=π²/6` 等在
  `1e-12` 内一致。
- 稳定差分：整数 `y` 用精确和，`gamma_ratio_log` 相对误差约 `1e-13`，
  `digamma_diff/trigamma_diff` 绝对误差 `≤ 1e-12`（与独立调和和比较）。
- **梯度**：在 `theta` 为 `O(0.1..10)` 的中等尺度，解析梯度与独立参考
  中心差分（`eps=1e-6`）相对误差 `≤ 1e-6`（60 个随机设计扫描的最坏值
  约 `1.4e-8`）；`theta ~ 1e8` 时函数值的浮点噪声使**有限差分参考本身**
  失效，此时以解析梯度/精确和为准，不用宽容差掩盖。
- **Hessian**：与解析梯度的中心差分逐分量比较，相对误差 `≤ 1e-5`
  （扫描最坏约 `9e-11`）。
- **PMF**：与独立参考逐点 `1e-10` 一致；全支撑求和到 5000 项归一化误差
  `≤ 4e-15`；`alpha=1e-7` 时与 Poisson 质量差 `< 1e-5`。
- **拟合收敛**：默认 `gtol=1e-7`（自由梯度最大绝对分量）、
  `step_tol=1e-6`（预测牛顿步长）。独立拟合之间（如加常数 offset 的
  等价模型）只保证在该终止精度内一致（约 `1e-6`），不声称机器精度。
  这些容差不用来掩盖边界错误：不能收敛就报 `max_iter/step_failure`，
  证明无解就抛异常。

## 已知限制（按本题数学子集，不宣称完整行业兼容）

- 仅支持 `n <= 100`、`p <= 5` 的稠密小规模数据；无稀疏矩阵、
  无稳健/三明治 SE、无零膨胀/广义化、无假设检验工具链。
- `theta` 被限制在 `[1e-8, 1e8]` 计算箱；超出即作为边界
  （`near_poisson` / `boundary_theta_min`）如实标注，不外推。
- 当最优均值超出可表示范围（如计数 `~1e250` 量级）时，联合 Hessian
  在浮点边缘丧失曲率，拟合如实报 `step_failure`/`max_iter`，
  **不返回伪收敛点**（`demo.py` 第 3 节实际触发）。
- 全零计数且**无截距**的部分设计，存在不可廉价判定的边界 MLE；
  库只对可严格证明的“含截距全零”情形抛 `NoFiniteMLEError`，
  其余无截距情形照常优化并如实报告状态。
- `range_probability` 的无界情形用补集 `1 - P(Y < lo)`，
  极端尾概率接近机器下溢时按 0 处理。
