# NB2 负二项计数回归（离线、仅标准库）

为计数测量实现的可复用“负二项过度离散拟合”模块。模型为 **NB2**：

```
y_i ~ NB2(mu_i, alpha)
mu_i  = exp(x_i · beta + offset_i)
Var(y_i) = mu_i + alpha · mu_i²        (alpha > 0)
```

联合优化 `beta` 与 `log alpha` 的对数似然；使用 `math.lgamma`，并自行实现
`digamma` / `trigamma`（递推 + 渐近展开，标准库没有这两个函数）。不调用、
不包装任何第三方同类实现；核心、测试、演示全部离线。

另提供带系数边界的入口 `fit_nb2_bounded`：调用方可指定部分 `beta_j >= 0`，
其余 beta 与 log alpha 仍自由；约束只作用于系数，绝不通过裁剪 y 或 mu 实现。

- **Python**：3.14.7（3.9+ 亦可运行；以 3.14.7 为验收版本）
- **平台**：Linux 原生
- **依赖**：无第三方依赖，仅 Python 标准库（`math`、`dataclasses`）。装好 Python 后无需 pip install。

## 运行命令

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 必要测试（39 个）
./.venv/bin/python demo.py                            # 固定输入演示，实测约 0.05 秒（远低于 90 秒预算）
```

（`./.venv/bin/python` 为固定的 Python 3.14.7 环境；若已在该环境内，直接 `python` 亦可。）

无需网络、账号、密钥、Docker、数据库或任何中间件；不读取任何配置或环境变量。

## 包接口

`import nbreg`，包为纯函数式、无全局状态，不修改任何输入：

### `fit_nb2(y, X, offset=None, fixed_alpha=None, alpha_init=None, max_iter=200, gtol=1e-8) -> NB2Result`

- `y`：长度 `n`（1 ≤ n ≤ 100）的**非负 Python int** 序列。拒绝 `bool`、`float`、负数。
- `X`：`n` 行，每行 1..5 个有限实数列。**不自动添加截距**，需要时自行放一列 1。
  秩亏（重复列、全零列或数值线性相关列）被拒绝并指出列号，不静默删除或纠正。
- `offset`：`None` 或长度 `n` 的有限实数列（与 `Xβ` 相加，不参与估计）。
- `fixed_alpha`：`None` 时联合估计 `(beta, log alpha)`；给定正有限实数时
  **固定 alpha 只优化 beta**，供独立核验与似然剖面分析。
- `alpha_init`：可选正有限初值；缺省用矩法起步。
- `max_iter`：`int`（拒绝 `bool`），范围 [1, 10^6]。
- `gtol`：收敛阈值，作用于投影梯度无穷范数 ≤ `gtol * max(1, |loglik|)`。

`NB2Result` 字段（`dataclass`）：

| 字段 | 含义 |
|---|---|
| `converged` | 仅当 `status == "converged"` 为真 |
| `status` | `converged` / `max_iter` / `line_search_failed` / `no_finite_mle` |
| `message` | 可定位的文字诊断 |
| `beta` / `log_alpha` / `alpha` | 返回点参数（固定 alpha 模式回传原值） |
| `loglik` / `gradient` | 该点的对数似然与梯度（固定模式梯度长度为 k） |
| `n_iter` / `n_rejected` | 迭代次数；回溯拒绝步数（含溢出拒绝） |
| `near_poisson` | `alpha < 1e-6` 时标记近 Poisson 边界 |
| `fixed_alpha` / `n_obs` / `n_params` | 模式与维度 |
| `active_bounds` | 边界拟合中最终钉在 0 的受约束系数下标（无约束拟合为空元组） |

**状态语义（搜索预算耗尽与“证明无解”严格分开）：**

- `max_iter` / `line_search_failed`：搜索预算耗尽，返回当前最优点，但**不**宣称最优，
  `converged=False`，不假收敛。
- `no_finite_mle`：数学上证明的结构性边界。目前在**全零计数**时给出：对任意有限
  `mu>0` 每个似然因子 `<1`，上确界 0 只在 `mu→0 (eta→-∞)` 达到，有限参数无 MLE；
  联合与固定 alpha 模式同此结论。

### `fit_nb2_bounded(y, X, offset=None, nonneg=None, fixed_alpha=None, alpha_init=None, max_iter=200, gtol=1e-8) -> NB2Result`

与 `fit_nb2` 同一模型、同一似然、同一 offset 语义，新增唯一参数：

- `nonneg`：长度 k 的 **bool 序列（必填）**，`nonneg[j] is True` 表示约束
  `beta_j >= 0`；其余 beta 与 log alpha 保持自由。元素必须是真正的 `bool`，
  长度不符或非 bool 给出带下标的 `NBFitError`。完全不约束时请继续用 `fit_nb2`。

语义与实现要点：

- **约束只作用于系数**：通过主动集 / 投影牛顿实现，从不裁剪或改写 y、mu、offset。
- **主动集 + 投影牛顿**：钉在 0 且梯度指向可行域外（`g_j <= 0`，最大化问题）的
  系数被冻结，牛顿步只在自由子空间求解；一旦冻结系数的梯度转正（唯一可能改善
  似然的方向），下一步自动释放。
- **线搜索同时保证可行与改善**：Armijo 回溯把试探点投影回可行盒（受约束系数
  截到 0——这是对*系数*的投影，不是对数据的裁剪），充分下降条件沿*实际发生*的
  位移度量；每个被接受的步都既可行又严格改善似然。
- **收敛判定**：自由坐标的投影梯度与边界 KKT 符号条件同时满足
  （`<= gtol * max(1, |loglik|)`）才报 `converged`，绝不因参数变化小而宣布成功。
- 显式 `fixed_alpha` 模式、联合模式的近 Poisson 边界诊断（`near_poisson`）、
  全零计数的 `no_finite_mle` 结构判定、秩亏拒绝、溢出回溯拒绝等行为与
  `fit_nb2` 完全一致（全零计数时上确界 0 也可经 `alpha -> inf` 逼近，故边界
  约束不改变该判定）。
- 返回的 `active_bounds` 列出最终钉在 0 的受约束系数下标。

### 纯函数（便于独立核验）

- `nb2_loglik(y, X, beta, log_alpha, offset=None) -> float`
- `nb2_objective(y, X, beta, log_alpha, offset=None) -> (loglik, gradient)`
  梯度长度 k+1：`∂ll/∂beta_j …, ∂ll/∂log alpha`。
- `nb2_pmf(k, mu, alpha) -> float`：单点概率质量，全对数空间计算；
  `lgamma(k+r)-lgamma(r)` 在 `y ≤ 4096` 时用精确有限和 `Σ log(r+m)`，
  避免 `alpha→0 (r→∞)` 时的灾难性抵消。

## 错误语义（契约）

- 所有数值输入拒绝 `NaN` / `±Infinity`，报错带具体位置，如 `X[3][1] must be finite, got nan`。
- 整数参数（`y` 元素、`max_iter`、`nb2_pmf` 的 `k`）拒绝 `bool`。
- 形状越界（n>100、列数超出 1..5、行列不齐）给出带维度的错误。
- 秩亏给出列号；溢出初值（`|eta|>700`）给出“computable region”错误。
- 拟合器是无状态批处理：非法输入在任何计算前抛出，**不存在部分变更**；
  输入对象永不被修改（有测试覆盖）。
- 异常类型为 `nbreg.NBFitError`（`ValueError` 子类）。

## 算法与设计取舍

1. **稳定似然形式**。令 `L = log1p(alpha·mu) = softplus(theta+eta)`，
   `w = mu/(1+alpha·mu) = exp(eta-L)`，`q = exp(-L)`。全程不直接计算
   `alpha·mu` 的指数，`L` 在大参数时按 `t + log1p(e^-t)` 计算。
2. **溢出用回溯拒绝步，不裁剪数据**。若候选点会使 `|eta|>700` 或
   `log alpha ∉ [-20, 25]`，该步判为无效并由 Armijo 回溯折半拒绝（计入
   `n_rejected`）；任何观测值都不会被截断、改写或剔除。
3. **观测信息牛顿法 + 阻尼**。对观测 Hessian 取负做 Cholesky 分解；非正定时
   对角阻尼递增（×100，最多 14 档），再退化为最速上升；单步分量上限 100，
   随后 Armijo 回溯（系数 1e-4，最多 60 次折半）。
4. **边界的主动集处理**。`log alpha` 顶到盒边界且梯度向外时冻结该坐标
   （KKT 投影），防止 `alpha→0` 时近奇异的 Hessian 把微小的 theta 移动耦合进
   beta 块导致停滞；同时置 `near_poisson`。`fit_nb2_bounded` 把同一
   主动集机制推广到系数的非负边界：冻结 `beta_j = 0` 且 `g_j <= 0` 的坐标，
   梯度转正即释放，线搜索投影回可行盒并沿实际位移检验 Armijo 下降。
5. **特殊函数自行实现**。`digamma`：递推至 x≥10 后用 Bernoulli 数渐近展开；
   `trigamma`：递推至 x≥12 后渐近展开。整数计数处的 `psi(y+r)-psi(r)` 等差值
   在 `y ≤ 4096` 时用精确有限和，避免大数相减。
6. **初值**：常数列吸收 `log(mean(y)+0.5) − mean(offset)`；alpha 用矩法
   （`(Var−mean)/mean²`）并夹到支持区间内。

## 误差口径（float64 / IEEE-754 双精度）

这些是**已验证的数值口径**，不是用宽容差掩盖边界错误：

- `digamma`/`trigamma` 与已知闭式值（ψ(1)=−γ、ψ₁(1)=π²/6 等）一致到约
  **1e-12 绝对误差**；trigamma 对 digamma 的数值导数一致到约 2e-7（中心差分 h=1e-6
  的自身误差量级）。
- 解析梯度对独立参考似然的**中心有限差分**（h=1e-6）：各分量绝对偏差 **≤ 3e-5**
  （该量级由有限差分截断/舍入主导）；在收敛点梯度无穷范数 **< 1e-8·max(1,|ll|)**。
- `nb2_pmf` 与独立的 pmf 比值递推路线逐点一致到 **1e-12 相对误差**；
  k=0..1999 求和与 1 的偏差 **< 1e-9**（截断尾项量级）；alpha=1e-9 时与
  Poisson pmf 一致到 1e-7（O(alpha) 模型偏差）。
- 固定 alpha 模式 beta 与独立数值梯度优化器一致到 **2e-4**（独立优化器自身较粗），
  对数似然一致到 1e-6；offset 平移截距性质到 1e-8。
- 边界拟合与活跃集完全枚举参照的 beta 一致到 **2e-4**、对数似然到 1e-6；
  被钉零的系数严格等于 0.0，边界 KKT 符号由独立有限差分梯度复核。
- 收敛默认 `gtol=1e-8`；这是梯度容差，不保证系数的标准误（本模块不计算 SE）。

## 测试与独立参考

- `tests/test_nb2.py`：39 个测试，覆盖全零计数、offset、重复列（秩亏拒绝）、
  高离散数据、近 Poisson 边界、输入校验与不修改输入、梯度有限差分、
  PMF 归一化与截断概率和、固定 alpha 独立优化器对比、预算耗尽 vs 无解分离；
  边界拟合另覆盖：最优恰为零（受约束系数被钉在 0.0）、约束使原负系数不可用、
  共线列拒绝、全零计数结构判定、回溯/溢出拒绝计数、约束不生效时与无约束
  拟合完全一致、联合模式近 Poisson 诊断保留、KKT 收敛报告。
- `tests/_reference.py`：**独立参考，不 import 被测核心**。用 lgamma 定义式、
  pmf 比值递推、中心有限差分、坐标式数值梯度上升各自独立给出期望值；
  边界拟合的参照是对活跃边界集合的**完全枚举**（每个受约束系数子集钉零、
  自由子空间用独立数值梯度优化器求解、再验 KKT），配合独立有限差分梯度。
  确定性数据由私有 LCG + CDF 反演生成。核心↔核心的“往返一致”只作补充
  （如 `nb2_loglik` 与拟合点一致）。

## 输入上限（接口边界）

- 样本数 1..100，列数 1..5；`log alpha ∈ [-20, 25]`（alpha 约 2.06e-9 .. 7.2e10），
  针对优化器与 `fixed_alpha`/`alpha_init` 参数。纯评估函数 `nb2_loglik`/
  `nb2_objective`/`nb2_pmf` 接受任意有限 `log alpha` 与任意正 alpha，用于核验。

## 已知限制（只实现题目契约的数学子集）

- 不提供标准误 / 协方差矩阵 / 置信区间 / 预测区间。
- 不自动加截距；不做列选择、正则化、权重、截尾/零膨胀模型。
- 联合似然表面依赖阻尼牛顿 + 矩法初值；在支持的规模与契约数据上收敛，
  但不对任意病态数据宣称全局保证。
- 全零计数判为无有限 MLE 并返回起点信息；不自动改用 Poisson 或边界模型。
- 不宣称与任何行业标准统计套件完全兼容。

## 文件

```
nbreg/__init__.py     公共接口
nbreg/nb2.py          核心：特殊函数、似然/梯度/Hessian、（投影）牛顿拟合
tests/test_nb2.py     必要测试（含边界拟合）
tests/_reference.py   独立参考（不调用被测核心；含活跃集枚举参照）
demo.py               固定输入演示：正常拟合 + 边界拟合 + 真实触发的异常/结构边界
README.md
```
