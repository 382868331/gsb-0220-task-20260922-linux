# 计数回归：NB2 与常数 logit 零膨胀 NB2（离线、仅标准库）

为计数测量实现的可复用“过度离散 / 零膨胀计数拟合”包，包含两个模型。

## NB2（原模型）

```
y_i ~ NB2(mu_i, alpha)
mu_i  = exp(x_i · beta + offset_i)
Var(y_i) = mu_i + alpha · mu_i²        (alpha > 0)
```

## ZINB（本次新增：常数 logit 零膨胀）

在 NB2 计数分量上叠加一个**无协变量的常数零膨胀分量**（单个 logit 参数），
计数均值保留原协变量：

```
Y = 0                 概率 pi  = sigmoid(logit_pi)   （结构零状态）
Y ~ NB2(mu_i, alpha)  概率 1-pi                      （计数状态）
mu_i = exp(x_i · beta + offset_i)
P(Y=0)   = pi + (1-pi) · f_NB(0 | mu, alpha)
P(Y=k>0) = (1-pi) · f_NB(k | mu, alpha)
```

按**混合似然**联合估计 `(beta, log alpha, logit pi)`；零观测**永不硬分组**，
估计全程携带软后验责任度 `tau_i = pi / P(Y_i=0)`（结构零后验概率，严格落在
(0,1)）。`pi → 0` 时模型精确退化为 NB2（嵌套）。

联合优化对数似然；使用 `math.lgamma`，并自行实现 `digamma` / `trigamma`
（递推 + 渐近展开，标准库没有这两个函数）。不调用、不包装任何第三方同类
实现；核心、测试、演示全部离线。

- **Python**：3.14.7（3.9+ 亦可运行；以 3.14.7 为验收版本）
- **平台**：Linux 原生
- **依赖**：无第三方依赖，仅 Python 标准库（`math`、`dataclasses`）。装好 Python 后无需 pip install。

## 运行命令

```bash
python -m unittest discover -s tests -v   # 全部测试（56 个：NB2 28 + ZINB 28）
python demo.py                            # 固定输入演示，实测约 0.02 秒（远低于 90 秒预算）
```

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

**状态语义（搜索预算耗尽与“证明无解”严格分开）：**

- `max_iter` / `line_search_failed`：搜索预算耗尽，返回当前最优点，但**不**宣称最优，
  `converged=False`，不假收敛。
- `no_finite_mle`：数学上证明的结构性边界。目前在**全零计数**时给出：对任意有限
  `mu>0` 每个似然因子 `<1`，上确界 0 只在 `mu→0 (eta→-∞)` 达到，有限参数无 MLE；
  联合与固定 alpha 模式同此结论。

### 纯函数（便于独立核验）

- `nb2_loglik(y, X, beta, log_alpha, offset=None) -> float`
- `nb2_objective(y, X, beta, log_alpha, offset=None) -> (loglik, gradient)`
  梯度长度 k+1：`∂ll/∂beta_j …, ∂ll/∂log alpha`。
- `nb2_pmf(k, mu, alpha) -> float`：单点概率质量，全对数空间计算；
  `lgamma(k+r)-lgamma(r)` 在 `y ≤ 4096` 时用精确有限和 `Σ log(r+m)`，
  避免 `alpha→0 (r→∞)` 时的灾难性抵消。

## ZINB 包接口

`import nbreg` 后与 NB2 并列；纯函数式、无全局状态，不修改任何输入。

### `fit_zinb(y, X, offset=None, max_iter=200, gtol=1e-8) -> ZINBResult`

- `y` / `X` / `offset` / `max_iter` / `gtol`：与 `fit_nb2` 完全相同的
  数据契约（n 1..100、列 1..5、int 计数、满秩、不自动加截距）。
- 联合优化 `(beta, log alpha, logit pi)`，观测信息牛顿法（复用 NB2 的阻尼 +
  Armijo 机制）。混合似然**一般不凹**：`pi=0`（NB2）可能是边界最优、同时
  又存在内点最优，因此拟合器从 **4 个确定性初值**（NB2 暖启动 + 3 个内点
  logit）分别跑牛顿，返回对数似然最高的点（`message` 注明 `best of N starts`）。

`ZINBResult` 字段（`dataclass`）：

| 字段 | 含义 |
|---|---|
| `converged` | 仅当 `status == "converged"` 为真 |
| `status` | `converged` / `max_iter` / `line_search_failed` / `no_finite_mle` |
| `message` | 可定位的文字诊断（含边界退化说明） |
| `beta` / `log_alpha` / `alpha` | **计数分量**参数（均值仍由原协变量决定） |
| `logit_pi` / `pi` | 常数零膨胀参数 |
| `loglik` / `gradient` | 对数似然与梯度（长度 k+2：beta、log alpha、logit pi） |
| `n_iter` / `n_rejected` | 最优起点的迭代数与回溯拒绝数 |
| `near_poisson` | 计数分量 `alpha < 1e-6` |
| `pi_at_lower_boundary` | `pi ≤ 1e-6`：无结构零，模型退化为 NB2 |
| `pi_at_upper_boundary` | `pi ≥ 1-1e-6`（全零结构判定时为真） |
| `structural_zero_degenerate` | 全零样本：无有限 MLE |
| `n_obs` / `n_zero` / `n_params` | 维度与零计数个数 |
| `responsibilities` | 每观测的**软**结构零后验概率 τ：正计数恒为 0，零严格落在 (0,1) |

**边界与退化语义：**

- **无结构零**（NB2 自身就能解释零比例，常见于大 alpha）：`pi → 0`，
  返回 `converged` 并置 `pi_at_lower_boundary=True`，似然与 NB2 一致
  （边界处 O(pi·n)≈1e-7 以内）。
- **额外零**：内点 MLE，`0 < pi < 1`，混合似然严格高于 NB2；零的
  τ 严格在 (0,1)，不做任何硬分组。
- **全零样本**：每个似然因子对有限参数 `<1`，上确界 0 只在 `pi→1
  (logit_pi→+∞)` 达到，此时 beta/alpha 不可识别，返回
  `status="no_finite_mle"`、`converged=False`、
  `structural_zero_degenerate=True`（数学判定，区别于预算耗尽）。
- 计数分量自身的 `alpha→0`（如完全饱和数据）由 `near_poisson` 标记。

### 纯函数（便于独立核验）

- `zinb_loglik(y, X, beta, log_alpha, logit_pi, offset=None) -> float`：
  `logit_pi` 接受任意有限实数（边界是 ±∞，有限值均良定义）。
- `zinb_objective(...) -> (loglik, gradient)`，梯度长度 k+2。
- `zinb_pmf(k, mu, alpha, pi) -> float`：混合 pmf；`pi` 取闭区间 [0,1]，
  `pi=0` 精确等于 `nb2_pmf`，`pi=1` 为零点点质量。
- `zinb_structural_probability(k, mu, alpha, pi) -> float`：结构零后验 τ；
  `τ(0)=pi/[pi+(1-pi)f_NB(0)]`，`τ(k>0)=0`。
- 优化盒：`logit_pi ∈ [-20, 20]`（常量 `LOGIT_PI_MIN/MAX`），
  `log alpha ∈ [-20, 25]`（沿用 `LOG_ALPHA_MIN/MAX`）。

## 错误语义（契约）

- 所有数值输入拒绝 `NaN` / `±Infinity`，报错带具体位置，如 `X[3][1] must be finite, got nan`。
- 整数参数（`y` 元素、`max_iter`、`nb2_pmf`/`zinb_pmf` 的 `k`）拒绝 `bool`。
- 形状越界（n>100、列数超出 1..5、行列不齐）给出带维度的错误。
- 秩亏给出列号；溢出初值（`|eta|>700`）给出“computable region”错误。
- ZINB 专用：`zinb_pmf`/`zinb_structural_probability` 的 `pi` 必须在闭区间
  `[0,1]`（如 `pi must be a finite real in [0, 1], got 1.3`）；
  `zinb_loglik`/`zinb_objective` 的 `logit_pi` 必须是有限实数。
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
4. **近 Poisson 边界的主动集处理**。`log alpha` 顶到边界且梯度向外时冻结该坐标
   （KKT 投影），防止 `alpha→0` 时近奇异的 Hessian 把微小的 theta 移动耦合进
   beta 块导致停滞；同时置 `near_poisson`。
5. **特殊函数自行实现**。`digamma`：递推至 x≥10 后用 Bernoulli 数渐近展开；
   `trigamma`：递推至 x≥12 后渐近展开。整数计数处的 `psi(y+r)-psi(r)` 等差值
   在 `y ≤ 4096` 时用精确有限和，避免大数相减。
6. **初值**：常数列吸收 `log(mean(y)+0.5) − mean(offset)`；alpha 用矩法
   （`(Var−mean)/mean²`）并夹到支持区间内。
7. **ZINB 稳定混合形式与软责任度**。零观测令 `u = log f_NB(0)`，
   `log P(0) = u + softplus(γ−u) − softplus(γ)`（logsumexp，不发生下溢），
   计数状态后验 `s = σ(u−γ)`、结构零后验 `τ = 1−s`。观测信息为
   **责任度加权 NB 信息 + 后验方差外积** `s·I_NB + s·τ·a·a'`（`a` 为零处
   NB 对数质量的得分；标准 EM 恒等式），零观测**不硬分组**。
8. **ZINB 非凹、多起点**。`pi=0` 可能是边界局部/全局最优、同时存在内点最优
   （数据中的零可由大 alpha 的 NB 自然解释），故以 NB2 暖启动 + 3 个内点
   logit 共 4 个确定性起点分别做阻尼牛顿，取对数似然最高者。
9. **ZINB 边界主动集与全零判定**。`logit_pi` 与 `log alpha` 顶到优化盒且
   梯度向外时冻结对应坐标（KKT）；全零样本直接给数学判定 `no_finite_mle`
   （上确界只在 pi→1 达到，计数参数不可识别），不进优化循环。

## 误差口径（float64 / IEEE-754 双精度）

这些是**已验证的数值口径**，不是用宽容差掩盖边界错误：

- `digamma`/`trigamma` 与已知闭式值（ψ(1)=−γ、ψ₁(1)=π²/6 等）一致到约
  **1e-12 绝对误差**；trigamma 对 digamma 的数值导数一致到约 2e-7（中心差分 h=1e-6
  的自身误差量级）。
- 两模型解析梯度对独立参考似然的**中心有限差分**（h=1e-6）：各分量绝对偏差
  **≤ 3e-5**（该量级由有限差分截断/舍入主导）；ZINB 解析**观测 Hessian**
  对梯度中心差分一致到相对 **1e-10**（混合外积项因此被验证）。
- 在收敛点梯度无穷范数 **< 1e-8·max(1,|ll|)**（ZINB 内点）。
- `nb2_pmf` / `zinb_pmf` 与独立递推/定义式逐点一致到 **1e-12 相对误差**；
  k=0..1999/2999 求和与 1 的偏差 **< 1e-9**（截断尾项量级）；`pi=0` 时
  ZINB pmf 与 NB2 逐点一致到 1e-15；alpha=1e-9 时与 Poisson 一致到 1e-7。
- 固定 alpha 模式 beta 与独立数值梯度优化器一致到 **2e-4**（独立优化器自身较粗），
  对数似然一致到 1e-6；offset 平移截距性质到 1e-8。
- ZINB MLE 与**完全独立的混合似然坐标上升**（有限差分梯度，多个确定性
  起点）在多个种子上对数似然一致到 **1e-5**、参数一致到 2e-3，覆盖内点
  MLE、`pi→0` 边界 MLE 与 alpha→0 情形。
- 收敛默认 `gtol=1e-8`；这是梯度容差，不保证系数的标准误（本模块不计算 SE）。

## 测试与独立参考

- `tests/test_nb2.py`：28 个测试，覆盖全零计数、offset、重复列（秩亏拒绝）、
  高离散数据、近 Poisson 边界、输入校验与不修改输入、梯度有限差分、
  PMF 归一化与截断概率和、固定 alpha 独立优化器对比、预算耗尽 vs 无解分离。
- `tests/test_zinb.py`：28 个测试，覆盖混合 pmf（定义式、归一化、`pi=0/1`
  边界、参数校验）、混合似然/梯度对独立参考有限差分、`pi=0` 退化为 NB2、
  额外零内点 MLE 对独立优化器、嵌套似然不低于 NB2、**无结构零**→pi 下界
  标记且似然等于 NB2、**全零**→no_finite_mle、软责任度（零严格 (0,1)、
  `mean τ=pi` 恒等式、offset 平移）、输入校验与不修改输入、维度上报、
  预算耗尽分离。
- `tests/_reference.py`：**独立参考，不 import 被测核心**。在原 NB2 参考
  （lgamma 定义式、pmf 比值递推、中心有限差分、坐标式数值梯度上升、私有
  LCG + CDF 反演采样）之上，独立给出 ZINB 混合 pmf/似然（直接从定义）、
  中心有限差分梯度与多起点坐标上升 MLE。核心↔核心的“往返一致”只作补充
  （如 `zinb_loglik` 与拟合点一致）。

## 输入上限（接口边界）

- 样本数 1..100，列数 1..5；`log alpha ∈ [-20, 25]`（alpha 约 2.06e-9 .. 7.2e10）、
  `logit_pi ∈ [-20, 20]`（pi 约 2e-9 .. 1−2e-9），针对优化器与
  `fixed_alpha`/`alpha_init` 参数。纯评估函数 `nb2_loglik`/`nb2_objective`/
  `nb2_pmf` 接受任意有限 `log alpha` 与任意正 alpha；`zinb_loglik`/
  `zinb_objective` 接受任意有限 `logit_pi`；`zinb_pmf` 接受闭区间 `pi∈[0,1]`，
  用于核验。

## 已知限制（只实现题面限定的数学子集）

- 不提供标准误 / 协方差矩阵 / 置信区间 / 预测区间。
- 不自动加截距；不做列选择、正则化、权重、截尾。
- **零分量为常数 logit**：不支持零分量协变量（ZIP/ZINB 的 `gamma·z`）、
  剖面置信区间、复杂边界推断（本次迭代明确不做）。
- 联合似然表面依赖阻尼牛顿 + 4 个确定性起点；在支持的规模与契约数据上
  （120 组随机确定性样本全部收敛）表现稳健，但不对任意病态数据宣称全局保证。
- 全零计数判为无有限 MLE；不自动改用边界模型。
- 不宣称与任何行业标准统计套件完全兼容。

## 文件

```
nbreg/__init__.py     公共接口（NB2 + ZINB）
nbreg/nb2.py          NB2 核心：特殊函数、逐观测量、似然/梯度/Hessian、牛顿拟合
nbreg/zinb.py         ZINB 核心：稳定混合似然、软责任度信息、多起点牛顿拟合
tests/test_nb2.py     NB2 测试（28 个，原样保留）
tests/test_zinb.py    ZINB 测试（28 个，新增）
tests/_reference.py   独立参考（不调用被测核心；含 ZINB 定义式与独立 MLE）
demo.py               演示：NB2 正常 + ZINB 内点/下界/全零边界 + 真实异常
README.md
```
