# NB2 / ZINB 计数回归（离线、仅标准库）

为计数测量实现的可复用“过度离散 / 零膨胀拟合”模块。包含两个模型：

**NB2**：

```
y_i ~ NB2(mu_i, alpha)
mu_i  = exp(x_i · beta + offset_i)
Var(y_i) = mu_i + alpha · mu_i²        (alpha > 0)
```

**ZINB（本轮新增）**：在 NB2 上叠加一个**常数 logit 零膨胀参数**
（零分量不带协变量），计数分量均值保留原协变量：

```
以概率 pi       : Y = 0                  （结构零）
以概率 (1 - pi) : Y ~ NB2(mu_i, alpha)   （计数分量，mu_i 同上）
logit(pi) = z                            （单个标量）
P(Y=0)   = pi + (1-pi)·p0_i,  p0_i = (1+alpha·mu_i)^(-1/alpha)
P(Y=k>0) = (1-pi)·f_NB(k; mu_i, alpha)
```

按**混合似然**联合优化 `(beta, log alpha, logit pi)`；零观测通过后验软权重
`tau_i = (1-pi)·p0_i / (pi+(1-pi)·p0_i)` 进入分数函数，**绝不硬分零组**。
ZINB 在 pi=0 处恰好退化为 NB2（嵌套）。使用 `math.lgamma`，并自行实现
`digamma` / `trigamma`（递推 + 渐近展开，标准库没有这两个函数）。不调用、
不包装任何第三方同类实现；核心、测试、演示全部离线。

- **Python**：3.14.7（3.9+ 亦可运行；以 3.14.7 为验收版本）
- **平台**：Linux 原生
- **依赖**：无第三方依赖，仅 Python 标准库（`math`、`dataclasses`）。装好 Python 后无需 pip install。

## 运行命令

```bash
python -m unittest discover -s tests -v   # 全部测试（54 个）
python demo.py                            # 固定输入演示，实测约 1.2 秒（远低于 90 秒预算）
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

### `fit_zinb(y, X, offset=None, pi_init=None, alpha_init=None, max_iter=200, gtol=1e-8) -> ZINBResult`

零膨胀 NB2 的混合似然拟合（本轮新增；`y`/`X`/`offset`/`max_iter`/`gtol`
契约与 `fit_nb2` 完全一致，输入校验也一致）。

- `pi_init`：`None` 或严格开区间 (0, 1) 内的实数（`0`/`1` 对应无穷 logit，
  被拒绝；`bool` 拒绝）。缺省按“零超出”起步：
  `pi = (零比例 − 平均 NB2 p0) / (1 − 平均 NB2 p0)`，夹到 [1e-4, 0.9]。
- `alpha_init`：同 `fit_nb2`。
- 优化变量 `(beta, theta=log alpha, z=logit pi)`；`z` 的拟合盒为
  `[LOGIT_PI_MIN, LOGIT_PI_MAX] = [-30, 30]`（pi≈9.4e-14 .. 1−9.4e-14），
  越界步由 Armijo 回溯拒绝，与 NB2 的溢出处理同款，不裁剪数据。

`ZINBResult` 字段（`dataclass`）：

| 字段 | 含义 |
|---|---|
| `converged` | 仅当 `status == "converged"` 为真（盒边界 KKT 也算收敛） |
| `status` | `converged` / `max_iter` / `line_search_failed` / `no_finite_mle` |
| `message` | 可定位的文字诊断 |
| `beta` / `log_alpha` / `alpha` | 计数分量参数（计数均值保留原协变量） |
| `logit_pi` / `pi` | 常数零膨胀参数（logit 与概率两种口径） |
| `loglik` / `gradient` | 该点混合对数似然与梯度（长度 k+2，顺序 beta, log α, logit π） |
| `n_iter` / `n_rejected` | 迭代次数；回溯拒绝步数（含越界/溢出拒绝） |
| `near_poisson` | `alpha < 1e-6` |
| `near_no_inflation` | `pi < 1e-6`：零膨胀实际未起作用，模型退化到 NB2 |
| `pi_boundary` | `"lower"`（pi 贴下边界=无膨胀退化）/ `"upper"`（贴上边界）/ `"none"` |
| `n_obs` / `n_zeros` / `n_params` | 样本数、零计数数、参数数（k+2） |

**三种数据情形的行为（均有测试与演示覆盖）：**

- **无结构零（数据实为普通 NB2）**：π 收敛到下边界，`near_no_inflation=True`，
  极限似然与 NB2 完全相等（嵌套关系，经独立参考公式核对）。
- **额外零（真实零膨胀）**：π 估在内部，`pi_boundary="none"`，混合似然严格
  优于 π=0 的 NB2 似然；拟合点与完全独立的多起点数值梯度优化器一致到
  1e-8 对数似然。
- **全零样本**：`status="no_finite_mle"` 结构性结论：对任意有限 μ、π<1，
  每个因子 π+(1−π)·p0 < 1，上确界 0 只在 μ→0 或 π→1 达到，有限参数无 MLE。
  与 `max_iter`（预算耗尽）严格区分。

**不硬分零组**：零观测对分数函数的贡献整体乘以软后验 τ_i（在 (0,1) 内随 μ_i
连续变化），测试比较了混合对数似然与“把每个零硬判给一个分量”的对数似然，
前者严格更大（`log(a+b) > max(log a, log b)`），即实现不可能是硬分组。

### ZINB 纯函数（便于独立核验）

- `zinb_loglik(y, X, beta, log_alpha, logit_pi, offset=None) -> float`
- `zinb_objective(y, X, beta, log_alpha, logit_pi, offset=None) -> (loglik, gradient)`
  梯度长度 k+2：`∂ll/∂beta_j …, ∂ll/∂log alpha, ∂ll/∂logit pi`。
- `zinb_pmf(k, mu, alpha, pi) -> float`：混合单点概率；纯评估允许端点
  π=0（等于 NB2）与 π=1（零处点质量），拟合器只估内部。

## 错误语义（契约）

- 所有数值输入拒绝 `NaN` / `±Infinity`，报错带具体位置，如 `X[3][1] must be finite, got nan`。
- 整数参数（`y` 元素、`max_iter`、`nb2_pmf` 的 `k`）拒绝 `bool`。
- 形状越界（n>100、列数超出 1..5、行列不齐）给出带维度的错误。
- 秩亏给出列号；溢出初值（`|eta|>700`）给出“computable region”错误。
- ZINB 额外拒绝 `pi_init ∉ (0,1)`（含 0、1、bool、NaN）；纯评估函数
  `zinb_pmf` 的 `pi` 只要求在闭区间 [0, 1]。
- 拟合器是无状态批处理：非法输入在任何计算前抛出，**不存在部分变更**；
  输入对象永不被修改（有测试覆盖）。
- 异常类型为 `nbreg.NBFitError`（`ValueError` 子类）；ZINB 的
  `nbreg.ZINBFitError` 是其子类，旧的 `except NBFitError` 调用不受影响。

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
7. **ZINB 混合似然的 EM 式软权**。零处密度 `D = pi+(1-pi)·p0` 用
   `log D = log(1-pi) + logaddexp(z, log p0)` 计算，后验
   `tau = 1/(1+exp(z−log p0))` 是唯一指数化的混合量，大参数时直接下溢为 0。
   观测 Hessian 中零观测同时含“加权 NB 曲率”和 τ 自身随参数变化的混合曲率
   （`d²log D/dz² = τ(1−τ) − π(1−π)`，可能为正，故负 Hessian 的 Cholesky
   阻尼与 NB2 同款保留）；z、theta 在盒边界且梯度向外时冻结该坐标（KKT）。
   初值 π 用零超出估计。

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
- **ZINB**：解析混合似然梯度对独立参考混合似然的中心有限差分一致到
  **1.3e-8**；解析观测 Hessian 对独立二阶中心差分在 h=2e-4 时一致到
  **2.2e-8**（差分误差随 h 减小而上升，确认是差分截断/舍入主导）。
  ZINB 拟合点与完全独立的多起点数值梯度优化器在对数似然上一致到 **1e-8**，
  参数一致到 1e-5（小样本、α 与 π 近乎不可分时独立优化器自身会在 α 上停滞，
  该情形只核对其真正达到的对数似然与 π）。手算两观测例子逐位一致。
  `zinb_pmf` 与独立直算混合公式逐点一致到 **1e-12**；k=0..1999 求和与 1
  的偏差 **< 1e-9**；π=0 时质量与 NB2 完全相等（嵌套）。观测 Hessian 对
  二阶中心差分在 h=2e-4 时两核验点偏差分别为 2.2e-8 / 1.5e-6，且误差随 h
  缩小而增大，确认为差分截断/舍入主导。
- 收敛默认 `gtol=1e-8`；这是梯度容差，不保证系数的标准误（本模块不计算 SE）。

## 测试与独立参考

- `tests/test_nb2.py`：28 个测试（**原样保留**），覆盖全零计数、offset、
  重复列（秩亏拒绝）、高离散数据、近 Poisson 边界、输入校验与不修改输入、
  梯度有限差分、PMF 归一化与截断概率和、固定 alpha 独立优化器对比、
  预算耗尽 vs 无解分离。
- `tests/test_zinb.py`：26 个新测试：手算两点似然与 PMF 端点、混合梯度有限
  差分（含全零/高膨胀点）、PMF 独立直算与归一化、**无结构零退化（π→0，
  似然等于 NB2）**、**额外零恢复且严格优于 NB2**、与独立多起点数值 MLE 一致、
  **软混合严格优于硬分零组**、**全零 no_finite_mle**、offset 平移、
  pi_init/形状/秩亏/NaN 校验、不修改输入、预算耗尽分离。
- `tests/_reference.py`：**独立参考，不 import 被测核心**。NB2 部分原样；
  新增 ZINB 混合 pmf/似然（直接由两组分定义写出）、中心有限差分梯度、
  多起点数值梯度联合优化器、两阶段（结构硬币与计数抽得分开）确定性
  ZINB 数据生成 `make_zinb_dataset`。核心↔核心的“往返一致”只作补充。

## 输入上限（接口边界）

- 样本数 1..100，列数 1..5；`log alpha ∈ [-20, 25]`（alpha 约 2.06e-9 .. 7.2e10），
  `logit pi ∈ [-30, 30]`（pi 约 9.4e-14 .. 1−9.4e-14），针对优化器与
  `fixed_alpha`/`alpha_init`/`pi_init` 参数。纯评估函数接受任意有限
  `log alpha`/`logit pi`，`zinb_pmf` 接受闭区间 π∈[0,1]，用于核验。

## 已知限制（只实现题目契约的数学子集）

- 不提供标准误 / 协方差矩阵 / 置信区间 / 预测区间，也不提供零膨胀参数的
  剖面置信区间或复杂边界推断。
- 零膨胀只有**一个常数 logit 参数**；零分量不支持协变量/分组 logit。
- 不自动加截距；不做列选择、正则化、权重、截尾、hurdle 模型。
- 联合似然表面依赖阻尼牛顿 + 矩法/零超之初值；在支持的规模与契约数据上
  收敛，但不对任意病态数据宣称全局保证。小样本下 α 与 π 可近共线（都只经
  p0 影响零概率），此时 π=0 边界退化是似然本身的真实结论而非数值事故
  （已由独立优化器逐例核实）。
- 全零计数判为无有限 MLE 并返回起点信息；不自动改用 Poisson 或边界模型。
- 不宣称与任何行业标准统计套件完全兼容。

## 文件

```
nbreg/__init__.py     公共接口（NB2 + ZINB）
nbreg/nb2.py          NB2 核心：特殊函数、似然/梯度/Hessian、牛顿拟合
nbreg/zinb.py         ZINB 核心：混合似然/梯度/Hessian（软后验权）、牛顿拟合
tests/test_nb2.py     NB2 测试（28 个，原样保留）
tests/test_zinb.py    ZINB 测试（26 个新增）
tests/_reference.py   独立参考（不调用被测核心；NB2 + ZINB）
demo.py               固定输入演示：正常拟合（NB2、ZINB）+ 边界退化 + 实际异常
README.md
```
