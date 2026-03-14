# Gauge Attention 原理说明

本文件详细说明本项目 **Gauge Transformer Attention（规范场注意力）** 的理论背景、数学结构以及代码实现方式。

该实现基于 **HSF-HD（Hidden Semantic Field – Hierarchical Dynamics）** 理论视角，将大型语言模型的推理过程重新解释为：

> **潜语义流形上的信息传播，在规范场作用下发生路径偏转。**

核心思想：

- **基础模型权重 = 底流形的度量**
- **Gauge 模块 = 规范连接（connection）**
- **Attention = 离散平行运输**

---

# 1 Transformer 的几何解释

Transformer attention 可以写为：

\[
S_{ij}=\frac{Q_i K_j^\top}{\sqrt d}
\]

softmax 后：

\[
P_{ij}=\text{softmax}(S_{ij})
\]

输出：

\[
h_i'=\sum_j P_{ij} V_j
\]

解释：

|符号|意义|
|---|---|
|Q,K|语义方向|
|S|语义距离|
|P|信息传播概率|
|V|信息载荷|

因此：

> Transformer 本质上是在 **token 图上的加权信息传播系统**

---

# 2 HSF-HD 视角

HSF-HD 理论中：

|对象|几何意义|
|---|---|
|LLM 权重|潜语义流形度量 \(g_{\mu\nu}\)|
|Hidden state|流形坐标|
|Attention|平行运输|
|Gauge|连接修正|

---

基础 attention：

\[
S_{ij} \sim g(h_i,h_j)
\]

Gauge 修改：

\[
S'_{ij}=S_{ij}+\Delta S_{ij}
\]

传播变为：

\[
P'_{ij}=\text{softmax}(S'_{ij})
\]

最终输出：

\[
h_i^{out}=\sum_j P'_{ij}V_j
\]

---

# 3 Gauge Attention 的核心思想

标准微调：

\[
g_{\mu\nu} \rightarrow g'_{\mu\nu}
\]

即 **修改底流形度量（模型权重）**

而 Gauge 微调：

\[
\Gamma_{ij} \rightarrow \Gamma_{ij}+\Delta\Gamma_{ij}
\]

即 **修改连接（传播路径）**

---

因此：

|方法|修改对象|
|---|---|
|Full finetune|权重|
|LoRA|权重低秩|
|Gauge|连接|

---

# 4 本项目实现结构

本实现采用 **两模块架构**：

```

GaugeQwen2Attention
│
├── Q/K/V projection
├── RoPE
├── KV cache
├── repeat_kv (GQA)
│
├── CovariantGaugeAdapter
│      ├─ 生成 ΔS (gauge_bias)
│      └─ 生成 ΔV (delta_v)
│
├── 官方 Attention kernel
│
└── 输出 = attention + delta_v

```

---

# 5 数学结构

## 5.1 基础 attention

\[
S_{ij}=\frac{Q_iK_j^\top}{\sqrt d}
\]

---

## 5.2 Gauge 偏置

本项目 Gauge 偏置定义为：

\[
\Delta S =
g_{attn}(b_1+b_2)+g_{rel}b_3
\]

其中：

### b1

\[
b_1=\frac{Q_iA_{k,j}^\top}{\sqrt d}
\]

### b2

\[
b_2=\frac{A_{q,i}K_j^\top}{\sqrt d}
\]

### b3

\[
b_3=w_h^\top \tanh(A_{k,j}-A_{q,i})
\]

含义：

|项|意义|
|---|---|
|b1|query 与目标 token 的规范场耦合|
|b2|query 位置的规范场对 key 的解释|
|b3|相对规范势差|

---

## 5.3 新 attention score

\[
S'_{ij}=S_{ij}+\Delta S_{ij}
\]

---

## 5.4 新传播概率

\[
P'_{ij}=\text{softmax}(S'_{ij})
\]

---

## 5.5 新 attention 输出

\[
h_i^{attn}=\sum_j P'_{ij}V_j
\]

---

## 5.6 局域势修正

规范场还产生一个 **局域 value 势**：

\[
\Delta V_i=W_v^A A_{v,i}
\]

最终输出：

\[
h_i^{out}=h_i^{attn}+\lambda\Delta V_i
\]

其中

\[
\lambda=\tanh(out\_scale)
\]

---

# 6 最终公式

完整表达：

\[
S'_{ij}=\frac{Q_iK_j^\top}{\sqrt d}+g_{attn}(b_1+b_2)+g_{rel}b_3
\]

\[
P'=\text{softmax}(S'+mask)
\]

\[
h^{attn}=\sum_j P'_{ij}V_j
\]

\[
h^{out}=h^{attn}+\tanh(\lambda)\Delta V
\]

---

# 7 为什么不再使用 base_out 叠加

旧实现：

```

base_out + gauge_out

```

会导致：

\[
PV + P'V
\]

重复计算。

新实现：

```

AttentionOfficial(Q,K,V; mask + ΔS)

```

只计算一次 attention。

优点：

- 无重复计算
- 保留官方 kernel 加速
- 理论更干净

---

# 8 与官方 Attention 的关系

本项目不重写 attention kernel，而是：

```

attention_mask + gauge_bias

```

直接输入官方 attention。

即：

\[
AttentionOfficial(Q,K,V;mask+\Delta S)
\]

优点：

- 兼容 Flash / SDPA
- 利用 PyTorch fused kernel
- 适配 Qwen2 GQA

---

# 9 计算流程

完整流程：

```

hidden_states
│
├── Q/K/V projection
│
├── RoPE
│
├── KV cache
│
├── repeat_kv (GQA)
│
├── GaugeAdapter
│      │
│      ├─ ΔS (gauge_bias)
│      └─ ΔV
│
├── official attention
│
└── + ΔV

```

---

# 10 Gauge 参数

Gauge 模块训练参数包括：

|参数|意义|
|---|---|
|field_generator|生成规范场|
|aq_proj|query connection 投影|
|ak_proj|key connection 投影|
|rel_bias_vec|相对场权重|
|value_proj|value 势投影|
|g_attn|耦合强度|
|g_rel|相对耦合|
|g_val|value 势权重|
|out_scale|value 门控|

---

# 11 参数规模

假设：

```

hidden_size = 8192
rank = 16
heads = 64

```

Gauge 参数规模约：

```

< 5M parameters

```

相比：

```

Qwen2.5-32B ≈ 32B parameters

```

几乎可以忽略。

---

# 12 训练优势

Gauge 微调具有以下优势：

### 无灾难性遗忘

基础模型权重不变。

---

### 多任务可叠加

规范场满足叠加：

\[
A_\mu=\sum_i A_\mu^{(i)}
\]

---

### 参数高效

训练参数：

```

< 0.02% of base model

```

---

### 可解释性

Gauge bias 可以可视化：

```

attention score = base + gauge

```

可以直接分析：

- 哪些 token 被强化
- 哪些路径被抑制

---

# 13 与物理 Gauge 理论的对应

|物理|Transformer|
|---|---|
|空间|token graph|
|metric|QK score|
|connection|attention bias|
|field A|gauge field|
|curvature|attention path distortion|
|parallel transport|value aggregation|

---

# 14 未来扩展

可能的扩展方向：

### 非阿贝尔 Gauge

多任务场：

\[
A_\mu=\sum_i A_\mu^{(i)}T_i
\]

---

### Curvature Regularization

约束：

\[
F_{ijk}=\Gamma_{ij}+\Gamma_{jk}+\Gamma_{ki}
\]

---

### 动态任务场

根据 prompt 生成 Gauge。

---

# 15 总结

Gauge Attention 将 Transformer 重新解释为：

> **潜语义流形上的离散规范场动力系统**

最终传播规则：

\[
h^{out}=Attention(Q,K,V;\;mask+\Delta S)+\lambda\Delta V
\]

其中：

- \(\Delta S\) 改变传播路径
- \(\Delta V\) 改变节点势能

---

**一句话总结**

> Transformer + Gauge = 离散规范场信息传播系统

