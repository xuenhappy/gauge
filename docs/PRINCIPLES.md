# Gauge-QA 原理说明

## 核心思想

本项目把 QA 适配分成三类：
1. Frozen：不训练任何参数。
2. LoRA：通过低秩增量改变等效线性映射。
3. Gauge：不改 base weights，而是在 attention 输出后叠加一个上下文相关的规范场残差支路。

## Gauge v2

Gauge 的最小实现由 `CovariantGaugeAdapter v2` 给出：
- 输入：当前层 `hidden_states` 与 base attention 的 `q/k/v`
- 输出：一个残差 `delta_out`
- 最终：`attn_out = base_attn_out + delta_out`

结构包括：
- 局域规范场发生器：`field_generator(hidden_states) -> (A_q, A_k, A_v)`
- 定向 attention bias：用 `q_base / k_base / A_q / A_k` 修正路径
- value modulation：用 `A_v` 修正值通道
- 零场初始化：`out_scale=0, g_attn=0, g_val=0`
