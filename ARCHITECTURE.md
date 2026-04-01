# tinyGPU v0 设计草案

## 目标

构建一个教学导向的 GPU 模拟器，用来帮助理解两套视角之间的映射关系：

- 软件模型：`grid -> block -> thread`
- 硬件模型：`SM -> warp -> lane`

`v0` 优先保证清晰、可扩展、便于逐步迭代，不追求真实硬件复杂度。

## v0 范围

- 仅支持通用计算，不涉及图形管线
- 一次只运行一个 kernel
- 单个 `SM`
- 固定 `warp size = 32`
- 一个 `SM` 上可驻留多个 warp
- 支持 `global memory` 和 `shared memory`
- 最小周期级 warp 调度器
- 最小指令集，能够支撑早期 kernel，例如 `vector add`

`v0` 明确不做：

- 图形流水线
- Cache 层次
- 原子操作
- 多个 `SM`
- 真实 CUDA/PTX 兼容
- 复杂 reconvergence
- 细粒度功能单元时序建模

## 关键决策

- 使用自定义微型指令集，不直接兼容真实 GPU ISA
- 以 warp 为调度粒度，而不是 thread 粒度
- 显式保留 thread 状态，保证软件模型可见
- 将 shared memory 视为 block 私有的 scratchpad
- 使用周期级模拟，便于观察 stall 和 warp 切换
- 优先保证行为确定性，而不是追求硬件拟真

## 核心运行模型

- `Kernel`
  描述 launch 维度、寄存器数量、shared memory 需求和指令流
- `ThreadState`
  保存每个 thread 的架构态，例如寄存器、`pc` 和完成状态
- `WarpState`
  保存一组固定大小的 thread、active mask 和调度可见状态
- `BlockState`
  持有 shared memory、barrier 状态和该 block 下的所有 warp
- `SM`
  承载当前驻留的 block/warp，每个 cycle 选择一个 ready warp 执行一步
- `Simulator`
  持有 global memory，构造 launch 状态，驱动整个运行过程并输出统计信息

## 初始指令集

- `mov_imm`
- `mov_thread_idx`
- `add`
- `and_imm`
- `mul`
- `load_global`
- `store_global`
- `branch_if_zero`
- `barrier`
- `exit`

第一版可运行骨架允许只实现其中一个子集，但工程结构默认按这组指令扩展。

## 第一版执行模型

- 一次 kernel launch 先在逻辑上生成完整的 grid、block 和 thread
- 模拟器将这些 block 映射到单个 `SM`
- 每个 block 按 `warp size = 32` 切分成多个 warp
- 每个 cycle，调度器选择一个 ready warp
- 被选中的 warp 以 lockstep 方式执行一条指令
- 访存延迟和 barrier 等待会让某个 warp 暂时不可调度
- 当一个 warp stall 时，调度器转去执行其他 ready warp

## 阶段计划

1. 建立工程骨架和可运行的模拟主循环
2. 补齐最小指令执行和 warp 调度
3. 跑通基于 global memory 的 `vector add`
4. 加入 active mask 和基础分支发散
5. 加入 shared memory 和 block barrier
6. 跑通分块矩阵乘示例

## 早期迭代的非目标

- 对齐任意厂商 GPU 架构
- 追求峰值性能
- 构建二进制加载或编译流水线
- 做高度真实的微架构时序模拟
