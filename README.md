# tinyGPU

`tinyGPU` 是一个教学导向的 GPU 模拟器，用来帮助理解两套常见视角之间的映射关系：

- 软件模型：`grid -> block -> thread`
- 硬件模型：`SM -> warp -> lane`

当前版本优先保证模型清晰、行为可解释、方便迭代，不追求真实硬件级复杂度。

## 当前包含的内容

- 单个 `SM` 上的简化 `SIMT` 执行模型
- 以 warp 为粒度的周期级调度
- `global memory` 与 `shared memory`
- 基础分支发散与 reconvergence
- `barrier` 同步
- 一组教学用 kernel：
  - `bootstrap`
  - `vector_add`
  - `branch_demo`
  - `shared_exchange`
  - `block_reduction`
  - `tiled_matmul`

## 指令子集

当前项目已经定义并逐步实现以下指令：

- `mov_imm`
- `mov_thread_idx`
- `mov_block_idx`
- `mov_block_thread_idx`
- `add`
- `mul`
- `and_imm`
- `set_lt_imm`
- `xor_imm`
- `load_global`
- `store_global`
- `load_shared`
- `store_shared`
- `branch_if_zero`
- `barrier`
- `exit`

## 快速开始

要求：

- CMake 3.16+
- 支持 C++17 的编译器
- GTest

配置并构建：

```bash
cmake -S . -B build
cmake --build build
```

运行示例程序：

```bash
./build/tinygpu
```

运行测试：

```bash
ctest --test-dir build --output-on-failure
```

## 运行效果

默认可执行程序会顺序运行几个教学 demo，并输出：

- kernel 名称
- cycle 数
- warp issue 次数
- global/shared memory 访问统计
- barrier 与分支发散统计
- 示例结果是否通过

如果想把教学 kernel 打印成更像小汇编的可读列表，可以使用：

- `tinygpu::disassemble_kernel(kernel)`

## 项目边界

当前版本明确不追求以下目标：

- 图形流水线
- 多个 `SM`
- cache 层次
- 原子操作
- 真实 CUDA/PTX 兼容
- 高拟真微架构时序建模

## 文档说明

- [ARCHITECTURE.md](ARCHITECTURE.md) 现在主要保留仓库文件结构与模块位置，便于快速导航
