# tinyGPU 目录说明

详细介绍看 [README.md](README.md)。这个文件只保留最基本的目录说明。

```text
tinyGPU/
├── CMakeLists.txt      # CMake 构建入口
├── README.md           # 项目介绍和使用方式
├── ARCHITECTURE.md     # 目录说明
├── include/tinygpu/    # 对外头文件
├── src/                # 模拟器与示例 kernel 实现
├── tests/              # GTest 测试
└── build/              # 本地构建产物
```

补充：

- `include/tinygpu/simulator.h` 定义模拟器核心类型和接口。
- `include/tinygpu/kernels.h` 声明教学用 kernel 构造函数。
- `src/main.cpp` 是示例程序入口。
