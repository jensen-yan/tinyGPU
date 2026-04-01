#include "tinygpu/disasm.h"
#include "tinygpu/kernels.h"
#include "tinygpu/simulator.h"

#include <gtest/gtest.h>

namespace {

tinygpu::Config default_config() {
    return tinygpu::Config{};
}

std::size_t total_threads(const tinygpu::Config& config) {
    return config.block_count * config.threads_per_block;
}

TEST(SimulatorTest, IsaMicroKernelStoresExpectedSum) {
    const tinygpu::Config config = default_config();
    tinygpu::Simulator simulator(config);

    const std::size_t thread_count = total_threads(config);
    const std::size_t out_base = 0;
    ASSERT_GE(simulator.global_memory_size(), out_base + thread_count);

    tinygpu::Kernel kernel;
    kernel.name = "isa_micro_add_store";
    kernel.instructions = {
        tinygpu::Instruction{tinygpu::OpCode::MovThreadIdx, 0, 0, 0, 0, 0},
        tinygpu::Instruction{tinygpu::OpCode::MovImm, 1, 0, 0, static_cast<std::int32_t>(out_base), 0},
        tinygpu::Instruction{tinygpu::OpCode::Add, 2, 0, 1, 0, 0},
        tinygpu::Instruction{tinygpu::OpCode::MovImm, 3, 0, 0, 1, 0},
        tinygpu::Instruction{tinygpu::OpCode::MovImm, 4, 0, 0, 2, 0},
        tinygpu::Instruction{tinygpu::OpCode::Add, 5, 3, 4, 0, 0},
        tinygpu::Instruction{tinygpu::OpCode::StoreGlobal, 0, 2, 5, 0, 0},
        tinygpu::Instruction{tinygpu::OpCode::Exit, 0, 0, 0, 0, 0},
    };

    const tinygpu::Stats stats = simulator.run(kernel);

    for (std::size_t i = 0; i < thread_count; ++i) {
        EXPECT_EQ(simulator.read_global(out_base + i), 3) << "thread " << i;
    }
    EXPECT_EQ(stats.global_load_count, 0u);
    EXPECT_EQ(stats.global_store_count, thread_count);
    EXPECT_EQ(stats.completed_warps, 4u);
}

TEST(SimulatorTest, VectorAddKernelProducesExpectedOutput) {
    const tinygpu::Config config = default_config();
    tinygpu::Simulator simulator(config);

    const std::size_t thread_count = total_threads(config);
    const std::size_t a_base = 0;
    const std::size_t b_base = a_base + thread_count;
    const std::size_t c_base = b_base + thread_count;
    ASSERT_GE(simulator.global_memory_size(), c_base + thread_count);

    for (std::size_t i = 0; i < thread_count; ++i) {
        simulator.write_global(a_base + i, static_cast<std::int32_t>(i));
        simulator.write_global(b_base + i, static_cast<std::int32_t>(1000 + i));
        simulator.write_global(c_base + i, -1);
    }

    const tinygpu::Kernel kernel = tinygpu::make_vector_add_kernel(
        static_cast<std::int32_t>(a_base),
        static_cast<std::int32_t>(b_base),
        static_cast<std::int32_t>(c_base));
    const tinygpu::Stats stats = simulator.run(kernel);

    for (std::size_t i = 0; i < thread_count; ++i) {
        EXPECT_EQ(simulator.read_global(c_base + i), static_cast<std::int32_t>(1000 + 2 * i)) << "index " << i;
    }
    EXPECT_EQ(stats.cycles, 48u);
    EXPECT_EQ(stats.warp_issue_count, 48u);
    EXPECT_EQ(stats.global_load_count, 2 * thread_count);
    EXPECT_EQ(stats.global_store_count, thread_count);
    EXPECT_EQ(stats.divergent_branch_count, 0u);
}

TEST(SimulatorTest, BranchDemoProducesEvenOddSplit) {
    const tinygpu::Config config = default_config();
    tinygpu::Simulator simulator(config);

    const std::size_t thread_count = total_threads(config);
    const std::size_t out_base = 3 * thread_count;
    ASSERT_GE(simulator.global_memory_size(), out_base + thread_count);

    for (std::size_t i = 0; i < thread_count; ++i) {
        simulator.write_global(out_base + i, -1);
    }

    const tinygpu::Kernel kernel = tinygpu::make_branch_demo_kernel(static_cast<std::int32_t>(out_base));
    const tinygpu::Stats stats = simulator.run(kernel);

    for (std::size_t i = 0; i < thread_count; ++i) {
        EXPECT_EQ(simulator.read_global(out_base + i), (i % 2 == 0) ? 200 : 300) << "index " << i;
    }
    EXPECT_EQ(stats.global_load_count, 0u);
    EXPECT_EQ(stats.global_store_count, thread_count);
    EXPECT_EQ(stats.divergent_branch_count, 4u);
    EXPECT_EQ(stats.completed_warps, 4u);
}

TEST(SimulatorTest, SharedExchangeUsesBarrierAndBlockSharedMemory) {
    const tinygpu::Config config = default_config();
    tinygpu::Simulator simulator(config);

    const std::size_t thread_count = total_threads(config);
    const std::size_t out_base = 4 * thread_count;
    ASSERT_GE(simulator.global_memory_size(), out_base + thread_count);

    for (std::size_t i = 0; i < thread_count; ++i) {
        simulator.write_global(out_base + i, -1);
    }

    const tinygpu::Kernel kernel = tinygpu::make_shared_exchange_kernel(static_cast<std::int32_t>(out_base));
    const tinygpu::Stats stats = simulator.run(kernel);

    for (std::size_t i = 0; i < thread_count; ++i) {
        const std::size_t local = i % config.threads_per_block;
        EXPECT_EQ(simulator.read_global(out_base + i), static_cast<std::int32_t>(local ^ 32U)) << "index " << i;
    }
    EXPECT_EQ(stats.global_load_count, 0u);
    EXPECT_EQ(stats.global_store_count, thread_count);
    EXPECT_EQ(stats.shared_load_count, thread_count);
    EXPECT_EQ(stats.shared_store_count, thread_count);
    EXPECT_EQ(stats.barrier_issue_count, 4u);
    EXPECT_EQ(stats.completed_warps, 4u);
}

TEST(SimulatorTest, BlockReductionProducesOneSumPerBlock) {
    const tinygpu::Config config = default_config();
    tinygpu::Simulator simulator(config);

    const std::size_t thread_count = total_threads(config);
    const std::size_t in_base = 5 * thread_count;
    const std::size_t out_base = in_base + thread_count;
    ASSERT_GE(simulator.global_memory_size(), out_base + config.block_count);

    for (std::size_t i = 0; i < thread_count; ++i) {
        simulator.write_global(in_base + i, 1);
    }
    for (std::size_t block = 0; block < config.block_count; ++block) {
        simulator.write_global(out_base + block, -1);
    }

    const tinygpu::Kernel kernel = tinygpu::make_block_reduction_kernel(
        static_cast<std::int32_t>(in_base),
        static_cast<std::int32_t>(out_base));
    const tinygpu::Stats stats = simulator.run(kernel);

    for (std::size_t block = 0; block < config.block_count; ++block) {
        EXPECT_EQ(simulator.read_global(out_base + block), static_cast<std::int32_t>(config.threads_per_block))
            << "block " << block;
    }
    EXPECT_EQ(stats.global_store_count, config.block_count);
    EXPECT_EQ(stats.barrier_issue_count, 28u);
    EXPECT_EQ(stats.completed_warps, 4u);
}

TEST(SimulatorTest, TiledMatmulProducesExpected8x8Tile) {
    tinygpu::Config config = default_config();
    config.block_count = 1;
    config.register_count = 10;
    tinygpu::Simulator simulator(config);

    constexpr std::size_t kDim = 8;
    constexpr std::size_t kMatrixSize = kDim * kDim;
    const std::size_t a_base = 0;
    const std::size_t b_base = a_base + kMatrixSize;
    const std::size_t c_base = b_base + kMatrixSize;
    ASSERT_GE(simulator.global_memory_size(), c_base + kMatrixSize);

    for (std::size_t row = 0; row < kDim; ++row) {
        for (std::size_t col = 0; col < kDim; ++col) {
            const std::size_t index = row * kDim + col;
            simulator.write_global(a_base + index, static_cast<std::int32_t>(row * 10 + col + 1));
            simulator.write_global(b_base + index, static_cast<std::int32_t>((row == col) ? 2 : 1));
            simulator.write_global(c_base + index, -1);
        }
    }

    const tinygpu::Kernel kernel = tinygpu::make_tiled_matmul_kernel(
        static_cast<std::int32_t>(a_base),
        static_cast<std::int32_t>(b_base),
        static_cast<std::int32_t>(c_base));
    const tinygpu::Stats stats = simulator.run(kernel);

    for (std::size_t row = 0; row < kDim; ++row) {
        for (std::size_t col = 0; col < kDim; ++col) {
            std::int32_t expected = 0;
            for (std::size_t k = 0; k < kDim; ++k) {
                const std::int32_t a = static_cast<std::int32_t>(row * 10 + k + 1);
                const std::int32_t b = static_cast<std::int32_t>((k == col) ? 2 : 1);
                expected += a * b;
            }
            EXPECT_EQ(simulator.read_global(c_base + row * kDim + col), expected)
                << "row=" << row << " col=" << col;
        }
    }
    EXPECT_EQ(stats.global_load_count, 2 * kMatrixSize);
    EXPECT_EQ(stats.global_store_count, kMatrixSize);
    EXPECT_EQ(stats.shared_load_count, 16 * kMatrixSize);
    EXPECT_EQ(stats.shared_store_count, 2 * kMatrixSize);
    EXPECT_EQ(stats.barrier_issue_count, 2u);
}

TEST(SimulatorTest, DisassemblerRendersReadableBranchKernel) {
    const tinygpu::Kernel kernel = tinygpu::make_branch_demo_kernel(192);
    const std::string text = tinygpu::disassemble_kernel(kernel);

    EXPECT_NE(text.find("kernel branch_demo"), std::string::npos);
    EXPECT_NE(text.find("mov_thread_idx r0"), std::string::npos);
    EXPECT_NE(text.find("branch_if_zero r3"), std::string::npos);
    EXPECT_NE(text.find("target="), std::string::npos);
    EXPECT_NE(text.find("join="), std::string::npos);
    EXPECT_NE(text.find("store_global [r2], r4"), std::string::npos);
}

}  // namespace
