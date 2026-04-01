#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tinygpu {

struct Config {
    std::size_t warp_size = 32;
    std::size_t threads_per_block = 64;
    std::size_t block_count = 2;
    std::size_t register_count = 8;
    std::size_t shared_memory_bytes = 1024;
    std::size_t global_memory_bytes = 4096;
    std::size_t max_cycles = 256;
};

enum class OpCode {
    MovImm,
    Add,
    Exit,
};

struct Instruction {
    OpCode opcode = OpCode::Exit;
    std::uint32_t dst = 0;
    std::uint32_t src0 = 0;
    std::uint32_t src1 = 0;
    std::int32_t imm = 0;
};

struct Kernel {
    std::string name;
    std::vector<Instruction> instructions;
};

struct Stats {
    std::size_t cycles = 0;
    std::size_t warp_issue_count = 0;
    std::size_t completed_warps = 0;
};

class Simulator {
public:
    explicit Simulator(Config config);

    Stats run(const Kernel& kernel);

private:
    struct ThreadState {
        std::size_t thread_index = 0;
        std::size_t pc = 0;
        bool done = false;
        std::vector<std::int32_t> registers;
    };

    struct WarpState {
        std::size_t warp_index = 0;
        std::size_t block_index = 0;
        bool done = false;
        std::vector<bool> active_mask;
        std::vector<ThreadState> threads;
    };

    struct BlockState {
        std::size_t block_index = 0;
        std::vector<std::uint8_t> shared_memory;
        std::vector<WarpState> warps;
    };

    Config config_;
    std::vector<std::uint8_t> global_memory_;

    std::vector<BlockState> build_blocks(const Kernel& kernel) const;
    bool step_warp(const Kernel& kernel, WarpState& warp);
};

Kernel make_bootstrap_kernel();

}  // namespace tinygpu
