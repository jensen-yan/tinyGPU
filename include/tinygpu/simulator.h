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
    std::size_t shared_memory_words = 256;
    std::size_t global_memory_words = 1024;
    std::size_t max_cycles = 1024;
};

enum class OpCode {
    MovImm,
    MovThreadIdx,
    MovBlockIdx,
    MovBlockThreadIdx,
    Add,
    AndImm,
    SetLtImm,
    XorImm,
    LoadGlobal,
    StoreGlobal,
    LoadShared,
    StoreShared,
    BranchIfZero,
    Barrier,
    Exit,
};

struct Instruction {
    OpCode opcode = OpCode::Exit;
    std::uint32_t dst = 0;
    std::uint32_t src0 = 0;
    std::uint32_t src1 = 0;
    std::int32_t imm = 0;
    // Branch target when the predicate is true.
    std::uint32_t target = 0;
    // Reconvergence PC used when a divergent branch should merge again later.
    std::uint32_t join_target = 0;
};

struct Kernel {
    std::string name;
    std::vector<Instruction> instructions;
};

struct Stats {
    std::size_t cycles = 0;
    std::size_t warp_issue_count = 0;
    std::size_t global_load_count = 0;
    std::size_t global_store_count = 0;
    std::size_t shared_load_count = 0;
    std::size_t shared_store_count = 0;
    std::size_t barrier_issue_count = 0;
    std::size_t divergent_branch_count = 0;
    std::size_t completed_warps = 0;
};

class Simulator {
public:
    explicit Simulator(Config config);

    Stats run(const Kernel& kernel);
    void write_global(std::size_t index, std::int32_t value);
    std::int32_t read_global(std::size_t index) const;
    std::size_t global_memory_size() const;

private:
    struct ThreadState {
        std::size_t thread_index = 0;
        bool done = false;
        std::vector<std::int32_t> registers;
    };

    struct WarpState {
        struct ReconvergenceFrame {
            // PC where the split paths should merge back together.
            std::size_t merge_pc = 0;
            // PC for the not-taken path, which runs after the taken path completes.
            std::size_t pending_pc = 0;
            bool pending_started = false;
            std::vector<bool> pending_mask;
            // Union of both path masks, restored after reconvergence.
            std::vector<bool> union_mask;
        };

        std::size_t warp_index = 0;
        std::size_t block_index = 0;
        // In this simplified SIMT model, the whole warp advances with one PC.
        std::size_t pc = 0;
        bool done = false;
        bool waiting_on_barrier = false;
        std::size_t barrier_generation = 0;
        std::vector<bool> active_mask;
        std::vector<ThreadState> threads;
        std::vector<ReconvergenceFrame> reconvergence_stack;
    };

    struct BlockState {
        std::size_t block_index = 0;
        std::size_t barrier_generation = 0;
        std::vector<std::int32_t> shared_memory;
        std::vector<WarpState> warps;
    };

    Config config_;
    std::vector<std::int32_t> global_memory_;
    Stats current_stats_;

    std::vector<BlockState> build_blocks(const Kernel& kernel) const;
    bool has_live_threads(const WarpState& warp, const std::vector<bool>& mask) const;
    bool all_threads_done(const WarpState& warp) const;
    void advance_reconvergence(WarpState& warp);
    bool step_warp(const Kernel& kernel, BlockState& block, WarpState& warp);
};

}  // namespace tinygpu
