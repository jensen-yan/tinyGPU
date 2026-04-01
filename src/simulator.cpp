#include "tinygpu/simulator.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace tinygpu {

Simulator::Simulator(Config config)
    : config_(config),
      global_memory_(config.global_memory_bytes, 0) {
    if (config_.warp_size == 0) {
        throw std::invalid_argument("warp_size must be greater than zero");
    }
    if (config_.threads_per_block == 0) {
        throw std::invalid_argument("threads_per_block must be greater than zero");
    }
    if (config_.threads_per_block % config_.warp_size != 0) {
        throw std::invalid_argument("threads_per_block must be divisible by warp_size in v0");
    }
    if (config_.register_count == 0) {
        throw std::invalid_argument("register_count must be greater than zero");
    }
}

std::vector<Simulator::BlockState> Simulator::build_blocks(const Kernel& kernel) const {
    (void)kernel;

    const std::size_t warps_per_block = config_.threads_per_block / config_.warp_size;
    std::vector<BlockState> blocks;
    blocks.reserve(config_.block_count);

    for (std::size_t block_index = 0; block_index < config_.block_count; ++block_index) {
        BlockState block;
        block.block_index = block_index;
        block.shared_memory.assign(config_.shared_memory_bytes, 0);
        block.warps.reserve(warps_per_block);

        for (std::size_t warp_index = 0; warp_index < warps_per_block; ++warp_index) {
            WarpState warp;
            warp.warp_index = warp_index;
            warp.block_index = block_index;
            warp.active_mask.assign(config_.warp_size, true);
            warp.threads.reserve(config_.warp_size);

            for (std::size_t lane = 0; lane < config_.warp_size; ++lane) {
                ThreadState thread;
                thread.thread_index = block_index * config_.threads_per_block + warp_index * config_.warp_size + lane;
                thread.registers.assign(config_.register_count, 0);
                warp.threads.push_back(std::move(thread));
            }

            block.warps.push_back(std::move(warp));
        }

        blocks.push_back(std::move(block));
    }

    return blocks;
}

bool Simulator::step_warp(const Kernel& kernel, WarpState& warp) {
    if (warp.done) {
        return false;
    }

    bool issued = false;

    for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
        if (!warp.active_mask[lane]) {
            continue;
        }

        ThreadState& thread = warp.threads[lane];
        if (thread.done) {
            continue;
        }
        if (thread.pc >= kernel.instructions.size()) {
            thread.done = true;
            continue;
        }

        const Instruction& inst = kernel.instructions[thread.pc];
        issued = true;

        switch (inst.opcode) {
        case OpCode::MovImm:
            thread.registers.at(inst.dst) = inst.imm;
            ++thread.pc;
            break;
        case OpCode::Add:
            thread.registers.at(inst.dst) =
                thread.registers.at(inst.src0) + thread.registers.at(inst.src1);
            ++thread.pc;
            break;
        case OpCode::Exit:
            thread.done = true;
            ++thread.pc;
            break;
        }
    }

    warp.done = std::all_of(warp.threads.begin(), warp.threads.end(), [](const ThreadState& thread) {
        return thread.done;
    });
    return issued;
}

Stats Simulator::run(const Kernel& kernel) {
    auto blocks = build_blocks(kernel);
    Stats stats;

    std::vector<WarpState*> resident_warps;
    for (auto& block : blocks) {
        for (auto& warp : block.warps) {
            resident_warps.push_back(&warp);
        }
    }

    if (resident_warps.empty()) {
        return stats;
    }

    std::size_t next_warp = 0;
    while (stats.cycles < config_.max_cycles) {
        bool any_progress = false;
        bool all_done = true;

        for (WarpState* warp : resident_warps) {
            if (!warp->done) {
                all_done = false;
                break;
            }
        }

        if (all_done) {
            break;
        }

        for (std::size_t attempt = 0; attempt < resident_warps.size(); ++attempt) {
            WarpState& warp = *resident_warps[(next_warp + attempt) % resident_warps.size()];
            if (warp.done) {
                continue;
            }
            if (step_warp(kernel, warp)) {
                ++stats.warp_issue_count;
                next_warp = (next_warp + attempt + 1) % resident_warps.size();
                any_progress = true;
                break;
            }
        }

        ++stats.cycles;
        if (!any_progress) {
            break;
        }
    }

    for (const WarpState* warp : resident_warps) {
        if (warp->done) {
            ++stats.completed_warps;
        }
    }

    return stats;
}

Kernel make_bootstrap_kernel() {
    Kernel kernel;
    kernel.name = "bootstrap";
    kernel.instructions = {
        Instruction{OpCode::MovImm, 0, 0, 0, 1},
        Instruction{OpCode::MovImm, 1, 0, 0, 2},
        Instruction{OpCode::Add, 2, 0, 1, 0},
        Instruction{OpCode::Exit, 0, 0, 0, 0},
    };
    return kernel;
}

}  // namespace tinygpu
