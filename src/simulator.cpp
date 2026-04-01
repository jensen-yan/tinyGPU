#include "tinygpu/simulator.h"

#include <algorithm>
#include <stdexcept>

namespace tinygpu {
bool Simulator::has_live_threads(const WarpState& warp, const std::vector<bool>& mask) const {
    for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
        if (mask[lane] && !warp.threads[lane].done) {
            return true;
        }
    }
    return false;
}

bool Simulator::all_threads_done(const WarpState& warp) const {
    return std::all_of(warp.threads.begin(), warp.threads.end(), [](const ThreadState& thread) {
        return thread.done;
    });
}

void Simulator::restore_next_path(WarpState& warp) {
    while (!warp.pending_paths.empty()) {
        WarpState::ExecContext ctx = std::move(warp.pending_paths.back());
        warp.pending_paths.pop_back();
        if (has_live_threads(warp, ctx.active_mask)) {
            warp.pc = ctx.pc;
            warp.active_mask = std::move(ctx.active_mask);
            return;
        }
    }
}

Simulator::Simulator(Config config)
    : config_(config),
      global_memory_(config.global_memory_words, 0) {
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
    if (config_.global_memory_words == 0) {
        throw std::invalid_argument("global_memory_words must be greater than zero");
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
            warp.pc = 0;
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

    if (!has_live_threads(warp, warp.active_mask)) {
        restore_next_path(warp);
        if (!has_live_threads(warp, warp.active_mask)) {
            warp.done = all_threads_done(warp);
            return false;
        }
    }

    if (warp.pc >= kernel.instructions.size()) {
        warp.done = all_threads_done(warp);
        return false;
    }

    bool issued = false;
    const Instruction& inst = kernel.instructions[warp.pc];
    issued = true;

    switch (inst.opcode) {
    case OpCode::MovImm:
        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            warp.threads[lane].registers.at(inst.dst) = inst.imm;
        }
        ++warp.pc;
        break;
    case OpCode::MovThreadIdx:
        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            warp.threads[lane].registers.at(inst.dst) = static_cast<std::int32_t>(warp.threads[lane].thread_index);
        }
        ++warp.pc;
        break;
    case OpCode::Add:
        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            warp.threads[lane].registers.at(inst.dst) =
                warp.threads[lane].registers.at(inst.src0) + warp.threads[lane].registers.at(inst.src1);
        }
        ++warp.pc;
        break;
    case OpCode::AndImm:
        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            warp.threads[lane].registers.at(inst.dst) =
                warp.threads[lane].registers.at(inst.src0) & inst.imm;
        }
        ++warp.pc;
        break;
    case OpCode::LoadGlobal:
        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            const std::size_t address = static_cast<std::size_t>(warp.threads[lane].registers.at(inst.src0));
            warp.threads[lane].registers.at(inst.dst) = static_cast<std::int32_t>(global_memory_.at(address));
            ++current_stats_.global_load_count;
        }
        ++warp.pc;
        break;
    case OpCode::StoreGlobal:
        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            const std::size_t address = static_cast<std::size_t>(warp.threads[lane].registers.at(inst.src0));
            global_memory_.at(address) = warp.threads[lane].registers.at(inst.src1);
            ++current_stats_.global_store_count;
        }
        ++warp.pc;
        break;
    case OpCode::BranchIfZero: {
        std::vector<bool> taken_mask(warp.active_mask.size(), false);
        std::vector<bool> fallthrough_mask(warp.active_mask.size(), false);

        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            if (warp.threads[lane].registers.at(inst.src0) == 0) {
                taken_mask[lane] = true;
            } else {
                fallthrough_mask[lane] = true;
            }
        }

        const bool any_taken = has_live_threads(warp, taken_mask);
        const bool any_fallthrough = has_live_threads(warp, fallthrough_mask);

        if (any_taken && any_fallthrough) {
            warp.pending_paths.push_back(WarpState::ExecContext{warp.pc + 1, fallthrough_mask});
            warp.active_mask = std::move(taken_mask);
            warp.pc = inst.target;
            ++current_stats_.divergent_branch_count;
        } else if (any_taken) {
            warp.active_mask = std::move(taken_mask);
            warp.pc = inst.target;
        } else {
            warp.active_mask = std::move(fallthrough_mask);
            ++warp.pc;
        }
        break;
    }
    case OpCode::Exit:
        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            warp.threads[lane].done = true;
        }
        ++warp.pc;
        break;
    }

    if (!has_live_threads(warp, warp.active_mask)) {
        restore_next_path(warp);
    }

    warp.done = all_threads_done(warp);
    return issued;
}

Stats Simulator::run(const Kernel& kernel) {
    auto blocks = build_blocks(kernel);
    current_stats_ = Stats{};

    std::vector<WarpState*> resident_warps;
    for (auto& block : blocks) {
        for (auto& warp : block.warps) {
            resident_warps.push_back(&warp);
        }
    }

    if (resident_warps.empty()) {
        return current_stats_;
    }

    std::size_t next_warp = 0;
    while (current_stats_.cycles < config_.max_cycles) {
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
                ++current_stats_.warp_issue_count;
                next_warp = (next_warp + attempt + 1) % resident_warps.size();
                any_progress = true;
                break;
            }
        }

        ++current_stats_.cycles;
        if (!any_progress) {
            break;
        }
    }

    for (const WarpState* warp : resident_warps) {
        if (warp->done) {
            ++current_stats_.completed_warps;
        }
    }

    return current_stats_;
}

void Simulator::write_global(std::size_t index, std::int32_t value) {
    global_memory_.at(index) = value;
}

std::int32_t Simulator::read_global(std::size_t index) const {
    return global_memory_.at(index);
}

std::size_t Simulator::global_memory_size() const {
    return global_memory_.size();
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

Kernel make_vector_add_kernel(std::int32_t a_base, std::int32_t b_base, std::int32_t c_base) {
    Kernel kernel;
    kernel.name = "vector_add";
    kernel.instructions = {
        Instruction{OpCode::MovThreadIdx, 0, 0, 0, 0},
        Instruction{OpCode::MovImm, 1, 0, 0, a_base},
        Instruction{OpCode::Add, 2, 0, 1, 0},
        Instruction{OpCode::LoadGlobal, 3, 2, 0, 0},
        Instruction{OpCode::MovImm, 4, 0, 0, b_base},
        Instruction{OpCode::Add, 5, 0, 4, 0},
        Instruction{OpCode::LoadGlobal, 6, 5, 0, 0},
        Instruction{OpCode::Add, 7, 3, 6, 0},
        Instruction{OpCode::MovImm, 1, 0, 0, c_base},
        Instruction{OpCode::Add, 2, 0, 1, 0},
        Instruction{OpCode::StoreGlobal, 2, 2, 7, 0},
        Instruction{OpCode::Exit, 0, 0, 0, 0},
    };
    return kernel;
}

Kernel make_branch_demo_kernel(std::int32_t out_base) {
    Kernel kernel;
    kernel.name = "branch_demo";
    kernel.instructions = {
        Instruction{OpCode::MovThreadIdx, 0, 0, 0, 0, 0},
        Instruction{OpCode::MovImm, 1, 0, 0, out_base, 0},
        Instruction{OpCode::Add, 2, 0, 1, 0, 0},
        Instruction{OpCode::AndImm, 3, 0, 0, 1, 0},
        Instruction{OpCode::BranchIfZero, 0, 3, 0, 0, 8},
        Instruction{OpCode::MovImm, 4, 0, 0, 300, 0},
        Instruction{OpCode::StoreGlobal, 0, 2, 4, 0, 0},
        Instruction{OpCode::Exit, 0, 0, 0, 0, 0},
        Instruction{OpCode::MovImm, 4, 0, 0, 200, 0},
        Instruction{OpCode::StoreGlobal, 0, 2, 4, 0, 0},
        Instruction{OpCode::Exit, 0, 0, 0, 0, 0},
    };
    return kernel;
}

}  // namespace tinygpu
