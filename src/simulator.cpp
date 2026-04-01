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

std::size_t Simulator::active_lane_count(const WarpState& warp) const {
    std::size_t count = 0;
    for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
        if (warp.active_mask[lane] && !warp.threads[lane].done) {
            ++count;
        }
    }
    return count;
}

StallReason Simulator::stall_reason(const WarpState& warp) const {
    if (warp.done) {
        return StallReason::Completed;
    }
    if (warp.waiting_on_memory) {
        return StallReason::WaitingGlobalMemory;
    }
    if (warp.waiting_on_barrier) {
        return StallReason::WaitingBarrier;
    }
    return StallReason::Ready;
}

void Simulator::advance_reconvergence(WarpState& warp) {
    while (!warp.reconvergence_stack.empty()) {
        WarpState::ReconvergenceFrame& frame = warp.reconvergence_stack.back();
        // A path is considered complete either when all active lanes finished,
        // or when the warp reached the branch's designated merge point.
        const bool path_complete = !has_live_threads(warp, warp.active_mask) || warp.pc == frame.merge_pc;
        if (!path_complete) {
            return;
        }

        if (!frame.pending_started) {
            // Run the other side of the split before restoring the merged mask.
            frame.pending_started = true;
            warp.pc = frame.pending_pc;
            warp.active_mask = frame.pending_mask;
            if (has_live_threads(warp, warp.active_mask)) {
                return;
            }
            continue;
        }

        // Both sides have finished. Continue with the merged active mask.
        warp.pc = frame.merge_pc;
        warp.active_mask = frame.union_mask;
        warp.reconvergence_stack.pop_back();
    }
}

void Simulator::complete_pending_global_loads(WarpState& warp, std::size_t cycle) {
    if (!warp.waiting_on_memory || !warp.pending_global_load.valid || cycle < warp.pending_global_load.ready_cycle) {
        return;
    }

    for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
        if (!warp.pending_global_load.lane_mask[lane] || warp.threads[lane].done) {
            continue;
        }
        warp.threads[lane].registers.at(warp.pending_global_load.dst) = warp.pending_global_load.values[lane];
    }

    warp.waiting_on_memory = false;
    warp.pending_global_load = WarpState::PendingGlobalLoad{};
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
    if (config_.shared_memory_words == 0) {
        throw std::invalid_argument("shared_memory_words must be greater than zero");
    }
    if (config_.global_memory_latency == 0) {
        throw std::invalid_argument("global_memory_latency must be greater than zero");
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
        block.shared_memory.assign(config_.shared_memory_words, 0);
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

Simulator::StepResult Simulator::step_warp(
    const Kernel& kernel,
    BlockState& block,
    WarpState& warp,
    std::size_t cycle) {
    StepResult result;
    if (warp.done || warp.waiting_on_barrier || warp.waiting_on_memory) {
        return result;
    }

    // Before issuing another instruction, see whether a previously divergent
    // branch should switch to its pending path or merge back together.
    advance_reconvergence(warp);
    if (!has_live_threads(warp, warp.active_mask)) {
        warp.done = all_threads_done(warp);
        return result;
    }

    if (warp.pc >= kernel.instructions.size()) {
        warp.done = all_threads_done(warp);
        return result;
    }

    const Instruction& inst = kernel.instructions[warp.pc];
    result.issued = true;
    result.issued_pc = warp.pc;
    result.issued_opcode = inst.opcode;

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
    case OpCode::MovBlockIdx:
        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            warp.threads[lane].registers.at(inst.dst) = static_cast<std::int32_t>(warp.block_index);
        }
        ++warp.pc;
        break;
    case OpCode::MovBlockThreadIdx:
        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            // Block-local thread id = warp slot within the block + lane index.
            const std::size_t local_index = warp.warp_index * config_.warp_size + lane;
            warp.threads[lane].registers.at(inst.dst) = static_cast<std::int32_t>(local_index);
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
    case OpCode::Mul:
        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            warp.threads[lane].registers.at(inst.dst) =
                warp.threads[lane].registers.at(inst.src0) * warp.threads[lane].registers.at(inst.src1);
        }
        ++warp.pc;
        break;
    case OpCode::AndImm:
        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            warp.threads[lane].registers.at(inst.dst) = warp.threads[lane].registers.at(inst.src0) & inst.imm;
        }
        ++warp.pc;
        break;
    case OpCode::SetLtImm:
        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            warp.threads[lane].registers.at(inst.dst) =
                (warp.threads[lane].registers.at(inst.src0) < inst.imm) ? 1 : 0;
        }
        ++warp.pc;
        break;
    case OpCode::XorImm:
        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            warp.threads[lane].registers.at(inst.dst) = warp.threads[lane].registers.at(inst.src0) ^ inst.imm;
        }
        ++warp.pc;
        break;
    case OpCode::LoadGlobal:
        if (config_.global_memory_latency <= 1) {
            for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
                if (!warp.active_mask[lane] || warp.threads[lane].done) {
                    continue;
                }
                const std::size_t address = static_cast<std::size_t>(warp.threads[lane].registers.at(inst.src0));
                warp.threads[lane].registers.at(inst.dst) = static_cast<std::int32_t>(global_memory_.at(address));
                ++current_stats_.global_load_count;
            }
        } else {
            warp.waiting_on_memory = true;
            warp.pending_global_load.valid = true;
            warp.pending_global_load.ready_cycle = cycle + config_.global_memory_latency;
            warp.pending_global_load.dst = inst.dst;
            warp.pending_global_load.lane_mask.assign(warp.threads.size(), false);
            warp.pending_global_load.values.assign(warp.threads.size(), 0);

            for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
                if (!warp.active_mask[lane] || warp.threads[lane].done) {
                    continue;
                }
                const std::size_t address = static_cast<std::size_t>(warp.threads[lane].registers.at(inst.src0));
                warp.pending_global_load.lane_mask[lane] = true;
                warp.pending_global_load.values[lane] = static_cast<std::int32_t>(global_memory_.at(address));
                ++current_stats_.global_load_count;
            }
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
    case OpCode::LoadShared:
        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            const std::size_t address = static_cast<std::size_t>(warp.threads[lane].registers.at(inst.src0));
            warp.threads[lane].registers.at(inst.dst) = block.shared_memory.at(address);
            ++current_stats_.shared_load_count;
        }
        ++warp.pc;
        break;
    case OpCode::StoreShared:
        for (std::size_t lane = 0; lane < warp.threads.size(); ++lane) {
            if (!warp.active_mask[lane] || warp.threads[lane].done) {
                continue;
            }
            const std::size_t address = static_cast<std::size_t>(warp.threads[lane].registers.at(inst.src0));
            block.shared_memory.at(address) = warp.threads[lane].registers.at(inst.src1);
            ++current_stats_.shared_store_count;
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
            // Divergence: execute the taken path first, remember the fallthrough
            // path, then merge both masks at join_target.
            std::vector<bool> union_mask = taken_mask;
            for (std::size_t lane = 0; lane < union_mask.size(); ++lane) {
                union_mask[lane] = union_mask[lane] || fallthrough_mask[lane];
            }
            warp.reconvergence_stack.push_back(
                WarpState::ReconvergenceFrame{inst.join_target, warp.pc + 1, false, fallthrough_mask, union_mask});
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
    case OpCode::Barrier: {
        warp.waiting_on_barrier = true;
        warp.barrier_generation = block.barrier_generation;
        ++current_stats_.barrier_issue_count;

        // A block-level barrier releases only when every unfinished warp in the
        // block has arrived at the same barrier generation.
        bool release = true;
        for (WarpState& block_warp : block.warps) {
            if (block_warp.done) {
                continue;
            }
            if (!block_warp.waiting_on_barrier || block_warp.barrier_generation != block.barrier_generation) {
                release = false;
                break;
            }
        }

        if (release) {
            for (WarpState& block_warp : block.warps) {
                if (!block_warp.done && block_warp.waiting_on_barrier &&
                    block_warp.barrier_generation == block.barrier_generation) {
                    // Barrier is complete, so each waiting warp advances past it.
                    block_warp.waiting_on_barrier = false;
                    ++block_warp.pc;
                }
            }
            ++block.barrier_generation;
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

    advance_reconvergence(warp);
    warp.done = all_threads_done(warp);
    return result;
}

Stats Simulator::run(const Kernel& kernel) {
    return run_with_trace(kernel).stats;
}

RunReport Simulator::run_with_trace(const Kernel& kernel) {
    auto blocks = build_blocks(kernel);
    current_stats_ = Stats{};
    last_timeline_.clear();

    struct ResidentWarp {
        BlockState* block = nullptr;
        WarpState* warp = nullptr;
    };

    std::vector<ResidentWarp> resident_warps;
    for (auto& block : blocks) {
        for (auto& warp : block.warps) {
            resident_warps.push_back(ResidentWarp{&block, &warp});
        }
    }

    if (resident_warps.empty()) {
        return RunReport{current_stats_, last_timeline_};
    }

    std::size_t next_warp = 0;
    while (current_stats_.cycles < config_.max_cycles) {
        for (ResidentWarp& entry : resident_warps) {
            complete_pending_global_loads(*entry.warp, current_stats_.cycles);
        }

        bool all_done = true;
        for (const ResidentWarp& entry : resident_warps) {
            if (!entry.warp->done) {
                all_done = false;
                break;
            }
        }
        if (all_done) {
            break;
        }

        bool had_issue = false;
        std::size_t issued_entry_index = 0;
        StepResult step_result;

        for (std::size_t attempt = 0; attempt < resident_warps.size(); ++attempt) {
            const std::size_t entry_index = (next_warp + attempt) % resident_warps.size();
            ResidentWarp& entry = resident_warps[entry_index];
            WarpState& warp = *entry.warp;
            if (warp.done || warp.waiting_on_barrier || warp.waiting_on_memory) {
                continue;
            }

            // Round-robin scheduling keeps the execution order simple and visible.
            step_result = step_warp(kernel, *entry.block, warp, current_stats_.cycles);
            if (step_result.issued) {
                ++current_stats_.warp_issue_count;
                next_warp = (entry_index + 1) % resident_warps.size();
                had_issue = true;
                issued_entry_index = entry_index;
                break;
            }
        }

        CycleTrace cycle_trace;
        cycle_trace.cycle = current_stats_.cycles;
        cycle_trace.had_issue = had_issue;

        bool any_memory_wait = false;
        bool any_barrier_wait = false;
        for (std::size_t index = 0; index < resident_warps.size(); ++index) {
            const ResidentWarp& entry = resident_warps[index];
            WarpTraceState warp_trace;
            warp_trace.block_index = entry.warp->block_index;
            warp_trace.warp_index = entry.warp->warp_index;
            warp_trace.pc = entry.warp->pc;
            warp_trace.active_lanes = active_lane_count(*entry.warp);
            warp_trace.stall_reason = stall_reason(*entry.warp);
            warp_trace.issued = had_issue && index == issued_entry_index;
            if (warp_trace.issued) {
                warp_trace.issued_pc = step_result.issued_pc;
                warp_trace.issued_opcode = step_result.issued_opcode;
            }

            any_memory_wait = any_memory_wait || warp_trace.stall_reason == StallReason::WaitingGlobalMemory;
            any_barrier_wait = any_barrier_wait || warp_trace.stall_reason == StallReason::WaitingBarrier;
            cycle_trace.warps.push_back(std::move(warp_trace));
        }

        if (!had_issue) {
            ++current_stats_.idle_cycle_count;
        }
        if (any_memory_wait) {
            ++current_stats_.cycles_with_memory_wait;
        }
        if (any_barrier_wait) {
            ++current_stats_.cycles_with_barrier_wait;
        }
        last_timeline_.push_back(std::move(cycle_trace));

        ++current_stats_.cycles;

        bool has_pending_work = false;
        for (const ResidentWarp& entry : resident_warps) {
            if (!entry.warp->done && (entry.warp->waiting_on_memory || entry.warp->waiting_on_barrier)) {
                has_pending_work = true;
                break;
            }
        }
        if (!had_issue && !has_pending_work) {
            break;
        }
    }

    for (const ResidentWarp& entry : resident_warps) {
        if (entry.warp->done) {
            ++current_stats_.completed_warps;
        }
    }

    return RunReport{current_stats_, last_timeline_};
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

const std::vector<CycleTrace>& Simulator::last_timeline() const {
    return last_timeline_;
}

}  // namespace tinygpu
