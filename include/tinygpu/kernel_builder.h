#pragma once

#include "tinygpu/simulator.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tinygpu {

using Register = std::uint32_t;

struct Label {
    std::string name;
};

class KernelBuilder {
public:
    explicit KernelBuilder(std::string kernel_name)
        : kernel_name_(std::move(kernel_name)) {}

    void emit(const Instruction& inst) {
        instructions_.push_back(inst);
    }

    void bind(const Label& label) {
        label_to_pc_[label.name] = instructions_.size();
    }

    void emit_branch_if_zero(Register src, const Label& target, const Label& join) {
        unresolved_branches_.push_back(UnresolvedBranch{
            instructions_.size(),
            target.name,
            join.name,
        });
        instructions_.push_back(Instruction{OpCode::BranchIfZero, 0, src, 0, 0, 0, 0});
    }

    Kernel build() const {
        Kernel kernel;
        kernel.name = kernel_name_;
        kernel.instructions = instructions_;

        for (const UnresolvedBranch& branch : unresolved_branches_) {
            const auto target_it = label_to_pc_.find(branch.target_label);
            if (target_it == label_to_pc_.end()) {
                throw std::runtime_error("unknown branch target label: " + branch.target_label);
            }
            const auto join_it = label_to_pc_.find(branch.join_label);
            if (join_it == label_to_pc_.end()) {
                throw std::runtime_error("unknown branch join label: " + branch.join_label);
            }
            kernel.instructions.at(branch.inst_index).target = static_cast<std::uint32_t>(target_it->second);
            kernel.instructions.at(branch.inst_index).join_target = static_cast<std::uint32_t>(join_it->second);
        }

        return kernel;
    }

private:
    struct UnresolvedBranch {
        std::size_t inst_index = 0;
        std::string target_label;
        std::string join_label;
    };

    std::string kernel_name_;
    std::vector<Instruction> instructions_;
    std::unordered_map<std::string, std::size_t> label_to_pc_;
    std::vector<UnresolvedBranch> unresolved_branches_;
};

inline Instruction mov_imm(Register dst, std::int32_t imm) {
    return Instruction{OpCode::MovImm, dst, 0, 0, imm, 0, 0};
}

inline Instruction mov_thread_idx(Register dst) {
    return Instruction{OpCode::MovThreadIdx, dst, 0, 0, 0, 0, 0};
}

inline Instruction mov_block_idx(Register dst) {
    return Instruction{OpCode::MovBlockIdx, dst, 0, 0, 0, 0, 0};
}

inline Instruction mov_block_thread_idx(Register dst) {
    return Instruction{OpCode::MovBlockThreadIdx, dst, 0, 0, 0, 0, 0};
}

inline Instruction add(Register dst, Register src0, Register src1) {
    return Instruction{OpCode::Add, dst, src0, src1, 0, 0, 0};
}

inline Instruction mul(Register dst, Register src0, Register src1) {
    return Instruction{OpCode::Mul, dst, src0, src1, 0, 0, 0};
}

inline Instruction and_imm(Register dst, Register src0, std::int32_t imm) {
    return Instruction{OpCode::AndImm, dst, src0, 0, imm, 0, 0};
}

inline Instruction set_lt_imm(Register dst, Register src0, std::int32_t imm) {
    return Instruction{OpCode::SetLtImm, dst, src0, 0, imm, 0, 0};
}

inline Instruction xor_imm(Register dst, Register src0, std::int32_t imm) {
    return Instruction{OpCode::XorImm, dst, src0, 0, imm, 0, 0};
}

inline Instruction load_global(Register dst, Register addr) {
    return Instruction{OpCode::LoadGlobal, dst, addr, 0, 0, 0, 0};
}

inline Instruction store_global(Register addr, Register src) {
    return Instruction{OpCode::StoreGlobal, 0, addr, src, 0, 0, 0};
}

inline Instruction load_shared(Register dst, Register addr) {
    return Instruction{OpCode::LoadShared, dst, addr, 0, 0, 0, 0};
}

inline Instruction store_shared(Register addr, Register src) {
    return Instruction{OpCode::StoreShared, 0, addr, src, 0, 0, 0};
}

inline Instruction branch_if_zero(Register src, std::uint32_t target, std::uint32_t join) {
    return Instruction{OpCode::BranchIfZero, 0, src, 0, 0, target, join};
}

inline Instruction barrier() {
    return Instruction{OpCode::Barrier, 0, 0, 0, 0, 0, 0};
}

inline Instruction exit_kernel() {
    return Instruction{OpCode::Exit, 0, 0, 0, 0, 0, 0};
}

}  // namespace tinygpu
