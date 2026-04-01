#include "tinygpu/disasm.h"

#include <sstream>

namespace tinygpu {

namespace {

std::string reg_name(std::uint32_t reg) {
    return "r" + std::to_string(reg);
}

}  // namespace

std::string format_instruction(std::size_t pc, const Instruction& inst) {
    std::ostringstream os;
    os << pc << ": ";

    switch (inst.opcode) {
    case OpCode::MovImm:
        os << "mov_imm " << reg_name(inst.dst) << ", " << inst.imm;
        break;
    case OpCode::MovThreadIdx:
        os << "mov_thread_idx " << reg_name(inst.dst);
        break;
    case OpCode::MovBlockIdx:
        os << "mov_block_idx " << reg_name(inst.dst);
        break;
    case OpCode::MovBlockThreadIdx:
        os << "mov_block_thread_idx " << reg_name(inst.dst);
        break;
    case OpCode::Add:
        os << "add " << reg_name(inst.dst) << ", " << reg_name(inst.src0) << ", " << reg_name(inst.src1);
        break;
    case OpCode::Mul:
        os << "mul " << reg_name(inst.dst) << ", " << reg_name(inst.src0) << ", " << reg_name(inst.src1);
        break;
    case OpCode::AndImm:
        os << "and_imm " << reg_name(inst.dst) << ", " << reg_name(inst.src0) << ", " << inst.imm;
        break;
    case OpCode::SetLtImm:
        os << "set_lt_imm " << reg_name(inst.dst) << ", " << reg_name(inst.src0) << ", " << inst.imm;
        break;
    case OpCode::XorImm:
        os << "xor_imm " << reg_name(inst.dst) << ", " << reg_name(inst.src0) << ", " << inst.imm;
        break;
    case OpCode::LoadGlobal:
        os << "load_global " << reg_name(inst.dst) << ", [" << reg_name(inst.src0) << "]";
        break;
    case OpCode::StoreGlobal:
        os << "store_global [" << reg_name(inst.src0) << "], " << reg_name(inst.src1);
        break;
    case OpCode::LoadShared:
        os << "load_shared " << reg_name(inst.dst) << ", [" << reg_name(inst.src0) << "]";
        break;
    case OpCode::StoreShared:
        os << "store_shared [" << reg_name(inst.src0) << "], " << reg_name(inst.src1);
        break;
    case OpCode::BranchIfZero:
        os << "branch_if_zero " << reg_name(inst.src0)
           << ", target=" << inst.target
           << ", join=" << inst.join_target;
        break;
    case OpCode::Barrier:
        os << "barrier";
        break;
    case OpCode::Exit:
        os << "exit";
        break;
    }

    return os.str();
}

std::string disassemble_kernel(const Kernel& kernel) {
    std::ostringstream os;
    os << "kernel " << kernel.name << "\n";
    for (std::size_t pc = 0; pc < kernel.instructions.size(); ++pc) {
        os << format_instruction(pc, kernel.instructions[pc]) << "\n";
    }
    return os.str();
}

}  // namespace tinygpu
