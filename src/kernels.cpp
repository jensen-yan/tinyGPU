#include "tinygpu/kernels.h"

namespace tinygpu {

Kernel make_bootstrap_kernel() {
    Kernel kernel;
    kernel.name = "bootstrap";
    kernel.instructions = {
        Instruction{OpCode::MovImm, 0, 0, 0, 1, 0, 0},
        Instruction{OpCode::MovImm, 1, 0, 0, 2, 0, 0},
        Instruction{OpCode::Add, 2, 0, 1, 0, 0, 0},
        Instruction{OpCode::Exit, 0, 0, 0, 0, 0, 0},
    };
    return kernel;
}

Kernel make_vector_add_kernel(std::int32_t a_base, std::int32_t b_base, std::int32_t c_base) {
    Kernel kernel;
    kernel.name = "vector_add";
    kernel.instructions = {
        // r0 = global thread index
        Instruction{OpCode::MovThreadIdx, 0, 0, 0, 0, 0, 0},
        // r2 = a_base + tid, r3 = A[tid]
        Instruction{OpCode::MovImm, 1, 0, 0, a_base, 0, 0},
        Instruction{OpCode::Add, 2, 0, 1, 0, 0, 0},
        Instruction{OpCode::LoadGlobal, 3, 2, 0, 0, 0, 0},
        // r5 = b_base + tid, r6 = B[tid]
        Instruction{OpCode::MovImm, 4, 0, 0, b_base, 0, 0},
        Instruction{OpCode::Add, 5, 0, 4, 0, 0, 0},
        Instruction{OpCode::LoadGlobal, 6, 5, 0, 0, 0, 0},
        // r7 = A[tid] + B[tid]
        Instruction{OpCode::Add, 7, 3, 6, 0, 0, 0},
        // r2 = c_base + tid, then store C[tid] = r7
        Instruction{OpCode::MovImm, 1, 0, 0, c_base, 0, 0},
        Instruction{OpCode::Add, 2, 0, 1, 0, 0, 0},
        Instruction{OpCode::StoreGlobal, 0, 2, 7, 0, 0, 0},
        Instruction{OpCode::Exit, 0, 0, 0, 0, 0, 0},
    };
    return kernel;
}

Kernel make_branch_demo_kernel(std::int32_t out_base) {
    Kernel kernel;
    kernel.name = "branch_demo";
    kernel.instructions = {
        // r0 = global thread index, r2 = output address
        Instruction{OpCode::MovThreadIdx, 0, 0, 0, 0, 0, 0},
        Instruction{OpCode::MovImm, 1, 0, 0, out_base, 0, 0},
        Instruction{OpCode::Add, 2, 0, 1, 0, 0, 0},
        // r3 = tid & 1, even threads branch to the taken path
        Instruction{OpCode::AndImm, 3, 0, 0, 1, 0, 0},
        Instruction{OpCode::BranchIfZero, 0, 3, 0, 0, 8, 11},
        // odd path: write 300
        Instruction{OpCode::MovImm, 4, 0, 0, 300, 0, 0},
        Instruction{OpCode::StoreGlobal, 0, 2, 4, 0, 0, 0},
        Instruction{OpCode::Exit, 0, 0, 0, 0, 0, 0},
        // even path: write 200
        Instruction{OpCode::MovImm, 4, 0, 0, 200, 0, 0},
        Instruction{OpCode::StoreGlobal, 0, 2, 4, 0, 0, 0},
        Instruction{OpCode::Exit, 0, 0, 0, 0, 0, 0},
    };
    return kernel;
}

Kernel make_shared_exchange_kernel(std::int32_t out_base) {
    Kernel kernel;
    kernel.name = "shared_exchange";
    kernel.instructions = {
        // r0 = global thread index, r1 = block-local thread index
        Instruction{OpCode::MovThreadIdx, 0, 0, 0, 0, 0, 0},
        Instruction{OpCode::MovBlockThreadIdx, 1, 0, 0, 0, 0, 0},
        // shared[local_tid] = local_tid
        Instruction{OpCode::StoreShared, 0, 1, 1, 0, 0, 0},
        // wait until every warp in the block has filled shared memory
        Instruction{OpCode::Barrier, 0, 0, 0, 0, 0, 0},
        // r2 = local_tid ^ 32, so each thread reads a partner value from the other warp
        Instruction{OpCode::XorImm, 2, 1, 0, 32, 0, 0},
        Instruction{OpCode::LoadShared, 3, 2, 0, 0, 0, 0},
        // store partner value to out_base + global_tid
        Instruction{OpCode::MovImm, 4, 0, 0, out_base, 0, 0},
        Instruction{OpCode::Add, 5, 0, 4, 0, 0, 0},
        Instruction{OpCode::StoreGlobal, 0, 5, 3, 0, 0, 0},
        Instruction{OpCode::Exit, 0, 0, 0, 0, 0, 0},
    };
    return kernel;
}

Kernel make_block_reduction_kernel(std::int32_t in_base, std::int32_t out_base) {
    Kernel kernel;
    kernel.name = "block_reduction";
    auto& insts = kernel.instructions;

    insts.push_back(Instruction{OpCode::MovThreadIdx, 0, 0, 0, 0, 0, 0});
    insts.push_back(Instruction{OpCode::MovBlockIdx, 2, 0, 0, 0, 0, 0});
    insts.push_back(Instruction{OpCode::MovBlockThreadIdx, 1, 0, 0, 0, 0, 0});
    insts.push_back(Instruction{OpCode::MovImm, 3, 0, 0, in_base, 0, 0});
    insts.push_back(Instruction{OpCode::Add, 3, 0, 3, 0, 0, 0});
    insts.push_back(Instruction{OpCode::LoadGlobal, 4, 3, 0, 0, 0, 0});
    insts.push_back(Instruction{OpCode::StoreShared, 0, 1, 4, 0, 0, 0});
    insts.push_back(Instruction{OpCode::Barrier, 0, 0, 0, 0, 0, 0});

    const int strides[] = {32, 16, 8, 4, 2, 1};
    for (int stride : strides) {
        insts.push_back(Instruction{OpCode::SetLtImm, 5, 1, 0, stride, 0, 0});
        const std::size_t branch_index = insts.size();
        insts.push_back(Instruction{OpCode::BranchIfZero, 0, 5, 0, 0, 0, 0});
        insts.push_back(Instruction{OpCode::LoadShared, 6, 1, 0, 0, 0, 0});
        insts.push_back(Instruction{OpCode::MovImm, 7, 0, 0, stride, 0, 0});
        insts.push_back(Instruction{OpCode::Add, 7, 1, 7, 0, 0, 0});
        insts.push_back(Instruction{OpCode::LoadShared, 7, 7, 0, 0, 0, 0});
        insts.push_back(Instruction{OpCode::Add, 6, 6, 7, 0, 0, 0});
        insts.push_back(Instruction{OpCode::StoreShared, 0, 1, 6, 0, 0, 0});
        const std::uint32_t merge_pc = static_cast<std::uint32_t>(insts.size());
        insts[branch_index].target = merge_pc;
        insts[branch_index].join_target = merge_pc;
        insts.push_back(Instruction{OpCode::Barrier, 0, 0, 0, 0, 0, 0});
    }

    const std::size_t branch_index = insts.size();
    insts.push_back(Instruction{OpCode::BranchIfZero, 0, 1, 0, 0, 0, 0});
    insts.push_back(Instruction{OpCode::Exit, 0, 0, 0, 0, 0, 0});
    const std::uint32_t write_pc = static_cast<std::uint32_t>(insts.size());
    insts.push_back(Instruction{OpCode::LoadShared, 6, 1, 0, 0, 0, 0});
    insts.push_back(Instruction{OpCode::MovImm, 7, 0, 0, out_base, 0, 0});
    insts.push_back(Instruction{OpCode::Add, 7, 2, 7, 0, 0, 0});
    insts.push_back(Instruction{OpCode::StoreGlobal, 0, 7, 6, 0, 0, 0});
    insts.push_back(Instruction{OpCode::Exit, 0, 0, 0, 0, 0, 0});
    const std::uint32_t end_pc = static_cast<std::uint32_t>(insts.size());
    insts[branch_index].target = write_pc;
    insts[branch_index].join_target = end_pc;
    return kernel;
}

}  // namespace tinygpu
