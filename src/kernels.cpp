#include "tinygpu/kernels.h"

namespace tinygpu {

Kernel make_bootstrap_kernel() {
    Kernel kernel;
    kernel.name = "bootstrap";
    kernel.instructions = {
        Instruction{OpCode::MovImm, 0, 0, 0, 1, 0},
        Instruction{OpCode::MovImm, 1, 0, 0, 2, 0},
        Instruction{OpCode::Add, 2, 0, 1, 0, 0},
        Instruction{OpCode::Exit, 0, 0, 0, 0, 0},
    };
    return kernel;
}

Kernel make_vector_add_kernel(std::int32_t a_base, std::int32_t b_base, std::int32_t c_base) {
    Kernel kernel;
    kernel.name = "vector_add";
    kernel.instructions = {
        // r0 = global thread index
        Instruction{OpCode::MovThreadIdx, 0, 0, 0, 0, 0},
        // r2 = a_base + tid, r3 = A[tid]
        Instruction{OpCode::MovImm, 1, 0, 0, a_base, 0},
        Instruction{OpCode::Add, 2, 0, 1, 0, 0},
        Instruction{OpCode::LoadGlobal, 3, 2, 0, 0, 0},
        // r5 = b_base + tid, r6 = B[tid]
        Instruction{OpCode::MovImm, 4, 0, 0, b_base, 0},
        Instruction{OpCode::Add, 5, 0, 4, 0, 0},
        Instruction{OpCode::LoadGlobal, 6, 5, 0, 0, 0},
        // r7 = A[tid] + B[tid]
        Instruction{OpCode::Add, 7, 3, 6, 0, 0},
        // r2 = c_base + tid, then store C[tid] = r7
        Instruction{OpCode::MovImm, 1, 0, 0, c_base, 0},
        Instruction{OpCode::Add, 2, 0, 1, 0, 0},
        Instruction{OpCode::StoreGlobal, 0, 2, 7, 0, 0},
        Instruction{OpCode::Exit, 0, 0, 0, 0, 0},
    };
    return kernel;
}

Kernel make_branch_demo_kernel(std::int32_t out_base) {
    Kernel kernel;
    kernel.name = "branch_demo";
    kernel.instructions = {
        // r0 = global thread index, r2 = output address
        Instruction{OpCode::MovThreadIdx, 0, 0, 0, 0, 0},
        Instruction{OpCode::MovImm, 1, 0, 0, out_base, 0},
        Instruction{OpCode::Add, 2, 0, 1, 0, 0},
        // r3 = tid & 1, even threads branch to the taken path
        Instruction{OpCode::AndImm, 3, 0, 0, 1, 0},
        Instruction{OpCode::BranchIfZero, 0, 3, 0, 0, 8},
        // odd path: write 300
        Instruction{OpCode::MovImm, 4, 0, 0, 300, 0},
        Instruction{OpCode::StoreGlobal, 0, 2, 4, 0, 0},
        Instruction{OpCode::Exit, 0, 0, 0, 0, 0},
        // even path: write 200
        Instruction{OpCode::MovImm, 4, 0, 0, 200, 0},
        Instruction{OpCode::StoreGlobal, 0, 2, 4, 0, 0},
        Instruction{OpCode::Exit, 0, 0, 0, 0, 0},
    };
    return kernel;
}

Kernel make_shared_exchange_kernel(std::int32_t out_base) {
    Kernel kernel;
    kernel.name = "shared_exchange";
    kernel.instructions = {
        // r0 = global thread index, r1 = block-local thread index
        Instruction{OpCode::MovThreadIdx, 0, 0, 0, 0, 0},
        Instruction{OpCode::MovBlockThreadIdx, 1, 0, 0, 0, 0},
        // shared[local_tid] = local_tid
        Instruction{OpCode::StoreShared, 0, 1, 1, 0, 0},
        // wait until every warp in the block has filled shared memory
        Instruction{OpCode::Barrier, 0, 0, 0, 0, 0},
        // r2 = local_tid ^ 32, so each thread reads a partner value from the other warp
        Instruction{OpCode::XorImm, 2, 1, 0, 32, 0},
        Instruction{OpCode::LoadShared, 3, 2, 0, 0, 0},
        // store partner value to out_base + global_tid
        Instruction{OpCode::MovImm, 4, 0, 0, out_base, 0},
        Instruction{OpCode::Add, 5, 0, 4, 0, 0},
        Instruction{OpCode::StoreGlobal, 0, 5, 3, 0, 0},
        Instruction{OpCode::Exit, 0, 0, 0, 0, 0},
    };
    return kernel;
}

}  // namespace tinygpu
