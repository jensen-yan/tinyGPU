#include "tinygpu/kernel_builder.h"
#include "tinygpu/kernels.h"

namespace tinygpu {

Kernel make_bootstrap_kernel() {
    constexpr Register r0 = 0;
    constexpr Register r1 = 1;
    constexpr Register r2 = 2;

    KernelBuilder kb("bootstrap");
    kb.emit(mov_imm(r0, 1));
    kb.emit(mov_imm(r1, 2));
    kb.emit(add(r2, r0, r1));
    kb.emit(exit_kernel());
    return kb.build();
}

Kernel make_vector_add_kernel(std::int32_t a_base, std::int32_t b_base, std::int32_t c_base) {
    constexpr Register r_tid = 0;
    constexpr Register r_base = 1;
    constexpr Register r_addr = 2;
    constexpr Register r_a = 3;
    constexpr Register r_b_base = 4;
    constexpr Register r_b_addr = 5;
    constexpr Register r_b = 6;
    constexpr Register r_sum = 7;

    KernelBuilder kb("vector_add");

    // r_tid = global thread index
    kb.emit(mov_thread_idx(r_tid));
    // r_addr = a_base + tid, r_a = A[tid]
    kb.emit(mov_imm(r_base, a_base));
    kb.emit(add(r_addr, r_tid, r_base));
    kb.emit(load_global(r_a, r_addr));
    // r_b_addr = b_base + tid, r_b = B[tid]
    kb.emit(mov_imm(r_b_base, b_base));
    kb.emit(add(r_b_addr, r_tid, r_b_base));
    kb.emit(load_global(r_b, r_b_addr));
    // r_sum = A[tid] + B[tid]
    kb.emit(add(r_sum, r_a, r_b));
    // r_addr = c_base + tid, then store C[tid] = r_sum
    kb.emit(mov_imm(r_base, c_base));
    kb.emit(add(r_addr, r_tid, r_base));
    kb.emit(store_global(r_addr, r_sum));
    kb.emit(exit_kernel());
    return kb.build();
}

Kernel make_branch_demo_kernel(std::int32_t out_base) {
    constexpr Register r_tid = 0;
    constexpr Register r_base = 1;
    constexpr Register r_addr = 2;
    constexpr Register r_pred = 3;
    constexpr Register r_value = 4;

    KernelBuilder kb("branch_demo");
    const Label even_path{"even_path"};
    const Label done{"done"};

    // r_tid = global thread index, r_addr = out_base + tid
    kb.emit(mov_thread_idx(r_tid));
    kb.emit(mov_imm(r_base, out_base));
    kb.emit(add(r_addr, r_tid, r_base));
    // r_pred = tid & 1, even threads branch to the taken path
    kb.emit(and_imm(r_pred, r_tid, 1));
    kb.emit_branch_if_zero(r_pred, even_path, done);
    // odd path: write 300
    kb.emit(mov_imm(r_value, 300));
    kb.emit(store_global(r_addr, r_value));
    kb.emit(exit_kernel());
    // even path: write 200
    kb.bind(even_path);
    kb.emit(mov_imm(r_value, 200));
    kb.emit(store_global(r_addr, r_value));
    kb.emit(exit_kernel());
    kb.bind(done);
    return kb.build();
}

Kernel make_shared_exchange_kernel(std::int32_t out_base) {
    constexpr Register r_tid = 0;
    constexpr Register r_local = 1;
    constexpr Register r_partner = 2;
    constexpr Register r_value = 3;
    constexpr Register r_out_base = 4;
    constexpr Register r_out_addr = 5;

    KernelBuilder kb("shared_exchange");

    // r_tid = global thread index, r_local = block-local thread index
    kb.emit(mov_thread_idx(r_tid));
    kb.emit(mov_block_thread_idx(r_local));
    // shared[local_tid] = local_tid
    kb.emit(store_shared(r_local, r_local));
    // wait until every warp in the block has filled shared memory
    kb.emit(barrier());
    // r_partner = local_tid ^ 32, so each thread reads a partner value from the other warp
    kb.emit(xor_imm(r_partner, r_local, 32));
    kb.emit(load_shared(r_value, r_partner));
    // store partner value to out_base + global_tid
    kb.emit(mov_imm(r_out_base, out_base));
    kb.emit(add(r_out_addr, r_tid, r_out_base));
    kb.emit(store_global(r_out_addr, r_value));
    kb.emit(exit_kernel());
    return kb.build();
}

Kernel make_block_reduction_kernel(std::int32_t in_base, std::int32_t out_base) {
    constexpr Register r_tid = 0;     // Global thread index, used for input addressing.
    constexpr Register r_local = 1;   // Block-local thread index, used for shared-memory slots.
    constexpr Register r_block = 2;   // Block index, used for the final block-sum writeback.
    constexpr Register r_addr = 3;    // Reusable address register for global-memory accesses.
    constexpr Register r_acc = 4;     // Current value loaded from global/shared memory.
    constexpr Register r_pred = 5;    // Predicate register: 1 when this lane participates in a reduction step.
    constexpr Register r_tmp = 6;     // Temporary value loaded from shared memory.
    constexpr Register r_offset = 7;  // Temporary address/offset register such as local_tid + stride.

    KernelBuilder kb("block_reduction");

    // Stage 1: each thread loads one input element and publishes it into
    // block-local shared memory. After the barrier, shared[local_tid]
    // contains the initial reduction state for this block.
    kb.emit(mov_thread_idx(r_tid));
    kb.emit(mov_block_idx(r_block));
    kb.emit(mov_block_thread_idx(r_local));
    kb.emit(mov_imm(r_addr, in_base));
    kb.emit(add(r_addr, r_tid, r_addr));
    kb.emit(load_global(r_acc, r_addr));
    kb.emit(store_shared(r_local, r_acc));
    kb.emit(barrier());

    const int strides[] = {32, 16, 8, 4, 2, 1};
    for (int stride : strides) {
        // Stage 2: tree reduction in shared memory.
        // Only threads with local_tid < stride participate in this round.
        // They accumulate shared[tid + stride] into shared[tid], then the
        // block synchronizes before the next stride begins.
        const Label skip_round{"skip_stride_" + std::to_string(stride)};
        const Label round_merge{"round_merge_" + std::to_string(stride)};
        kb.emit(set_lt_imm(r_pred, r_local, stride));
        kb.emit_branch_if_zero(r_pred, skip_round, round_merge);
        kb.emit(load_shared(r_tmp, r_local));
        kb.emit(mov_imm(r_offset, stride));
        kb.emit(add(r_offset, r_local, r_offset));
        kb.emit(load_shared(r_offset, r_offset));
        kb.emit(add(r_tmp, r_tmp, r_offset));
        kb.emit(store_shared(r_local, r_tmp));
        kb.bind(skip_round);
        kb.bind(round_merge);
        kb.emit(barrier());
    }

    // Stage 3: one thread per block writes the final block sum back to
    // global memory at out_base + blockIdx.
    const Label write_block_sum{"write_block_sum"};
    const Label reduction_done{"reduction_done"};
    kb.emit_branch_if_zero(r_local, write_block_sum, reduction_done);
    kb.emit(exit_kernel());
    kb.bind(write_block_sum);
    kb.emit(load_shared(r_tmp, r_local));
    kb.emit(mov_imm(r_offset, out_base));
    kb.emit(add(r_offset, r_block, r_offset));
    kb.emit(store_global(r_offset, r_tmp));
    kb.emit(exit_kernel());
    kb.bind(reduction_done);
    return kb.build();
}

Kernel make_tiled_matmul_kernel(std::int32_t a_base, std::int32_t b_base, std::int32_t c_base) {
    constexpr Register r_tid = 0;       // Global thread index, useful for debug symmetry with other kernels.
    constexpr Register r_local = 1;     // Block-local thread index in [0, 63], one thread per output element.
    constexpr Register r_tmp = 2;       // General-purpose immediate scratch register.
    constexpr Register r_addr = 3;      // Reusable global/shared address register.
    constexpr Register r_acc = 4;       // Accumulator for C[row][col].
    constexpr Register r_row_base = 5;  // row * 8, used to index one row of the A tile.
    constexpr Register r_col = 6;       // col within the 8x8 tile.
    constexpr Register r_a = 7;         // Current A[row][k] value loaded from shared memory.
    constexpr Register r_b = 8;         // Current B[k][col] value loaded from shared memory.
    constexpr Register r_mul = 9;       // Temporary product A[row][k] * B[k][col].

    KernelBuilder kb("tiled_matmul");

    // v0 matmul uses one 8x8 block to compute one 8x8 output tile.
    // Each thread owns one output element C[row][col], where:
    //   row = local_tid / 8
    //   col = local_tid % 8
    //
    // Shared layout:
    //   shared[0..63]   -> A tile
    //   shared[64..127] -> B tile

    // Stage 1: each thread loads one A element and one B element from
    // global memory into block-local shared memory.
    kb.emit(mov_thread_idx(r_tid));
    kb.emit(mov_block_thread_idx(r_local));

    kb.emit(mov_imm(r_tmp, a_base));
    kb.emit(add(r_addr, r_local, r_tmp));
    kb.emit(load_global(r_acc, r_addr));
    kb.emit(store_shared(r_local, r_acc));

    kb.emit(mov_imm(r_tmp, b_base));
    kb.emit(add(r_addr, r_local, r_tmp));
    kb.emit(load_global(r_acc, r_addr));
    kb.emit(mov_imm(r_tmp, 64));
    kb.emit(add(r_addr, r_local, r_tmp));
    kb.emit(store_shared(r_addr, r_acc));
    kb.emit(barrier());

    // Stage 2: derive row_base and col from the block-local thread id.
    kb.emit(mov_imm(r_acc, 0));          // accumulator
    kb.emit(and_imm(r_row_base, r_local, 56));  // row * 8
    kb.emit(and_imm(r_col, r_local, 7));        // col

    // Stage 3: unrolled dot product across the shared A/B tiles.
    for (int k = 0; k < 8; ++k) {
        kb.emit(mov_imm(r_tmp, k));
        kb.emit(add(r_addr, r_row_base, r_tmp));
        kb.emit(load_shared(r_a, r_addr));

        kb.emit(mov_imm(r_tmp, 64 + k * 8));
        kb.emit(add(r_addr, r_col, r_tmp));
        kb.emit(load_shared(r_b, r_addr));

        kb.emit(mul(r_mul, r_a, r_b));
        kb.emit(add(r_acc, r_acc, r_mul));
    }

    // Stage 4: write C[row][col] back to global memory.
    kb.emit(mov_imm(r_tmp, c_base));
    kb.emit(add(r_addr, r_local, r_tmp));
    kb.emit(store_global(r_addr, r_acc));
    kb.emit(exit_kernel());
    return kb.build();
}

}  // namespace tinygpu
