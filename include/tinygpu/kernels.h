#pragma once

#include "tinygpu/simulator.h"

namespace tinygpu {

Kernel make_bootstrap_kernel();
Kernel make_vector_add_kernel(std::int32_t a_base, std::int32_t b_base, std::int32_t c_base);
Kernel make_branch_demo_kernel(std::int32_t out_base);
Kernel make_shared_exchange_kernel(std::int32_t out_base);
Kernel make_block_reduction_kernel(std::int32_t in_base, std::int32_t out_base);

}  // namespace tinygpu
