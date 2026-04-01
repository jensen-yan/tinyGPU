#include "tinygpu/kernels.h"
#include "tinygpu/simulator.h"

#include <exception>
#include <iostream>

namespace {

bool run_vector_add_demo(tinygpu::Simulator& simulator, const tinygpu::Config& config) {
    const std::size_t element_count = config.block_count * config.threads_per_block;
    const std::size_t a_base = 0;
    const std::size_t b_base = a_base + element_count;
    const std::size_t c_base = b_base + element_count;

    if (c_base + element_count > simulator.global_memory_size()) {
        std::cerr << "fatal: global memory is too small for vector add demo\n";
        return false;
    }

    for (std::size_t i = 0; i < element_count; ++i) {
        simulator.write_global(a_base + i, static_cast<std::int32_t>(i));
        simulator.write_global(b_base + i, static_cast<std::int32_t>(1000 + i));
        simulator.write_global(c_base + i, -1);
    }

    const tinygpu::Kernel kernel = tinygpu::make_vector_add_kernel(
        static_cast<std::int32_t>(a_base),
        static_cast<std::int32_t>(b_base),
        static_cast<std::int32_t>(c_base));
    const tinygpu::Stats stats = simulator.run(kernel);

    bool ok = true;
    for (std::size_t i = 0; i < element_count; ++i) {
        const std::int32_t expected = static_cast<std::int32_t>(i + (1000 + i));
        if (simulator.read_global(c_base + i) != expected) {
            ok = false;
            break;
        }
    }

    std::cout << "tinyGPU vector add demo\n";
    std::cout << "kernel: " << kernel.name << "\n";
    std::cout << "threads: " << element_count << "\n";
    std::cout << "cycles: " << stats.cycles << "\n";
    std::cout << "warp_issues: " << stats.warp_issue_count << "\n";
    std::cout << "global_loads: " << stats.global_load_count << "\n";
    std::cout << "global_stores: " << stats.global_store_count << "\n";
    std::cout << "shared_loads: " << stats.shared_load_count << "\n";
    std::cout << "shared_stores: " << stats.shared_store_count << "\n";
    std::cout << "barriers: " << stats.barrier_issue_count << "\n";
    std::cout << "divergent_branches: " << stats.divergent_branch_count << "\n";
    std::cout << "completed_warps: " << stats.completed_warps << "\n";
    std::cout << "sample C[0]: " << simulator.read_global(c_base) << "\n";
    std::cout << "sample C[last]: " << simulator.read_global(c_base + element_count - 1) << "\n";
    std::cout << "status: " << (ok ? "PASS" : "FAIL") << "\n\n";
    return ok;
}

bool run_branch_demo(tinygpu::Simulator& simulator, const tinygpu::Config& config) {
    const std::size_t element_count = config.block_count * config.threads_per_block;
    const std::size_t out_base = 3 * element_count;

    if (out_base + element_count > simulator.global_memory_size()) {
        std::cerr << "fatal: global memory is too small for branch demo\n";
        return false;
    }

    for (std::size_t i = 0; i < element_count; ++i) {
        simulator.write_global(out_base + i, -1);
    }

    const tinygpu::Kernel kernel = tinygpu::make_branch_demo_kernel(static_cast<std::int32_t>(out_base));
    const tinygpu::Stats stats = simulator.run(kernel);

    bool ok = true;
    for (std::size_t i = 0; i < element_count; ++i) {
        const std::int32_t expected = (i % 2 == 0) ? 200 : 300;
        if (simulator.read_global(out_base + i) != expected) {
            ok = false;
            break;
        }
    }

    std::cout << "tinyGPU branch demo\n";
    std::cout << "kernel: " << kernel.name << "\n";
    std::cout << "threads: " << element_count << "\n";
    std::cout << "cycles: " << stats.cycles << "\n";
    std::cout << "warp_issues: " << stats.warp_issue_count << "\n";
    std::cout << "global_loads: " << stats.global_load_count << "\n";
    std::cout << "global_stores: " << stats.global_store_count << "\n";
    std::cout << "shared_loads: " << stats.shared_load_count << "\n";
    std::cout << "shared_stores: " << stats.shared_store_count << "\n";
    std::cout << "barriers: " << stats.barrier_issue_count << "\n";
    std::cout << "divergent_branches: " << stats.divergent_branch_count << "\n";
    std::cout << "completed_warps: " << stats.completed_warps << "\n";
    std::cout << "sample out[0]: " << simulator.read_global(out_base) << "\n";
    std::cout << "sample out[1]: " << simulator.read_global(out_base + 1) << "\n";
    std::cout << "status: " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool run_shared_exchange_demo(tinygpu::Simulator& simulator, const tinygpu::Config& config) {
    const std::size_t element_count = config.block_count * config.threads_per_block;
    const std::size_t out_base = 4 * element_count;

    if (out_base + element_count > simulator.global_memory_size()) {
        std::cerr << "fatal: global memory is too small for shared exchange demo\n";
        return false;
    }

    for (std::size_t i = 0; i < element_count; ++i) {
        simulator.write_global(out_base + i, -1);
    }

    const tinygpu::Kernel kernel = tinygpu::make_shared_exchange_kernel(static_cast<std::int32_t>(out_base));
    const tinygpu::Stats stats = simulator.run(kernel);

    bool ok = true;
    for (std::size_t i = 0; i < element_count; ++i) {
        const std::size_t local = i % config.threads_per_block;
        const std::int32_t expected = static_cast<std::int32_t>(local ^ 32U);
        if (simulator.read_global(out_base + i) != expected) {
            ok = false;
            break;
        }
    }

    std::cout << "tinyGPU shared exchange demo\n";
    std::cout << "kernel: " << kernel.name << "\n";
    std::cout << "threads: " << element_count << "\n";
    std::cout << "cycles: " << stats.cycles << "\n";
    std::cout << "warp_issues: " << stats.warp_issue_count << "\n";
    std::cout << "global_loads: " << stats.global_load_count << "\n";
    std::cout << "global_stores: " << stats.global_store_count << "\n";
    std::cout << "shared_loads: " << stats.shared_load_count << "\n";
    std::cout << "shared_stores: " << stats.shared_store_count << "\n";
    std::cout << "barriers: " << stats.barrier_issue_count << "\n";
    std::cout << "divergent_branches: " << stats.divergent_branch_count << "\n";
    std::cout << "completed_warps: " << stats.completed_warps << "\n";
    std::cout << "sample out[0]: " << simulator.read_global(out_base) << "\n";
    std::cout << "sample out[32]: " << simulator.read_global(out_base + 32) << "\n";
    std::cout << "status: " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

bool run_block_reduction_demo(tinygpu::Simulator& simulator, const tinygpu::Config& config) {
    const std::size_t element_count = config.block_count * config.threads_per_block;
    const std::size_t in_base = 5 * element_count;
    const std::size_t out_base = in_base + element_count;

    if (out_base + config.block_count > simulator.global_memory_size()) {
        std::cerr << "fatal: global memory is too small for block reduction demo\n";
        return false;
    }

    for (std::size_t i = 0; i < element_count; ++i) {
        simulator.write_global(in_base + i, 1);
    }
    for (std::size_t block = 0; block < config.block_count; ++block) {
        simulator.write_global(out_base + block, -1);
    }

    const tinygpu::Kernel kernel = tinygpu::make_block_reduction_kernel(
        static_cast<std::int32_t>(in_base),
        static_cast<std::int32_t>(out_base));
    const tinygpu::Stats stats = simulator.run(kernel);

    bool ok = true;
    for (std::size_t block = 0; block < config.block_count; ++block) {
        if (simulator.read_global(out_base + block) != static_cast<std::int32_t>(config.threads_per_block)) {
            ok = false;
            break;
        }
    }

    std::cout << "tinyGPU block reduction demo\n";
    std::cout << "kernel: " << kernel.name << "\n";
    std::cout << "threads: " << element_count << "\n";
    std::cout << "cycles: " << stats.cycles << "\n";
    std::cout << "warp_issues: " << stats.warp_issue_count << "\n";
    std::cout << "global_loads: " << stats.global_load_count << "\n";
    std::cout << "global_stores: " << stats.global_store_count << "\n";
    std::cout << "shared_loads: " << stats.shared_load_count << "\n";
    std::cout << "shared_stores: " << stats.shared_store_count << "\n";
    std::cout << "barriers: " << stats.barrier_issue_count << "\n";
    std::cout << "divergent_branches: " << stats.divergent_branch_count << "\n";
    std::cout << "completed_warps: " << stats.completed_warps << "\n";
    std::cout << "sample block_sum[0]: " << simulator.read_global(out_base) << "\n";
    std::cout << "sample block_sum[1]: " << simulator.read_global(out_base + 1) << "\n";
    std::cout << "status: " << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

}  // namespace

int main() {
    try {
        tinygpu::Config config;
        tinygpu::Simulator simulator(config);
        const bool vector_ok = run_vector_add_demo(simulator, config);
        const bool branch_ok = run_branch_demo(simulator, config);
        std::cout << "\n";
        const bool shared_ok = run_shared_exchange_demo(simulator, config);
        std::cout << "\n";
        const bool reduction_ok = run_block_reduction_demo(simulator, config);
        return (vector_ok && branch_ok && shared_ok && reduction_ok) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "fatal: " << ex.what() << "\n";
        return 1;
    }
}
