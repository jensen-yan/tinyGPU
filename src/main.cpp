#include "tinygpu/simulator.h"

#include <algorithm>
#include <exception>
#include <iostream>

int main() {
    try {
        tinygpu::Config config;
        tinygpu::Simulator simulator(config);
        const std::size_t element_count = config.block_count * config.threads_per_block;
        const std::size_t a_base = 0;
        const std::size_t b_base = a_base + element_count;
        const std::size_t c_base = b_base + element_count;

        if (c_base + element_count > simulator.global_memory_size()) {
            std::cerr << "fatal: global memory is too small for vector add demo\n";
            return 1;
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
        std::cout << "completed_warps: " << stats.completed_warps << "\n";
        std::cout << "sample C[0]: " << simulator.read_global(c_base) << "\n";
        std::cout << "sample C[last]: " << simulator.read_global(c_base + element_count - 1) << "\n";
        std::cout << "status: " << (ok ? "PASS" : "FAIL") << "\n";
        return ok ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "fatal: " << ex.what() << "\n";
        return 1;
    }
}
