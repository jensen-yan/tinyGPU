#include "tinygpu/simulator.h"

#include <exception>
#include <iostream>

int main() {
    try {
        tinygpu::Config config;
        tinygpu::Simulator simulator(config);
        const tinygpu::Kernel kernel = tinygpu::make_bootstrap_kernel();
        const tinygpu::Stats stats = simulator.run(kernel);

        std::cout << "tinyGPU bootstrap run\n";
        std::cout << "kernel: " << kernel.name << "\n";
        std::cout << "cycles: " << stats.cycles << "\n";
        std::cout << "warp_issues: " << stats.warp_issue_count << "\n";
        std::cout << "completed_warps: " << stats.completed_warps << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "fatal: " << ex.what() << "\n";
        return 1;
    }
}
