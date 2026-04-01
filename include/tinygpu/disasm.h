#pragma once

#include "tinygpu/simulator.h"

#include <string>

namespace tinygpu {

std::string format_instruction(std::size_t pc, const Instruction& inst);
std::string disassemble_kernel(const Kernel& kernel);

}  // namespace tinygpu
