#pragma once

#include "tinygpu/simulator.h"

#include <string>

namespace tinygpu {

std::string format_instruction(std::size_t pc, const Instruction& inst);
std::string disassemble_kernel(const Kernel& kernel);
std::string format_stall_reason(StallReason reason);
std::string render_timeline_text(const Kernel& kernel, const std::vector<CycleTrace>& timeline);
std::string render_timeline_html(const Kernel& kernel, const std::vector<CycleTrace>& timeline);

}  // namespace tinygpu
