#include "tinygpu/disasm.h"

#include <iomanip>
#include <sstream>

namespace tinygpu {

namespace {

std::string reg_name(std::uint32_t reg) {
    return "r" + std::to_string(reg);
}

std::string opcode_name(OpCode opcode) {
    switch (opcode) {
    case OpCode::MovImm:
        return "mov_imm";
    case OpCode::MovThreadIdx:
        return "mov_thread_idx";
    case OpCode::MovBlockIdx:
        return "mov_block_idx";
    case OpCode::MovBlockThreadIdx:
        return "mov_block_thread_idx";
    case OpCode::Add:
        return "add";
    case OpCode::Mul:
        return "mul";
    case OpCode::AndImm:
        return "and_imm";
    case OpCode::SetLtImm:
        return "set_lt_imm";
    case OpCode::XorImm:
        return "xor_imm";
    case OpCode::LoadGlobal:
        return "load_global";
    case OpCode::StoreGlobal:
        return "store_global";
    case OpCode::LoadShared:
        return "load_shared";
    case OpCode::StoreShared:
        return "store_shared";
    case OpCode::BranchIfZero:
        return "branch_if_zero";
    case OpCode::Barrier:
        return "barrier";
    case OpCode::Exit:
        return "exit";
    }
    return "unknown";
}

std::string html_escape(const std::string& text) {
    std::ostringstream os;
    for (char ch : text) {
        switch (ch) {
        case '&':
            os << "&amp;";
            break;
        case '<':
            os << "&lt;";
            break;
        case '>':
            os << "&gt;";
            break;
        case '"':
            os << "&quot;";
            break;
        default:
            os << ch;
            break;
        }
    }
    return os.str();
}

std::string html_class_for(StallReason reason, bool issued) {
    if (issued) {
        return "issued";
    }
    switch (reason) {
    case StallReason::Ready:
        return "ready";
    case StallReason::WaitingBarrier:
        return "barrier";
    case StallReason::WaitingGlobalMemory:
        return "memory";
    case StallReason::Completed:
        return "done";
    }
    return "ready";
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

std::string format_stall_reason(StallReason reason) {
    switch (reason) {
    case StallReason::Ready:
        return "ready";
    case StallReason::WaitingBarrier:
        return "wait_barrier";
    case StallReason::WaitingGlobalMemory:
        return "wait_global_mem";
    case StallReason::Completed:
        return "done";
    }
    return "unknown";
}

std::string render_timeline_text(const Kernel& kernel, const std::vector<CycleTrace>& timeline) {
    std::ostringstream os;
    os << "timeline " << kernel.name << "\n";
    for (const CycleTrace& cycle : timeline) {
        os << "cycle " << cycle.cycle << (cycle.had_issue ? " issue\n" : " idle\n");
        for (const WarpTraceState& warp : cycle.warps) {
            os << "  b" << warp.block_index
               << "/w" << warp.warp_index
               << " pc=" << std::setw(2) << warp.pc
               << " active=" << std::setw(2) << warp.active_lanes
               << " state=" << format_stall_reason(warp.stall_reason);
            if (warp.issued) {
                os << " issued_pc=" << warp.issued_pc
                   << " op=" << opcode_name(warp.issued_opcode);
            }
            os << "\n";
        }
    }
    return os.str();
}

std::string render_timeline_html(const Kernel& kernel, const std::vector<CycleTrace>& timeline) {
    std::ostringstream os;
    os << "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n";
    os << "<title>tinyGPU timeline - " << html_escape(kernel.name) << "</title>\n";
    os << "<style>"
          "body{font-family:ui-monospace,Menlo,monospace;background:#f4efe6;color:#1f1d1a;margin:24px;}"
          "h1{margin:0 0 12px 0;} table{border-collapse:collapse;width:max-content;min-width:100%;}"
          "th,td{border:1px solid #c7b89c;padding:6px 8px;vertical-align:top;font-size:12px;}"
          "th{background:#e8dcc7;position:sticky;top:0;} .issued{background:#d9f2d9;}"
          ".ready{background:#faf6ef;} .barrier{background:#f7d9a8;} .memory{background:#f7c5bf;}"
          ".done{background:#d9dde5;color:#5a6472;} .note{font-weight:700;display:block;margin-bottom:2px;}"
          "</style></head><body>\n";
    os << "<h1>tinyGPU timeline: " << html_escape(kernel.name) << "</h1>\n";
    os << "<p>Each row is one cycle. Each warp cell shows state after that cycle's scheduler step.</p>\n";
    os << "<table><thead><tr><th>cycle</th>";
    if (!timeline.empty()) {
        for (const WarpTraceState& warp : timeline.front().warps) {
            os << "<th>b" << warp.block_index << "/w" << warp.warp_index << "</th>";
        }
    }
    os << "</tr></thead><tbody>\n";

    for (const CycleTrace& cycle : timeline) {
        os << "<tr><td>" << cycle.cycle << (cycle.had_issue ? " issue" : " idle") << "</td>";
        for (const WarpTraceState& warp : cycle.warps) {
            os << "<td class=\"" << html_class_for(warp.stall_reason, warp.issued) << "\">";
            if (warp.issued) {
                os << "<span class=\"note\">issue " << html_escape(opcode_name(warp.issued_opcode))
                   << " @pc " << warp.issued_pc << "</span>";
            } else {
                os << "<span class=\"note\">" << html_escape(format_stall_reason(warp.stall_reason))
                   << "</span>";
            }
            os << "pc=" << warp.pc << "<br>active=" << warp.active_lanes;
            os << "</td>";
        }
        os << "</tr>\n";
    }

    os << "</tbody></table>\n";
    os << "<h2>Kernel Listing</h2><pre>" << html_escape(disassemble_kernel(kernel)) << "</pre>\n";
    os << "</body></html>\n";
    return os.str();
}

}  // namespace tinygpu
