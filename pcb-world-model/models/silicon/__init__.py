"""Revision 8 Silicon-to-Board Co-Design package: Verilog RTL, FPGA .pcf co-design, and OpenLane ASIC flow."""

from models.silicon.asic_openlane import (
    AsicCompileResult,
    compile_asic_openlane,
)
from models.silicon.fpga_codesign import (
    FPGA_TARGETS,
    FpgaCodesignResult,
    FpgaTargetSpec,
    count_ratnest_crossings,
    optimize_fpga_pcb_codesign,
    run_fpga_toolchain,
)
from models.silicon.verilog_composer import (
    SiliconFeasibilityAssessment,
    SimulationResult,
    VerilogSynthesisBundle,
    assess_silicon_feasibility,
    compose_verilog_module,
    verify_verilog_simulation,
)

__all__ = [
    "AsicCompileResult",
    "FPGA_TARGETS",
    "FpgaCodesignResult",
    "FpgaTargetSpec",
    "SiliconFeasibilityAssessment",
    "SimulationResult",
    "VerilogSynthesisBundle",
    "assess_silicon_feasibility",
    "compile_asic_openlane",
    "compose_verilog_module",
    "count_ratnest_crossings",
    "optimize_fpga_pcb_codesign",
    "run_fpga_toolchain",
    "verify_verilog_simulation",
]
