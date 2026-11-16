"""Symbolic and physical DRC verifier layer."""

from verifier.fast_clearance import (
    DRCError,
    check_copper_clearances,
    check_courtyard_overlaps,
    point_to_segment_dist,
    segment_to_rect_dist,
    segment_to_segment_dist,
    verify_board,
)
from verifier.kicad_drc import find_kicad_cli, verify

__all__ = [
    "DRCError",
    "verify",
    "verify_board",
    "find_kicad_cli",
    "check_courtyard_overlaps",
    "check_copper_clearances",
    "point_to_segment_dist",
    "segment_to_segment_dist",
    "segment_to_rect_dist",
]
