"""Argument layouts shared by Coarse and Fine matrix-score selectors.

Keep shared contracts here instead of importing one stage's private ABI from
another. This module contains no host builder or kernel generator dependencies.
"""

from seoulkv4._kernels.common.argument_abi import CompileABI, layout

SELECT_READER_TAIL = layout("width", "cohort", "cohorts")
SELECT_COMPUTE_RT = layout("chunks", "tail", "base", "cohort", "cohorts")
SCORE_COMPUTE_CT = CompileABI(layout("n", "phase"))
