"""Fine input supply split between NCRISC and the score-output BRISC.

CB10 is a one-page local job queue; CB11 acknowledges completed upper stores.
NCRISC remains the sole producer of CB5. BRISC services jobs while polling for
scores, never blocking on a score whose upper input it must produce itself.
The original scalar upper permutation and BF16 decoder are unchanged.
"""

import os


def require_supported_profile():
    """The old NCRISC-only fetch analyzer cannot describe two producers."""
    if (
        os.environ.get("SKV4_FINE_FETCH_PROFILE") == "1"
        and os.environ.get("SKV4_FINE_PACKED_PROFILE") == "1"
    ):
        raise ValueError(
            "The legacy Fine fetch profile assumes NCRISC supplies both planes. "
            "Use SKV4_FINE_PACKED_PROFILE without SKV4_FINE_FETCH_PROFILE for "
            "dual-RISC endpoints; FN_SERVICE_WAIT includes BRISC upper work."
        )
