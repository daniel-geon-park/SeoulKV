"""Compile-time switches shared by explicit kernel declarations.

This header is constructed before the body. It never edits a rendered kernel.
The private CODEGEN names do not alias kernels' existing compile-time PROFILE
arguments, which also represent cumulative-stop diagnostics.
"""

PROFILE_MODES = (
    "plain",
    "packed",
    "issue",
    "split",
    "fetch",
    "fetch_issue",
    "fetch_split",
    "cleanup",
)
ROLES = (None, "stage16", "stage2", "attention")
PHASES = ("full", "input", "copy", "decode", "pack", "matmul")
PROFILE_IDS = {name: i for i, name in enumerate(PROFILE_MODES)}
ROLE_IDS = {name: i for i, name in enumerate(ROLES)}
PHASE_IDS = {name: i for i, name in enumerate(PHASES)}


def profile_defines(*, profile, **layout):
    """Emit named, numeric C++ switches from validated host specifications."""
    if isinstance(profile, str):
        mode = PROFILE_IDS[profile]
        enabled = profile != "plain"
    else:
        mode = int(bool(profile))
        enabled = bool(profile)
    lines = [
        f"#define SKV4_PROFILE {int(enabled)}",
        f"#define SKV4_PROFILE_MODE {mode}",
    ]
    lines.extend(
        f"#define SKV4_PROFILE_{name.upper()} {i}"
        for i, name in enumerate(PROFILE_MODES)
    )
    # Named profiling families keep layout guards independent of clock modes.
    if isinstance(profile, str):
        groups = {
            "PACKED_CLOCKS": profile not in ("plain", "cleanup"),
            "CLEANUP_CLOCKS": profile == "cleanup",
            "FETCH_GROUP": profile in ("fetch", "fetch_issue", "fetch_split"),
            "WAIT_CLOCKS": profile in ("issue", "split", "fetch_issue", "fetch_split"),
            "SPLIT_CLOCKS": profile in ("split", "fetch_split"),
        }
        lines.extend(
            f"#define SKV4_PROFILE_{name} {int(value)}"
            for name, value in groups.items()
        )
    for name, value in layout.items():
        if name not in (
            "role",
            "phase",
            "select",
            "debug_heads",
            "shared_upper",
            "external_tags",
            "dynamic",
            "dual_risc",
            "has_id_bits",
            "retain_middle",
            "debug",
            "wait_frontier",
        ):
            raise ValueError(f"unknown layout switch: {name}")
        if name == "role":
            value = ROLE_IDS[value]
        elif name == "phase":
            value = PHASE_IDS[value]
        elif not isinstance(value, bool):
            raise ValueError(f"non-boolean layout switch: {name}")
        lines.append(f"#define SKV4_CODEGEN_{name.upper()} {int(value)}")
    lines.extend(
        f"#define SKV4_ROLE_{str(name).upper()} {i}" for i, name in enumerate(ROLES)
    )
    lines.extend(
        f"#define SKV4_PHASE_{name.upper()} {i}" for i, name in enumerate(PHASES)
    )
    return "\n".join(lines) + "\n"
