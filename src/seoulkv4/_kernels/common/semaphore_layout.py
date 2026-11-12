"""Stage-agnostic semaphore enum-to-C++ emission; preserves stage-local IDs."""


def cpp_defines(layout):
    """Emit numeric compile-time names from the host's same stage-local enum."""
    return "".join(f"#define SKV4_SEM_{item.name} {int(item)}\n" for item in layout)
