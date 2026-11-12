"""Stage-agnostic CB enum-to-C++ emission. IDs are defined by each stage ABI."""


def cpp_defines(layout):
    """Emit compile-time CB constants from the same enum used by the host.

    Numeric preprocessor constants preserve literal/template API compatibility
    and permit independent preprocessor equivalence against the old kernels.
    No completed kernel source is accepted or rewritten here.
    """
    return "".join(f"#define SKV4_CB_{item.name} {int(item)}\n" for item in layout)
