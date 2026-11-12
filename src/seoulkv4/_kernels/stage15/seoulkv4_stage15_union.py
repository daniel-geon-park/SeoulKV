"""Offline oracle for the production source-owned Stage-1.5 union."""


def source_union_reference(head_positions, n15, workers, m):
    """Return source ownership and stable, balanced four-way union output."""
    if n15 % workers:
        raise ValueError("source workers must partition N15")
    n = n15 // workers
    selected = {
        int(position)
        for head in head_positions
        for position in head[:m]
        if 0 <= int(position) < n15
    }
    by_source = [
        sorted(position for position in selected if position // n == worker)
        for worker in range(workers)
    ]
    ordered = [position for row in by_source for position in row]
    q, r = divmod(len(ordered), 4)
    output = [
        ordered[head * q + min(head, r) : (head + 1) * q + min(head + 1, r)]
        for head in range(4)
    ]
    return by_source, output
