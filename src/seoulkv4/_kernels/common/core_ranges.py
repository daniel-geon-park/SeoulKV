"""Exact rectangular core sets for multicastable program configuration."""

from collections import defaultdict


def rectangles(points):
    """Coalesce equal horizontal runs on consecutive rows, without holes."""
    rows = defaultdict(set)
    for x, y in points:
        rows[y].add(x)
    active = {}
    result = []
    for y in sorted(rows):
        runs = []
        for x in sorted(rows[y]):
            if runs and x == runs[-1][1] + 1:
                runs[-1] = (runs[-1][0], x)
            else:
                runs.append((x, x))
        pending = {}
        for run in runs:
            previous = active.pop(run, None)
            if previous is not None and previous[3] == y - 1:
                pending[run] = (*previous[:3], y)
            else:
                if previous is not None:
                    result.append(previous)
                pending[run] = (run[0], y, run[1], y)
        result.extend(active.values())
        active = pending
    result.extend(active.values())
    return sorted(result)


def core_ranges(tt, points):
    return tt.CoreRangeSet(
        [
            tt.CoreRange(tt.CoreCoord(x0, y0), tt.CoreCoord(x1, y1))
            for x0, y0, x1, y1 in rectangles(points)
        ]
    )
