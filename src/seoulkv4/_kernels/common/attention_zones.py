"""Query-independent, disjoint sink/retrieval/local attention ranges."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionZones:
    context: int
    sink: int
    local: int

    def __post_init__(self):
        if self.context <= 0 or min(self.sink, self.local) < 0:
            raise ValueError("require positive context and nonnegative sink/local")
        if self.sink + self.local > self.context:
            raise ValueError("sink and local must be disjoint and inside context")

    @classmethod
    def from_counts(cls, context, sink, local):
        if context <= 0 or min(sink, local) < 0:
            raise ValueError("require positive context and nonnegative sink/local")
        sink = min(sink, context)
        return cls(context, sink, min(local, context - sink))

    @classmethod
    def from_fixture(cls, cells, *, context, sink=None, local=None):
        if not cells:
            raise ValueError("attention requires at least one fixture cell")
        zones = []
        for cell in cells:
            if cell["context"] != context:
                raise ValueError("fixture attention context mismatch")
            zone = cls(context, int(cell["r0"]), context - int(cell["r1"]))
            requested = cls.from_counts(
                context,
                zone.sink if sink is None else sink,
                zone.local if local is None else local,
            )
            if requested != zone:
                raise ValueError(
                    f"sink/local setting {requested.sink}/{requested.local} does not "
                    f"match fixture {zone.sink}/{zone.local}; rebuild the fixture "
                    "with the requested --sink/--local"
                )
            zones.append(zone)
        if any(zone != zones[0] for zone in zones):
            raise ValueError("all fixture cells must use the same sink/local zones")
        return zones[0]

    @property
    def local_begin(self):
        return self.context - self.local

    @property
    def dense_count(self):
        return self.sink + self.local

    def count(self, retrieval=100):
        if not 0 <= retrieval <= self.local_begin - self.sink:
            raise ValueError("retrieval count exceeds the indexed range")
        return self.dense_count + retrieval

    def metadata(self, retrieval=100):
        total = self.count(retrieval)
        return dict(
            context=self.context,
            sink=self.sink,
            local=self.local,
            retrieval_begin=self.sink,
            retrieval_end=self.local_begin,
            retrieval_tokens=retrieval,
            attention_tokens=total,
            chunk_keys=128,
            chunks=(total + 127) // 128,
        )
