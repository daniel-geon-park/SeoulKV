"""Stage-agnostic kernel argument packing shared by host and C++ emitters.

Widths count uint32 argument words, not bytes. Tensor accessor encodings are
opaque: the host asks TT for their words and C++ chains next_compile_time_args_offset.
No completed kernel source is accepted or rewritten here.
"""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class Field:
    name: str
    words: int = 1


@dataclass(frozen=True)
class Layout:
    fields: tuple[Field, ...]

    def __post_init__(self):
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("duplicate argument field")
        for field in self.fields:
            if not field.name.isidentifier() or field.words < 0:
                raise ValueError(f"invalid argument field: {field}")

    @property
    def words(self):
        return sum(field.words for field in self.fields)

    def offset(self, name):
        offset = 0
        for field in self.fields:
            if field.name == name:
                return offset
            offset += field.words
        raise KeyError(name)

    def pack(self, **values):
        expected = {field.name for field in self.fields}
        if values.keys() != expected:
            raise ValueError(
                f"argument fields: missing={expected - values.keys()}, "
                f"unknown={values.keys() - expected}"
            )
        result = []
        for field in self.fields:
            value = values[field.name]
            if field.words == 1:
                value = (value,)
            else:
                value = tuple(value)
            if len(value) != field.words:
                raise ValueError(f"{field.name}: expected {field.words} words")
            for word in value:
                if not isinstance(word, Integral) or not 0 <= word <= 0xFFFFFFFF:
                    raise ValueError(f"{field.name}: not a uint32 word: {word!r}")
                result.append(int(word))
        return result

    def then(self, other):
        return Layout(self.fields + other.fields)

    def prefixed(self, prefix):
        return Layout(
            tuple(Field(f"{prefix}_{field.name}", field.words) for field in self.fields)
        )

    def cpp_defines(self, prefix, *, base=0):
        return (
            "".join(
                f"#define {prefix}_{field.name.upper()} {base + self.offset(field.name)}\n"
                for field in self.fields
            )
            + f"#define {prefix}_END {base + self.words}\n"
        )


def layout(*fields):
    return Layout(
        tuple(Field(field) if isinstance(field, str) else field for field in fields)
    )


@dataclass(frozen=True)
class CompileABI:
    scalars: Layout
    tensors: tuple[str, ...] = ()

    def pack(self, tt, *, tensors, **scalars):
        if tensors.keys() != set(self.tensors):
            raise ValueError("tensor accessor fields do not match compile ABI")
        result = self.scalars.pack(**scalars)
        for name in self.tensors:
            result.extend(tt.TensorAccessorArgs(tensors[name]).get_compile_time_args())
        return result

    def cpp_defines(self, prefix):
        start = self.scalars.words
        return (
            self.scalars.cpp_defines(prefix) + f"#define {prefix}_ACCESSORS {start}\n"
        )
