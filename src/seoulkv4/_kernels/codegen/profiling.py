"""Build bounded clock regions directly around explicitly supplied phase bodies."""


class ClockRegions:
    def __init__(self, *, record_endpoints=False):
        self.names = []
        self.record_endpoints = record_endpoints

    def region(self, name, body):
        if not name.isidentifier() or name in self.names:
            raise ValueError("clock region names must be unique identifiers")
        self.names.append(name)
        end = (
            f"\n    skv_end_{name}=skv_tick(); skv_valid_{name}=true;\n"
            f"    skv_sum_{name}+=skv_end_{name}-tp_{name};\n"
            if self.record_endpoints
            else f"\n    skv_sum_{name}+=skv_tick()-tp_{name};\n"
        )
        return f"\n    uint32_t tp_{name}=skv_tick();\n" + body + end

    def kernel(self, *, body, includes=(), helpers=""):
        declarations = "\n".join(
            f"uint32_t skv_sum_{name};"
            + (
                f" uint32_t skv_end_{name}; bool skv_valid_{name};"
                if self.record_endpoints
                else ""
            )
            for name in self.names
        )
        reset = "".join(f"skv_sum_{name}=0;" for name in self.names)
        emit = "\n".join(
            f'    DeviceTimestampedData("SKVA_{name}",skv_sum_{name});'
            for name in self.names
        )
        if self.record_endpoints:
            reset += "".join(f"skv_valid_{name}=false;" for name in self.names)
            emit += "\n" + "\n".join(
                f'    if(skv_valid_{name}) DeviceTimestampedData("SKVE_{name}",skv_end_{name});'
                for name in self.names
            )
        clock = r"""inline __attribute__((always_inline)) uint32_t skv_tick() {
    asm volatile("" ::: "memory");
    uint32_t t=*reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    asm volatile("" ::: "memory"); return t;
}
"""
        return (
            '#include "tools/profiler/kernel_profiler.hpp"\n'
            + "\n".join(includes)
            + "\n"
            + clock
            + declarations
            + "\n"
            + helpers
            + "\nvoid kernel_main() {\n"
            + "    "
            + reset
            + "\n"
            + body
            + "\n"
            + emit
            + "\n}\n"
        )
