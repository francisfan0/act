"""Functions exported to actsim.

The C bridge looks these up by name and passes integer token ids. Keeping the
names stable is the contract the ACT graph compiles against. The functional
backend is imported on first use.
"""

FUNCTIONAL = {
    "f_init": "init",
    "f_source_prompt": "source_prompt",
    "f_source_layer": "source_layer",
    "f_fork": "fork",
    "f_compute1": "compute1",
    "f_compute2": "compute2",
    "f_commit": "commit",
    "f_continue_after": "continue_after",
    "f_decode_input": "decode_input",
    "f_finish_layer": "finish_layer",
    "f_finish_generate": "finish_generate",
}


def __getattr__(name: str):
    if name in FUNCTIONAL:
        from runtime import functional

        return getattr(functional, FUNCTIONAL[name])
    raise AttributeError(name)
