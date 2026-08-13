from typing import Any, Callable, Generic, TypeVar, overload

_T = TypeVar("_T", bound=Callable[..., Any])

class JITFunction(Generic[_T]):
    def __getitem__(self, grid: Any) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def run(self, *args: Any, **kwargs: Any) -> Any: ...

@overload
def jit(fn: _T) -> JITFunction[_T]: ...
@overload
def jit(
    *,
    version: Any = None,
    repr: Any = None,
    launch_metadata: Any = None,
    do_not_specialize: Any = None,
    do_not_specialize_on_alignment: Any = None,
    debug: bool | None = None,
    noinline: bool | None = None,
) -> Callable[[_T], JITFunction[_T]]: ...
def jit(*args: Any, **kwargs: Any) -> Any: ...
