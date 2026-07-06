from __future__ import annotations

from typing import (
    Any, Tuple, List, Callable, Optional, ClassVar, Generic, TypeVar,
    get_args, get_origin,
)
import bisect

from .robot_state import BaseRobotState



StateT = TypeVar('StateT', bound=BaseRobotState)


class BaseNode(Generic[StateT]):

    # ----- class variables -----
    _state_type: ClassVar[Optional[type[BaseRobotState]]] = None

    # ----- member variables -----
    state: StateT

    _entered: bool

    _edge_list: List[Tuple[Any, int, BaseNode, Callable[[], bool]]]


    @classmethod
    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        resolved = cls._resolve_state_type()
        if resolved is not None: # when _state_type is inherit from parent
            cls._state_type = resolved


    @classmethod
    def _resolve_state_type(cls) -> Optional[type[BaseRobotState]]:
        for base in cls.__dict__.get('__orig_bases__', ()):
            origin = get_origin(base)
            if isinstance(origin, type) and issubclass(origin, BaseNode):
                args = get_args(base)
                if args and isinstance(args[0], type) and issubclass(args[0], BaseRobotState):
                    return args[0]
        return None


    def __init__(self, state: StateT):
        if self._state_type is not None and not isinstance(state, self._state_type):
            raise TypeError(
                f'robot state of {type(self).__name__} expects {self._state_type.__name__} '
                f'but {type(state).__name__} was given.'
            )

        self.state = state
        self._entered = False
        self._edge_list = []


    # ----- edge management (instance level) -----
    def add_edge(self, id: Any, ord: int, next: BaseNode[StateT], fn: Callable[[], bool]):
        if next.state is not self.state:
            raise ValueError(
                f'{type(self).__name__} and {type(next).__name__} have'
                f'different robot state instances.\n'
                f'Node in same graph must share same robot state instance.'
            )

        if any(e[0] == id for e in self._edge_list):
            raise KeyError(f'Edge id {id} is already registered.')
        bisect.insort(self._edge_list, (id, ord, next, fn), key=lambda x: x[1])


    def remove_edge(self, id: Any):
        for i, e in enumerate(self._edge_list):
            if e[0] == id:
                self._edge_list.pop(i)
                return
        raise KeyError(f'Edge id {id} is not registered.')


    # ----- lifecycle (template method) -----
    def update(self) -> BaseNode[StateT]:
        if not self._entered:
            self.on_enter()
            self._entered = True

        self.on_update()

        next_node: BaseNode[StateT] = self
        for (id, ord, next, fn) in self._edge_list:
            if fn():
                next_node = next
                break

        if next_node is not self:
            self.on_exit()
            self._entered = False

        return next_node


    # ----- subclass override -----
    def on_enter(self): ...
    def on_update(self): ...
    def on_exit(self): ...
