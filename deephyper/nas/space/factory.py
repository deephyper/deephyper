from abc import ABC, abstractclassmethod
import random

from typeguard import typechecked

from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.utils.vis_utils import check_pydot

from deephyper.nas.space import KSearchSpace
from deephyper.core.exceptions import DeephyperRuntimeError

__all__ = ["SpaceFactory"]


class SpaceFactory(ABC):
    def __init__(self, input_shape, output_shape, **kwargs) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
        # Add all k,v from kwargs as attributes
        self.__dict__.update({k: v for k, v in kwargs.items()})

    def create_space(self) -> KSearchSpace:
        return self.build(self.input_shape, self.output_shape)

    @abstractclassmethod
    def build(self, input_shape, output_shape) -> KSearchSpace:
        """Return a search space.
        """
        pass

    @typechecked
    def check_op_list(self, space: KSearchSpace, ops: list) -> list:
        if len(ops) == 0:
            ops = [random.random() for _ in range(space.num_nodes)]
        else:
            if not (len(ops) == space.num_nodes):
                raise DeephyperRuntimeError(
                    f"The argument list 'ops' should be of length {space.num_nodes} but is {len(ops)}!"
                )
        return ops

    @typechecked
    def plot_space(self, ops: list = [], fname: str = "space.dot") -> None:
        space = self.create_space()

        if not (ops is None):
            ops = self.check_op_list(space, ops)

            space.set_ops(ops)

        space.draw_graphviz(fname)

    @typechecked
    def plot_model(
        self, ops: list = [], fname: str = "random_model.png", show_shapes: bool = True
    ) -> None:
        space = self.create_space()
        ops = self.check_op_list(space, ops)
        space.set_ops(ops)
        model = space.create_model()
        plot_model(model, to_file=fname, show_shapes=show_shapes)

    def test(self):
        space = self.create_space()
        ops = self.check_op_list(space, [])
        space.set_ops(ops)
        model = space.create_model()
