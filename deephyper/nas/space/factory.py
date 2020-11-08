from abc import ABC, abstractclassmethod
import random
from copy import deepcopy

from typeguard import typechecked

from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.utils.vis_utils import check_pydot

from deephyper.nas.space import KSearchSpace
from deephyper.core.exceptions import DeephyperRuntimeError

__all__ = ["SpaceFactory"]


class SpaceFactory(ABC):
    def __call__(self, input_shape, output_shape, **kwargs) -> KSearchSpace:
        return self.build(input_shape, output_shape, **kwargs)

    @abstractclassmethod
    def build(self, input_shape, output_shape, **kwargs) -> KSearchSpace:
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
    def plot_space(
        self,
        input_shape,
        output_shape,
        ops: list = [],
        fname: str = "space.dot",
        **kwargs,
    ) -> None:
        space = self(input_shape, output_shape, **kwargs)

        if not (ops is None):
            ops = self.check_op_list(space, ops)

            space.set_ops(ops)

        space.draw_graphviz(fname)

    @typechecked
    def plot_model(
        self,
        input_shape,
        output_shape,
        ops: list = [],
        fname: str = "random_model.png",
        show_shapes: bool = True,
        **kwargs,
    ) -> None:
        space = self(input_shape, output_shape, **kwargs)
        ops = self.check_op_list(space, ops)
        space.set_ops(ops)
        model = space.create_model()
        plot_model(model, to_file=fname, show_shapes=show_shapes)

    def test(self, input_shape, output_shape, **kwargs):
        space = self(input_shape, output_shape, **kwargs)
        ops = self.check_op_list(space, [])
        space.set_ops(ops)
        model = space.create_model()
