from torch import Tensor
from .random_apply import RandomApply


class AugmentationBase:
    def __init__(self, p: float) -> None:
        self._random_apply = RandomApply(self, p)

    def __call__(self, data: Tensor) -> tuple[Tensor, list[str]]:
        return self._random_apply(data)

    def forward(self, data: Tensor) -> tuple[Tensor, list[str]]:
        raise NotImplementedError()
