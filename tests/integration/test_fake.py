from torch.nn import ReLU

from lightcat.utils import setup_object


def test_fake() -> None:
    module = ReLU()
    assert setup_object(module) is module
