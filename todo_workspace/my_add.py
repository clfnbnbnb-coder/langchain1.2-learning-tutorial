from typing import Union

Number = Union[int, float]


def add(a: Number, b: Number) -> Number:
    """返回两个数的和。

    Args:
        a: 第一个数
        b: 第二个数

    Returns:
        两个数的和

    Examples:
        >>> add(2, 3)
        5
        >>> add(-1, 1)
        0
        >>> add(1.5, 2.5)
        4.0
    """
    if not isinstance(a, (int, float)):
        raise TypeError(f"第一个参数必须是数字类型，收到: {type(a).__name__}")
    if not isinstance(b, (int, float)):
        raise TypeError(f"第二个参数必须是数字类型，收到: {type(b).__name__}")
    return a + b
