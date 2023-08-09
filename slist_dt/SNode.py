from typing import Union, Optional


class SNode:
    """Implementation of Single List Node."""
    def __init__(
            self,
            el: Union[int, str],
            next_el: Optional['SNode'] = None
    ):
        self.element = el
        self.next_element = next_el

    @property
    def element(self) -> Union[int, str]:
        return self._element

    @element.setter
    def element(self, el: Union[int, str]):
        if not isinstance(el, (int, str)):
            raise TypeError(
                f"Supported data types are 'int' and 'str' but got '{type(el)}'"
            )

        self._element = el

    @property
    def next_element(self) -> Optional['SNode']:
        return self._next_element

    @next_element.setter
    def next_element(self, next_el: Optional['SNode']):
        if not isinstance(next_el, SNode) and next_el is not None:
            raise TypeError(
                f"Supported data types are 'SNode' and 'None' but got '{type(next_el)}'"
            )

        self._next_element = next_el

    def __str__(self):
        return f'{self._element} --> {self._next_element}'
