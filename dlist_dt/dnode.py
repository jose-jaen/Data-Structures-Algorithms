from typing import Union, Optional


class DNode:
    """Implementation of Doubly Linked List Node."""
    def __init__(
            self,
            element: Union[int, str],
            previous: Optional['DNode'] = None,
            next_node: Optional['DNode'] = None
    ):
        self.previous = previous
        self.element = element
        self.next_node = next_node

    @property
    def previous(self) -> Optional['DNode']:
        return self._previous

    @previous.setter
    def previous(self, previous: Optional['DNode']):
        if not isinstance(previous, DNode) and previous is not None:
            raise TypeError(
                f"Supported data types are 'DNode' and 'None' but got '{type(previous)}'"
            )
        self._previous = previous

    @property
    def element(self) -> Union[int, str]:
        return self._element

    @element.setter
    def element(self, element: Union[int, str]):
        if not isinstance(element, (int, str)):
            raise TypeError(
                f"Supported data types are 'int' and 'str' but got '{type(element)}'"
            )
        self._element = element

    @property
    def next_node(self) -> 'DNode':
        return self._next_node

    @next_node.setter
    def next_node(self, next_node: Optional['DNode']):
        if not isinstance(next_node, DNode) and next_node is not None:
            raise TypeError(
                f"Supported data types are 'DNode' and 'None' but got '{type(next_node)}'"
            )
        self._next_node = next_node

    def __str__(self) -> str:
        """Graphical representation of DNodes with pointers."""
        left = self._previous.element if self._previous else None
        right = self._next_node.element if self._next_node else None
        sign_left = '<-->' if self._previous else '<--'
        sign_right = '<-->' if self._next_node else '-->'
        return f'{left} {sign_left} {self._element} {sign_right} {right}'
