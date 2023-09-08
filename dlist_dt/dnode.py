from typing import Union, Optional

from tree_dt.node import Node


class DNode:
    """Implementation of Doubly Linked List Node."""
    def __init__(
            self,
            element: Union[int, str, Node],
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
                f"Expected 'DNode' or 'None' but got '{type(previous).__name__}'"
            )
        self._previous = previous

    @property
    def element(self) -> Union[int, str, Node]:
        return self._element

    @element.setter
    def element(self, element: Union[int, str, Node]):
        if not isinstance(element, (int, str)) and type(element).__name__ != 'Node':
            raise TypeError(
                f"Expected 'int', 'str', 'Node' but got '{type(element).__name__}'"
            )
        self._element = element

    @property
    def next_node(self) -> 'DNode':
        return self._next_node

    @next_node.setter
    def next_node(self, next_node: Optional['DNode']):
        if not isinstance(next_node, DNode) and next_node is not None:
            raise TypeError(
                f"Expected 'DNode' or 'None' but got '{type(next_node).__name__}'"
            )
        self._next_node = next_node

    def __str__(self) -> str:
        """Graphical representation of DNodes with pointers."""
        left = self._previous.element if self._previous else None
        right = self._next_node.element if self._next_node else None
        sign_left = '<-->' if self._previous else '<--'
        sign_right = '<-->' if self._next_node else '-->'
        if isinstance(self._element, Node) or type(self._element).__name__ == 'Node':
            node_element = self._element.data
        else:
            node_element = self._element
        return f'{left} {sign_left} {node_element} {sign_right} {right}'
