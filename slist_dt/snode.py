from typing import Union, Optional

from tree_dt.node import Node


class SNode:
    """Implementation of Singly Linked List Node."""
    def __init__(
            self,
            el: Union[int, str, Node],
            next_el: Optional['SNode'] = None
    ):
        self.element = el
        self.next_node = next_el

    @property
    def element(self) -> Union[int, str]:
        return self._element

    @element.setter
    def element(self, el: Union[int, str]):
        if not isinstance(el, (int, str)) and not type(el).__name__ == 'Node':
            raise TypeError(
                f"Expected 'int', 'str' or 'Node' but got '{type(el).__name__}'"
            )

        self._element = el

    @property
    def next_node(self) -> Optional['SNode']:
        return self._next_node

    @next_node.setter
    def next_node(self, next_el: Optional['SNode']):
        if not isinstance(next_el, SNode) and next_el is not None:
            raise TypeError(
                f"Supported data types are 'SNode' and 'None' but got '{type(next_el)}'"
            )

        self._next_node = next_el

    def __str__(self):
        if isinstance(self._element, Node):
            snode_element = self._element.data
        else:
            snode_element = self._element
        return f'{snode_element} --> {self._next_node}'
