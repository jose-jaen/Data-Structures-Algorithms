from typing import NoReturn, Optional

from dlist_dt.DNode import *


class DList:
    """Implementation of Doubly Linked List Data Structure."""
    def __init__(self):
        self.size = 0
        self.head = None
        self.tail = None

    def isempty(self) -> bool:
        return self.size == 0

    def __error__(self, op: str) -> NoReturn:
        """Throw an error if the Doubly Linked List is empty.

        Args:
            - op (str): Operation to carry out on the Singly Linked List
        """
        if self.isempty():
            raise ValueError(f"The Doubly Linked List is empty, cannot apply '{op}'")

    def add_first(self, el: Union[int, str]) -> NoReturn:
        """Set a new head at any time."""
        if self.isempty():
            new_node = DNode(element=el)
            self.head = new_node
            self.tail = new_node
        else:
            new_node = DNode(element=el, next_node=self.head)
            self.head.previous = new_node
            self.head = new_node
        self.size += 1

    def add_last(self, el: Union[int, str]) -> Union[int, str]:
        """Set a new tail at any time."""
        if self.isempty():
            self.add_first(el)
            return self.head.element
        else:
            new_node = DNode(element=el, previous=self.tail)
            self.tail.next_node = new_node
            self.tail = new_node
            self.size += 1
            return self.tail.element

    def remove_first(self) -> Optional[Union[int, str]]:
        if self.isempty():
            return self.__error__(op='remove_first')

        removed = self.head.element
        if self.size == 1:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.next_node
            self.head.previous = None
        self.size -= 1
        return removed

    def remove_last(self) -> Optional[Union[int, str]]:
        if self.isempty():
            return self.__error__(op='remove_last')

        removed = self.tail.element
        if self.size == 1:
            self.head = None
            self.tail = None
        else:
            self.tail = self.tail.previous
            self.tail.next_element = None
        self.size -= 1
        return removed

    def get_at(self, index: int) -> Optional[Union[int, str]]:
        if not isinstance(index, int):
            raise TypeError(
                f"Supported data type for 'index' is 'int' but got '{type(index)}'"
            )

        if index < 0 or index >= self.size:
            raise ValueError('Index out of range')

        if index <= self.size // 2:
            count = 0
            node = self.head
            while count != index:
                node = node.next_node
                count += 1
        else:
            count = self.size - 1
            node = self.tail
            while count != index:
                node = node.previous
                count -= 1
        return node.element

    def __str__(self):
        return str(self.head)
