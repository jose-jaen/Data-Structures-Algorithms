from typing import NoReturn

from SNode import *


class SList:
    """Implementation of Single List Data Structure."""
    def __init__(self):
        self.size = 0
        self.head = None

    def isempty(self) -> bool:
        return self.size == 0

    def __error__(self, op: str) -> NoReturn:
        """Throw an error if the Singly Linked List is empty.

        Args:
            - op (str): Operation to carry out on the Singly Linked List
        """
        if self.isempty():
            raise ValueError(f"The Singly Linked List is empty, cannot apply '{op}'")

    def add_first(self, element: Union[int, str]) -> NoReturn:
        """Set a node as head at any moment."""
        node = SNode(el=element)
        node.next = self.head
        self.head = node
        self.size += 1

    def remove_first(self) -> Optional[Union[int, str]]:
        if self.isempty():
            return self.__error__(op='remove_first')

        removed = self.head.element
        self.head = self.head.next_element
        self.size -= 1
        return removed

    def add_last(self, element: Union[int, str]) -> NoReturn:
        if self.isempty():
            self.add_first(element)
        else:
            new_node = SNode(element)
            node = self.head
            while node.next_element:
                node = node.next_element
            node.next_element = new_node
            self.size += 1

    def remove_last(self) -> Optional[Union[int, str]]:
        if self.isempty():
            return self.__error__(op='remove_last')

        # Iterate over the Single Linked List
        node = self.head
        previous = None
        while node.next_element:
            previous = node
            node = node.next_element

        # One-element list
        if previous is None:
            return self.remove_first()

        # Rest of cases
        result = node.element
        previous.next_element = None
        self.size -= 1
        return result

    def contains(self, element: Union[int, str]) -> int:
        """Retrieve the index of the first matching element."""
        index = 0
        node = self.head
        while index < self.size and node is not None:
            if node.element == element:
                return index
            node = node.next_element
            index += 1

        # Not found
        return -1

    def insert_at(self, index: int, element: Union[int, str]) -> Optional[SNode]:
        if not isinstance(index, int):
            raise TypeError(f"Supported type is 'int' but got '{type(index)}'")

        elif index < 0:
            raise ValueError(f"Index cannot be smaller than '0'")

        elif index == 0:
            self.add_first(element)

        elif index >= self.size:
            self.add_last(element)

        else:
            previous = None
            node = self.head
            count = 0
            while count < index:
                previous = node
                node = node.next_element
                count += 1
            new_node = SNode(element, SNode(node.element, node.next_element))
            previous.next_element = new_node
            self.size += 1
            return new_node

    def remove_at(self, index: int) -> Optional[Union[int, str]]:
        if not isinstance(index, int):
            raise TypeError(f"Supported type is 'int' but got '{type(index)}'")

        elif index < 0 or index > self.size:
            raise ValueError(f"Index out of range")

        count = 0
        node = self.head
        previous = None
        while count < index:
            previous = node
            node = node.next_element
            count += 1
        previous.next_element = node.next_element
        removed = node.element
        self.size -= 1
        return removed

    def get_at(self, index: int) -> Optional[Union[int, str]]:
        if not isinstance(index, int):
            raise TypeError(f"Supported type is 'int' but got '{type(index)}'")

        elif index < 0 or index >= self.size:
            raise ValueError(f"Index out of range")

        # Iterate over SList
        count = 0
        node = self.head
        while count < index:
            node = node.next_element
            count += 1
        return node.element

    def __str__(self) -> str:
        return str(self.head)
