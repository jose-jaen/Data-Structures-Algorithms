from typing import NoReturn

from slist_dt.snode import *


class SList:
    """Implementation of Singly Linked List Data Structure."""
    def __init__(self):
        self.size: int = 0
        self.head: Optional[SNode] = None

    def isempty(self) -> bool:
        return self.size == 0

    def _error(self, op: str) -> NoReturn:
        """Throw an error if the Singly Linked List is empty.

        Args:
            - op (str): Operation to carry out on the Singly Linked List
        """
        if self.isempty():
            raise ValueError(f"The Singly Linked List is empty, cannot apply '{op}'")

    def _check_index(self, index: int) -> NoReturn:
        """Throw an error if index is invalid."""
        if not isinstance(index, int):
            raise TypeError(
                f"Supported data type for 'index' is 'int' but got '{type(index)}'"
            )

        elif index < 0 or index >= self.size:
            raise ValueError('Index out of range')

    def add_first(self, element: Union[int, str, Node]) -> NoReturn:
        """Set a node as head at any moment."""
        node = SNode(el=element, next_el=self.head)
        self.head = node
        self.size += 1

    def remove_first(self) -> Optional[Union[int, str, Node]]:
        # Do not proceed if SList is empty
        self._error(op='remove_first')

        removed = self.head.element
        self.head = self.head.next_node
        self.size -= 1
        return removed

    def add_last(self, element: Union[int, str, Node]) -> NoReturn:
        if self.isempty():
            self.add_first(element)
        else:
            new_node = SNode(element)
            node = self.head
            while node.next_node:
                node = node.next_node
            node.next_node = new_node
            self.size += 1

    def remove_last(self) -> Optional[Union[int, str, Node]]:
        """Remove element at a specified position."""
        self._error(op='remove_last')

        # Iterate over the Single Linked List
        node = self.head
        previous = None
        while node.next_node:
            previous = node
            node = node.next_node

        # One-element list
        if previous is None:
            return self.remove_first()

        # Rest of cases
        result = node.element
        previous.next_node = None
        self.size -= 1
        return result

    def contains(self, element: Union[int, str, Node]) -> Optional[int]:
        """Retrieve the index of the first matching element."""
        self._error(op='contains')

        index = 0
        node = self.head
        while index < self.size and node is not None:
            if node.element == element:
                return index
            node = node.next_node
            index += 1

        # Not found
        return -1

    def insert_at(
            self,
            index: int,
            element: Union[int, str, Node]
    ) -> Optional[SNode]:
        # Check index validity
        self._check_index(index)

        if index == 0:
            self.add_first(element)

        elif index == self.size:
            self.add_last(element)

        else:
            previous = None
            node = self.head
            count = 0
            while count < index:
                previous = node
                node = node.next_node
                count += 1
            new_node = SNode(element, SNode(node.element, node.next_node))
            previous.next_node = new_node
            self.size += 1
            return new_node

    def remove_at(self, index: int) -> Optional[Union[int, str, Node]]:
        # Check index validity
        self._check_index(index)

        if index == 0:
            self.remove_first()

        elif index == self.size - 1:
            self.remove_last()

        else:
            count = 0
            node = self.head
            previous = None
            while count < index:
                previous = node
                node = node.next_node
                count += 1
            previous.next_node = node.next_node
            removed = node.element
            self.size -= 1
            return removed

    def get_at(self, index: int) -> Optional[Union[int, str, Node]]:
        # Check index validity
        self._check_index(index)

        # Iterate over SList
        count = 0
        node = self.head
        while count != index:
            node = node.next_node
            count += 1
        return node.element

    def __str__(self) -> str:
        return str(self.head)
