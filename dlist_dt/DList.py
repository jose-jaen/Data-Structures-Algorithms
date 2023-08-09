from typing import NoReturn

from dlist_dt.DNode import *


class DList:
    """Implementation of Doubly Linked List Data Structure."""
    def __init__(self):
        self.size = 0
        self.head = None
        self.tail = None

    def isempty(self) -> bool:
        return self.size == 0

    def _error(self, op: str) -> NoReturn:
        """Throw an error if the Doubly Linked List is empty.

        Args:
            - op (str): Operation to carry out on the Singly Linked List
        """
        if self.isempty():
            raise ValueError(f"The Doubly Linked List is empty, cannot apply '{op}'")

    def _check_index(self, index: int) -> NoReturn:
        """Throw an error if index is invalid."""
        if not isinstance(index, int):
            raise TypeError(
                f"Supported data type for 'index' is 'int' but got '{type(index)}'"
            )

        elif index < 0 or index >= self.size:
            raise ValueError('Index out of range')

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

    def remove_first(self) -> Optional[Union[int, str]]:
        # Do not proceed if DList is empty
        self._error(op='remove_first')

        removed = self.head.element
        if self.size == 1:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.next_node
            self.head.previous = None
        self.size -= 1
        return removed

    def add_last(self, el: Union[int, str]) -> NoReturn:
        """Set a new tail at any time."""
        if self.isempty():
            self.add_first(el)
        else:
            new_node = DNode(element=el, previous=self.tail)
            self.tail.next_node = new_node
            self.tail = new_node
            self.size += 1

    def remove_last(self) -> Optional[Union[int, str]]:
        # Do not proceed if DList is empty
        self._error(op='remove_last')

        removed = self.tail.element
        if self.size == 1:
            self.head = None
            self.tail = None
        else:
            self.tail = self.tail.previous
            self.tail.next_node = None
        self.size -= 1
        return removed

    def contains(self, el: Union[int, str]) -> bool:
        # Do not proceed if DList is empty
        self._error(op='contains')

        # Iterate over first half of DList
        count = 0
        node = self.head
        while count != (self.size - 1) // 2 and node.element != el:
            node = node.next_node
            count += 1

        # Iterate from last haf otherwise
        if count == (self.size - 1) // 2 and node.element != el:
            node = self.tail
            count = self.size - 1
            while count != (self.size - 1) // 2 and node.element != el:
                node = node.previous
                count -= 1

            # Not found
            if count == (self.size - 1) // 2 and node.element != el:
                return False
        return True

    def insert_at(self, index: int, el: Union[int, str]) -> NoReturn:
        # Check index validity
        self._check_index(index)

        if index == 0:
            self.add_first(el)

        elif index == self.size - 1:
            new_node = DNode(el, previous=self.tail.previous, next_node=self.tail)
            self.tail.previous = new_node
            new_node.previous.next_node = new_node
            self.size += 1

        # Start from head if faster
        elif index <= (self.size - 1) // 2:
            count = 0
            node = self.head
            prev = None
            while count != index:
                prev = node
                node = node.next_node
                count += 1
            new_node = DNode(el, previous=prev, next_node=node)
            prev.next_node = new_node
            node.previous = new_node
            self.size += 1

        # Start from tail otherwise
        else:
            count = self.size - 1
            node = self.tail
            prev = node.previous
            while count != index:
                prev = prev.previous
                node = node.previous
                count -= 1
            new_node = DNode(el, previous=prev, next_node=node)
            prev.next_node = new_node
            node.previous = new_node
            self.size += 1

    def remove_at(self, index: int) -> Optional[Union[int, str]]:
        """Remove an element from a certain position and return it."""
        # Do not proceed if DList is empty or index is invalid
        self._error(op='remove_at')
        self._check_index(index)

        # Constant solutions
        if index == 0:
            self.remove_first()

        elif index == self.size - 1:
            self.remove_last()

        # Efficient search otherwise
        else:
            if index <= (self.size - 1) // 2:
                count = 0
                node = self.head
                prev = None
                while count != index:
                    prev = node
                    node = node.next_node
                    count += 1
                removed = node.element
                prev.next_node = node.next_node
                node.next_node.previous = prev
            else:
                count = self.size - 1
                node = self.tail
                prev = node.previous
                while count != index:
                    prev = prev.previous
                    node = node.previous
                    count -= 1
                removed = node.element
                prev.next_node = node.next_node
                node.next_node.previous = prev
            self.size -= 1
            return removed

    def get_at(self, index: int) -> Optional[Union[int, str]]:
        """Retrieve the element at the specified position."""
        self._check_index(index)

        if index <= (self.size - 1) // 2:
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
        """Graphical representation of DList."""
        res = 'None'
        node = self.head
        while node.next_node:
            if node == self.head:
                res += f' <-- {node.element}'
            else:
                res += f' <--> {node.element}'
            node = node.next_node
        res += f' <--> {self.tail.element} --> None'
        return res
