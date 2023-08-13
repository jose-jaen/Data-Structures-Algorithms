from typing import Union, NoReturn, Optional

from dlist import DList


class DList2(DList):
    """Singly Linked List Data Structure with additional methods."""
    def __init__(self):
        super().__init__()

    def remove(self, el: Union[int, str]) -> NoReturn:
        """Eliminate first occurrence of the specified element."""
        if not self.contains(el):
            raise ValueError(f"Element '{el}' not present in Doubly Linked List")

        # Iterate
        prev = None
        node = self.head
        while node.element != el:
            prev = node
            node = node.next_node

        # Control for first element
        if prev is None:
            self.head = node.next_node

        # Control for last element
        elif node == self.tail:
            prev.next_node = None
            self.tail = prev

        # Middle elements
        else:
            prev.next_node = node.next_node
        self.size -= 1

    def remove_all(self, el: Union[int, str]) -> NoReturn:
        """Eliminate all occurrences of a given element."""
        self._error(op='remove_all')

        # Do not proceed if not contained
        if not self.contains(el):
            raise ValueError(f"Element '{el}' not present in Doubly Linked List")

        # Control for one-element lists
        if self.size == 1:
            self.head = None
            self.size -= 1

        # Iterate from the tail
        else:
            node = self.tail
            prev = node.previous
            while prev:
                if node.element == el and node == self.tail:
                    prev.next_node = None
                    self.tail = prev
                    self.size -= 1

                elif node.element == el:
                    prev.next_node = node.next_node
                    node.next_node.previous = prev
                    self.size -= 1

                prev = prev.previous
                node = node.previous

                if prev is None and node.element == el:
                    self.head = node.next_node
                    self.size -= 1

    def get_at_rev(self, index: int) -> Optional[Union[int, str]]:
        """Retrieve the specified element with reversed positioning."""
        self._error(op='get_at_rev')
        return self.get_at(self.size - 1 - index)

    def get_middle(self) -> Optional[Union[int, str]]:
        """Return the middle element."""
        self._error(op='get_middle')

        # Check if size is even
        if self.size % 2 == 0:
            print((self.size // 2) + 1)
            return self.get_at(((self.size - 1) // 2) + 1)
        else:
            return self.get_at((self.size - 1) // 2)

    def count(self, el: Union[int, str]) -> Optional[int]:
        """Return total occurrences of an element."""
        self._error(op='count')
        counter = 0

        # Check if element is in the list
        if self.contains(el):
            node = self.head
            while node:
                if node.element == el:
                    counter += 1
                node = node.next_node
        return counter

    def ispalindrome(self) -> bool:
        """Check if there are palindrome words."""
        self._error(op='ispalindrome')

        # Iterate first half
        index = 0
        node = self.head
        while index < (self.size - 1) // 2:
            if isinstance(node.element, str) and node.element == node.element[::-1]:
                return True
            index += 1
            node = node.next_node

        # Iterate over last half
        node = self.tail
        index = self.size - 1
        while index > self.size // 2:
            if isinstance(node.element, str) and node.element == node.element[::-1]:
                return True
            index -= 1
            node = node.previous
        return False

    def issorted(self) -> Optional[bool]:
        """Check if the list follows ascending order."""
        self._error(op='issorted')

        # Check one element lists
        if isinstance(self.head.element, str):
            raise TypeError("Supported data type is 'int' but got 'str'")

        if self.size > 1:
            # Iterate from the end
            node = self.tail
            prev = node.previous
            while prev:
                if isinstance(prev.element, str) or isinstance(node.element, str):
                    raise TypeError("Supported data type is 'int' but got 'str'")
                if prev.element > node.element:
                    return False
                prev = prev.previous
                node = node.previous
        return True
