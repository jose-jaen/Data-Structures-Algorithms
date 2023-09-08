from typing import Union, NoReturn, Optional

from dlist import DList
from tree_dt.node import Node


class DList2(DList):
    """Doubly Linked List Data Structure with additional methods."""
    def __init__(self):
        super().__init__()

    def remove(self, el: Union[int, str, Node]) -> None:
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

    def remove_all(self, el: Union[int, str, Node]) -> None:
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

    def get_at_rev(self, index: int) -> Optional[Union[int, str, Node]]:
        """Retrieve the specified element with reversed positioning."""
        self._error(op='get_at_rev')
        return self.get_at(self.size - 1 - index)

    def get_middle(self) -> Optional[Union[int, str, Node]]:
        """Return the middle element."""
        self._error(op='get_middle')

        # Check if size is even
        if self.size % 2 == 0:
            return self.get_at(((self.size - 1) // 2) + 1)
        else:
            return self.get_at((self.size - 1) // 2)

    def count(self, el: Union[int, str, Node]) -> Optional[int]:
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

    def ispalindrome(self) -> Optional[bool]:
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
        if not isinstance(self.head.element, int):
            raise TypeError(f"Expected 'int' but got {type(self.head.element).__name__}")

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

    def remove_duplicates_sorted(self) -> NoReturn:
        """Eliminate duplicates from a sorted list."""
        self._error(op='remove_duplicates_sorted')

        if not self.issorted():
            raise ValueError('Doubly Linked List is not sorted')

        if self.size > 1:
            node = self.tail
            prev = node.previous
            while prev:
                if prev.element == node.element and node == self.tail:
                    prev.next_node = node.next_node
                    self.tail = prev
                    self.size -= 1
                elif prev.element == node.element:
                    prev.next_node = node.next_node
                    self.size -= 1
                node = node.previous
                prev = prev.previous

    def remove_duplicates(self) -> NoReturn:
        self._error(op='remove_duplicates')

        try:
            # Check if the list is sorted
            self.remove_duplicates_sorted()

        except ValueError as e:
            if str(e) == 'Doubly Linked List is not sorted':

                # Implement a new logic if unsorted
                if self.size > 1:
                    cache = DList2()
                    prev = self.head
                    node = prev.next_node
                    while node:
                        # Get distinct elements
                        if cache.isempty():
                            cache.add_first(prev.element)

                        elif not cache.contains(prev.element):
                            cache.add_last(prev.element)

                        else:
                            self.size -= 1
                            prev.previous.next_node = node
                            node.previous = prev.previous
                        prev = prev.next_node
                        node = node.next_node

                    # Check last value
                    if cache.contains(prev.element):
                        prev.previous.next_node = prev.next_node
                        self.tail = prev.previous
                        self.size -= 1

    def move_last(self) -> NoReturn:
        """Move the last element to the beginning without using any method."""
        self._error(op='move_last')

        if self.size > 1:
            last = self.tail
            new_tail = last.previous
            last.next_node = self.head
            self.head.previous = last
            self.head = last
            new_tail.next_node = None
            self.tail = new_tail

    def intersection(self, l2: 'DList2') -> Optional['DList2']:
        """Return common elements of two sorted lists."""
        if not self.issorted():
            raise ValueError('The first Doubly Linked List is not sorted')
        elif not l2.issorted():
            raise ValueError('The second Doubly Linked List is not sorted')

        # Remove duplicates
        res = DList2()
        self.remove_duplicates_sorted()
        l2.remove_duplicates()

        # Start with the list with the lowest value
        short_list = self if self.size < l2.size else l2
        long_list = self if self.size >= l2.size else l2
        length = self.size if self.size < l2.size else l2.size

        # Retrieve common elements
        index = 0
        while index < length:
            element = short_list.get_at(index)
            if long_list.contains(element):
                res.add_last(element)
            index += 1
        return res

    def segregate_odd_even(self) -> 'DList2':
        """Display even elements before odd elements."""
        self._error(op='segregate_odd_even')

        # Even and odd lists
        node = self.head
        odd_list = DList2()
        even_list = DList2()
        while node:
            if node.element % 2 == 0:
                even_list.add_last(node.element)
            else:
                odd_list.add_last(node.element)
            node = node.next_node

        if even_list.size != 0 and odd_list.size != 0:
            even_list.tail.next_node = odd_list.head
            even_list.tail = odd_list.tail
            even_list.remove_duplicates()
            return even_list
        elif odd_list.size == 0:
            even_list.remove_duplicates()
            return even_list
        else:
            odd_list.remove_duplicates()
            return odd_list
