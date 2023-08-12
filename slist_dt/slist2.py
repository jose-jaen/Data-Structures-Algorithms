from typing import Union, NoReturn

from slist import SList, SNode


class SList2(SList):
    """Singly Linked List Data Structure with additional methods."""
    def __init__(self):
        super().__init__()

    def remove(self, el: Union[int, str]) -> NoReturn:
        """Eliminate first occurrence of the specified element."""
        if self.contains(el) == -1:
            raise ValueError(f"Element '{el}' not found in SList")

        prev = None
        node = self.head
        while node.next_node and node.element != el:
            prev = node
            node = node.next_node

        # Remove element otherwise
        if prev:
            prev.next_node = node.next_node
        else:
            self.head = node.next_node
        self.size -= 1

    def remove_all(self, el: Union[int, str]) -> NoReturn:
        """Eliminate all occurrences of a given element."""
        if self.contains(el) == -1:
            raise ValueError(f"Element '{el}' not found in SList")

        # Handle first element
        while self.head.element == el:
            self.head = self.head.next_node
            self.size -= 1

        # Iterate through the list
        prev = self.head
        node = self.head.next_node
        while node.next_node:
            if node.element == el:
                node = node.next_node
                self.size -= 1
            else:
                prev.next_node = node
                prev = node
                node = node.next_node

        # Check last value
        if node.element == el:
            prev.next_node = None
            self.size -= 1

    def get_at_rev(self, index: int) -> Union[int, str]:
        """Retrieve the specified element with reversed positioning."""
        self._error(op='get_at_rev')
        self._check_index(index)
        return self.get_at((self.size - 1) - index)

    def get_middle(self) -> Union[int, str]:
        """Return the middle element."""
        self._error(op='get_middle')

        # Single-element list
        if self.size == 1:
            return self.head.element

        # Iterate otherwise
        count = 0
        node = self.head
        index = (self.size - 1) // 2 + 1 if self.size % 2 == 0 else self.size // 2
        while count < index:
            node = node.next_node
            count += 1
        return node.element

    def count(self, el: Union[int, str]) -> int:
        """Return total occurrences of an element."""
        if self.contains(el) == -1:
            return 0

        # Iterate over SList
        total = 0
        index = 0
        node = self.head
        while index != self.size:
            if node.element == el:
                total += 1
            index += 1
            node = node.next_node
        return total

    def ispalindrome(self) -> bool:
        """Check if there are palindrome words."""
        index = 0
        node = self.head
        while index != self.size:
            if isinstance(node.element, str):
                size = len(node.element)
                reverse = [node.element[size - 1 - j] for j in range(size)]
                if ''.join(reverse) == node.element:
                    return True
            index += 1
            node = node.next_node
        return False

    def issorted(self) -> bool:
        """Check if the list follows ascending order."""
        index = 0
        node = self.head
        while node.next_node:
            if isinstance(node.element, str) or isinstance(node.next_node.element, str):
                return False

            # Not sorted
            if node.element > node.next_node.element:
                return False

            # Sorted
            if node.element <= node.next_node.element:
                index += 1
                node = node.next_node
        return True

    def remove_duplicates_sorted(self) -> NoReturn:
        """Eliminate duplicates from a sorted list."""
        self._error(op='remove_duplicates_sorted')

        # Identify duplicates
        if self.issorted() and self.size > 1:
            current = self.head
            while current and current.next_node:
                if current.element == current.next_node.element:
                    current.next_node = current.next_node.next_node
                    self.size -= 1
                else:
                    current = current.next_node

    def remove_duplicates(self) -> NoReturn:
        # Check if list is sorted
        self.remove_duplicates_sorted()

        if self.size > 1:
            prev = self.head
            node = prev.next_node

            # List for caching values
            cache = SList2()
            cache.add_first(prev.element)
            while node.next_node:
                if cache.contains(node.element) != -1:
                    node = node.next_node
                    self.size -= 1
                else:
                    cache.add_first(node.element)
                    prev.next_node = node
                    prev = node
                    node = node.next_node

            # Check last value
            if cache.contains(node.element) != -1:
                self.size -= 1
                prev.next_node = None
            else:
                prev.next_node = node

    def move_last(self) -> NoReturn:
        """Move the last element to the beginning without using any method."""
        if self.isempty():
            self._error(op='move_last')

        # Iterate over SList
        index = 0
        prev = None
        node = self.head
        while index != self.size - 1:
            index += 1
            prev = node
            node = node.next_node
        node.next_node = self.head
        self.head = node
        prev.next_node = None

    def intersection(self, l2: 'SList2') -> 'SList2':
        """Return common elements of two sorted lists."""
        if self.issorted() and l2.issorted():
            self.remove_duplicates()
            l2.remove_duplicates()

            # Check common elements
            res = SList2()
            for i in range(self.size):
                el1 = self.get_at(i)
                if l2.contains(el1) != -1:
                    res.add_last(el1)
            return res
        else:
            raise ValueError('At least one of the given lists is not sorted')
