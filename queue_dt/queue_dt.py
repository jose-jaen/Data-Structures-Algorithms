from typing import Union, List, Optional


from tree_dt.node import Node


class Queue:
    """Implementation of Queue Data Structure."""
    def __init__(self):
        self.items: List[Optional[int, str, Node]] = []

    def isempty(self) -> bool:
        return len(self.items) == 0

    def __error__(self, operation: str) -> None:
        """Throw an error if Queue is empty."""
        if self.isempty():
            raise ValueError(f"The Queue is empty, cannot apply method '{operation}'")

    def front(self) -> Optional[Union[int, str, Node]]:
        self.__error__(operation='front')
        return self.items[0]

    def enqueue(self, value: Union[int, str, Node]) -> List[Union[int, str, Node]]:
        if not isinstance(value, (int, str)) and not (type(value).__name__ == 'Node'):
            raise TypeError(
                f"Supported types are 'int', 'str', 'Node' but got '{type(value).__name__}'"
            )
        self.items.append(value)
        return self.items

    def dequeue(self) -> Optional[Union[int, str, Node]]:
        self.__error__(operation='dequeue')
        return self.items.pop(0)

    def get_at(self, index: int) -> Optional[Union[int, str, Node]]:
        """Get element at the specified index."""
        self.__error__(operation='get_at')

        # Check validity
        if not isinstance(index, int):
            raise TypeError(f"'index' must be 'int' but got '{type(index).__name__}'")

        if index < 0 or index >= self.size():
            raise ValueError(f"'index' out of range")
        return self.items[index]

    def size(self) -> int:
        return len(self.items)

    def __str__(self) -> str:
        return str(self.items)
