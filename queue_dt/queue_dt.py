from typing import Union, NoReturn


class Queue:
    """Implementation of Queue Data Structure."""
    def __init__(self):
        self.items = []

    def isempty(self) -> bool:
        return len(self.items) == 0

    def __error__(self, operation: str) -> NoReturn:
        """Throw an error if Queue is empty."""
        if self.isempty():
            raise ValueError(f"The Queue is empty, cannot apply method '{operation}'")

    def front(self) -> Union[int, str]:
        if self.isempty():
            return self.__error__(operation='front')
        return self.items[0]

    def enqueue(self, value: Union[int, str]) -> list:
        if not isinstance(value, (int, str)):
            raise TypeError(f"Supported types are 'int' and 'str' but got '{type(value)}'")
        self.items.append(value)
        return self.items

    def dequeue(self) -> Union[int, str]:
        if self.isempty():
            return self.__error__(operation='dequeue')
        return self.items.pop(0)

    def size(self) -> int:
        return len(self.items)

    def __str__(self) -> str:
        return str(self.items)
