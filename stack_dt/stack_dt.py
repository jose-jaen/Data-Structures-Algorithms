from typing import Union, NoReturn


class Stack:
    """Implementation of Stack Data Structure."""
    def __init__(self):
        self.items = []

    def isempty(self) -> bool:
        return len(self.items) == 0

    def _error(self, operation: str) -> NoReturn:
        """Throw an error if Stack is empty."""
        if self.isempty():
            raise ValueError(f"The Stack is empty, cannot apply method '{operation}'")

    def top(self) -> Union[int, str]:
        if self.isempty():
            self._error(operation='top')
        return self.items[-1]

    def push(self, value: Union[int, str]) -> list:
        if not isinstance(value, (int, str)):
            raise TypeError(
                f"Supported data types are 'int' and 'str' but got '{type(value)}'"
            )
        self.items.append(value)
        return self.items

    def pop(self) -> Union[int, str]:
        if self.isempty():
            self._error(operation='pop')
        return self.items.pop()

    def size(self) -> int:
        return len(self.items)
    
    def __str__(self) -> str:
        return str(self.items)
