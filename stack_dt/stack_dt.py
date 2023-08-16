from typing import Union, NoReturn, List, Optional


class Stack:
    """Implementation of Stack Data Structure."""
    def __init__(self):
        self.items: List[Optional[int, str]] = []

    def isempty(self) -> bool:
        return len(self.items) == 0

    def _error(self, operation: str) -> NoReturn:
        """Throw an error if Stack is empty."""
        if self.isempty():
            raise ValueError(f"The Stack is empty, cannot apply method '{operation}'")

    def top(self) -> Optional[Union[int, str]]:
        self._error(operation='top')
        return self.items[-1]

    def push(self, value: Union[int, str]) -> List[int, str]:
        if not isinstance(value, (int, str)):
            raise TypeError(
                f"Supported data types are 'int' and 'str' but got '{type(value)}'"
            )
        self.items.append(value)
        return self.items

    def pop(self) -> Optional[Union[int, str]]:
        self._error(operation='pop')
        return self.items.pop()

    def size(self) -> int:
        return len(self.items)

    def __str__(self) -> str:
        return str(self.items)
