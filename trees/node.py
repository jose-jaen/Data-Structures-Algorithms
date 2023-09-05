from typing import Any, Union, Optional


class Node:
    def __init__(
            self,
            data: Union[int, str],
            parent: Optional['Node'] = None,
            left: Optional['Node'] = None,
            right: Optional['Node'] = None
    ):
        self.data = data
        self.parent = parent
        self.left = left
        self.right = right

    @staticmethod
    def _type_error(attribute: Any, att_name: str) -> None:
        """Raise TypeError if attribute has unexpected type."""
        if not isinstance(attribute, Node) and attribute is not None:
            raise TypeError(
                f"'parent' must be either 'Node' or 'None' but got '{type(att_name).__name__}'"
            )

    @property
    def parent(self) -> Optional['Node']:
        return self._parent

    @parent.setter
    def parent(self, parent: Optional['Node']) -> None:
        self._type_error(attribute=parent, att_name='parent')
        self._parent = parent

    @property
    def left(self) -> Optional['Node']:
        return self._left

    @left.setter
    def left(self, left: Optional['Node']) -> None:
        self._type_error(attribute=left, att_name='left')
        self._left = left

    @property
    def right(self) -> Optional['Node']:
        return self._right

    @right.setter
    def right(self, right: Optional['Node']) -> None:
        self._type_error(attribute=right, att_name='right')
        self._right = right
