from stack_dt.stack_dt import Stack


def reverse(word: str) -> str:
    """Reverse the order of a word."""
    if not isinstance(word, str):
        raise TypeError(f"Expected 'str' bur got '{type(word)}'")

    # Create Stack
    s = Stack()
    for i in word:
        s.push(i)

    # Update stack to retrieve result
    result = ''
    while not s.isempty():
        result += s.top()
        s.pop()
    return result


def balanced_parenthesis(expression: str) -> bool:
    """Check whether a mathematical expression has balanced paranthesis."""
    if not isinstance(expression, str):
        raise TypeError(f"Expected 'str' but got '{type(expression)}'")

    # Instantiate stack
    s = Stack()
    for i in expression:
        if i == '(':
            s.push(i)

        elif i == ')':
            if s.isempty():
                return False
            s.pop()
    return s.isempty()
