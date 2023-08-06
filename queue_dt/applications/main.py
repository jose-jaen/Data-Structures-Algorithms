from queue_dt.queue_dt import Queue


def josephus(total: int, skip: int) -> Queue:
    """Solve Josephus problem using a Queue."""
    for param in [total, skip]:
        if not isinstance(param, int):
            raise TypeError(f"Supported type is 'int' but got '{type(param)}'")

    # Instantiate Queue
    q = Queue()
    for i in range(1, total + 1):
        q.enqueue(i)

    # Iterate over queue until only one survives
    cont = 0
    while q.size() != 1:
        cont += 1

        # Check if skipped
        if cont == skip:
            q.dequeue()
            cont = 1

        first = q.front()
        q.dequeue()
        q.enqueue(first)
    return q
