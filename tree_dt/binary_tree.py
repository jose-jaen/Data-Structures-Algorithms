import os
from typing import Optional
from collections import deque
from time import sleep

from graphviz import Digraph

from node import Node
from queue_dt.queue_dt import Queue


class BinaryTree:
    def __init__(self, root: Optional[Node]):
        self.root = root

    def construct_tree(self, seq: str) -> Optional[Node]:
        """Connect nodes to create a Binary Tree.

        Args:
            seq (str): String representation of a Binary Tree

        Returns:
            Connected Nodes (Optional[Node])
        """
        if not seq or seq[0] == 'N':
            return None

        # Convert into strings
        seq_list = list(map(str, seq.split(' ')))

        # Create root
        self.root = Node(data=seq_list[0])
        size = 0

        # Queue for storing current values
        q = Queue()
        q.enqueue(self.root)
        size += 1

        # Start traversal
        i = 1
        while size > 0 and i < len(seq_list):
            current_node = q.dequeue()
            size -= 1

            # Add new value if not null in left node
            if seq_list[i] != 'N':
                try:
                    eval(seq_list[i])
                    current_value = int(seq_list[i])
                except NameError:
                    current_value = seq_list[i]
                current_node.left = Node(data=current_value, parent=current_node)
                q.enqueue(current_node.left)
                size += 1

            i += 1
            if i >= len(seq_list):
                break

            # Check right node
            if seq_list[i] != 'N':
                try:
                    eval(seq_list[i])
                    current_value = int(seq_list[i])
                except NameError:
                    current_value = seq_list[i]
                current_node.right = Node(data=current_value, parent=current_node)
                q.enqueue(current_node.right)
                size += 1

            i += 1
        return self.root

    def height(self, node: Node) -> int:
        """Get the zero-based height of a (sub)tree.

        Args:
            node (Node): Root node

        Returns:
            Zero-based height (int)
        """
        if node is None:
            return -1
        return 1 + max(self.height(node.left), self.height(node.right))

    def size(self, node: Node) -> int:
        """Get the number of nodes of a (sub)tree.

        Args:
            node (Node): Root node

        Returns:
            Number of nodes (int)
        """
        if node is None:
            return 0
        return 1 + self.size(node.left) + self.size(node.right)

    def level_order(self) -> None:
        """Level-oder traversal of a Binary Tree."""
        q = Queue()
        q.enqueue(self.root)

        # Traverse
        while not q.isempty():
            current_node = q.get_at(0)
            print(current_node.data)
            q.dequeue()
            if current_node.left:
                q.enqueue(current_node.left)
            if current_node.right:
                q.enqueue(current_node.right)

    def visualize_tree(self) -> Digraph:
        """Graphical representation of a Binary Tree."""
        dot = Digraph()
        q = Queue()
        q.enqueue(self.root)

        # Initiate level-order traversal
        while not q.isempty():
            current_node = q.get_at(0)
            q.dequeue()

            # Include current node
            dot.node(str(current_node.data), str(current_node.data))

            # Add left node
            if current_node.left:
                q.enqueue(current_node.left)
                dot.edge(str(current_node.data), str(current_node.left.data))

            # Add right node
            if current_node.right:
                q.enqueue(current_node.right)
                dot.edge(str(current_node.data), str(current_node.right.data))

        # Display graphical representation
        dot.render('Binary Tree', view=True, format='png')

        # Remove local files
        sleep(5)
        if os.path.isfile('Binary Tree.png'):
            os.remove('Binary Tree.png')
            os.remove('Binary Tree')
        return dot
