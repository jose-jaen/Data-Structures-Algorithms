from typing import Optional
from collections import deque

from graphviz import Digraph

from node import Node


class BinaryTree:
    def __init__(self, root: Node):
        self.root: Node = root

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
        self.root = Node(data=int(seq_list[0]))
        size = 0

        # Store values
        q = deque()
        q.append(self.root, )
        size += 1
        i = 1
        while size > 0 and i < len(seq_list):
            current_node = q[0]
            q.popleft()
            size -= 1
            current_value = seq_list[i]

            # Add new value if not null in left node
            if current_value != 'N':
                current_node.left = Node(data=int(current_value), parent=current_node)
                q.append(current_node.left)
                size += 1

            i += 1
            if i >= len(seq_list):
                break

            # Check right node
            current_value = seq_list[i]
            if current_value != 'N':
                current_node.right = Node(data=int(current_value), parent=current_node)
                q.append(current_node.right)
                size += 1

            i += 1
        return self.root

    def height(self, node: Node) -> int:
        """Get the zero-based height of a tree.

        Args:
            node (Node): Root node

        Returns:
            Zero-based height (int)
        """
        if node is None:
            return -1
        return 1 + max(self.height(node.left), self.height(node.right))

    def size(self, node: Node) -> int:
        """Get the number of nodes of a tree.

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
        q = deque()
        q.append(self.root)

        while q:
            current_node = q[0]
            print(current_node.data)
            q.popleft()
            if current_node.left:
                q.append(current_node.left)
            if current_node.right:
                q.append(current_node.right)

    def to_dot(self):
        """Graphical representation of a Binary Tree."""
        dot = Digraph()
        q = deque()
        q.append(self.root)

        # Initiate level-order traversal
        while q:
            current_node = q[0]
            q.popleft()

            dot.node(str(current_node.data), str(current_node.data))

            if current_node.left:
                q.append(current_node.left)
                dot.edge(str(current_node.data), str(current_node.left.data))

            if current_node.right:
                q.append(current_node.right)
                dot.edge(str(current_node.data), str(current_node.right.data))

        dot.render('Binary Tree', view=True, format='png')
        return dot
