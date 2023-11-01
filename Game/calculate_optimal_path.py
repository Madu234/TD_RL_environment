class Node:
    def __init__(self, parent, position):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

class CalculateOptimalPath:
    def __init__(self, maze, start, end):
        self.maze = [list(row) for row in zip(*maze)]
        self.start = Node(None, start)
        self.end = Node(None, end)

    def calculate(self):
        open_list = []
        closed_list = []
        debug_var = True
        open_list.append(self.start)

        while open_list:
            current_node = min(open_list, key=lambda x: x.f)
            open_list.remove(current_node)
            closed_list.append(current_node)

            if current_node == self.end:
                path = []
                current = current_node
                while current:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]

            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                if not (0 <= node_position[0] < len(self.maze) and 0 <= node_position[1] < len(self.maze[0])):
                    continue

                if self.maze[node_position[0]][node_position[1]] in ['wall', 'tower']:
                    continue

                new_node = Node(current_node, node_position)
                children.append(new_node)

            for child in children:
                if child in closed_list:
                    continue

                child.g = current_node.g + 1
                child.h = (child.position[0] - self.end.position[0]) ** 2 + (child.position[1] - self.end.position[1]) ** 2
                child.f = child.g + child.h

                if any(open_node for open_node in open_list if child == open_node and child.g >= open_node.g):
                    continue

                open_list.append(child)
        return None