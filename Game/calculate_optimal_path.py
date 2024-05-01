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
        #print(end)
        reversed_end = (end[1],end[0])
        # print(reversed_end)
        # print(maze)
        self.maze = [list(row) for row in zip(*maze)]
        self.start = Node(None, start)
        self.end = Node(None, reversed_end)

    def calculate(self):
        open_list = []
        closed_list = []

        open_list.append(self.start)

        while len(open_list) > 0:
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            open_list.pop(current_index)
            closed_list.append(current_node)
            # print(f"Current node: {current_node.position} End node: {self.end.position} ")
            if current_node == self.end:
                path = []
                current = current_node
                while current is not None:
                    path.append((current.position[0], current.position[1]))
                    current = current.parent
                return path[::-1]

            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                if node_position[0] > (len(self.maze) - 1) or node_position[0] < 0 or node_position[1] > (
                        len(self.maze[len(self.maze) - 1]) - 1) or node_position[1] < 0:
                    continue
                
                if self.maze[node_position[0]][node_position[1]] == 'finish':
                    new_node = Node(current_node, node_position)
                    children.append(new_node)
                # Change this line to account for impassable cells
                if self.maze[node_position[0]][node_position[1]] != '':
                    continue

                new_node = Node(current_node, node_position)
                children.append(new_node)

            for child in children:
                # Add child to open list if it's not in the closed list
                if child not in closed_list:
                    child.g = current_node.g + 1
                    child.h = ((child.position[0] - self.end.position[0]) ** 2) + ((child.position[1] - self.end.position[1]) ** 2)
                    child.f = child.g + child.h
                    # print(f"child_pos: {child.position} and h:{child.h}")
                    # Add child to open list if it doesn't exist or if it has a better g (cost) value
                    if not any(open_node.position == child.position and open_node.g <= child.g for open_node in open_list):
                        open_list.append(child)

            # Print the open list
            # print(f"Open List: {[node.position for node in open_list]}")
        return None
