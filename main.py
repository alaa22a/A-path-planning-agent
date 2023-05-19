import heapq
import matplotlib.pyplot as plt
import tkinter as tk


class Agent:
    def __init__(self, grid):
        # the graph
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        # dictionary for tracing the path after finding it , it keeps track by the parent of each node
        self.parent = {}
        # dictionary for tracing the cost taken to reach a node from starting node
        self.cost = {}

    # the heuristic function
    def heuristic(self, node, goal):
        # manhattan distance
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    # finding the neighbors of a node
    def get_neighbors(self, node):
        neighbors = []
        # look for the 4 direction up down right left
        if node[0] > 0:
            neighbors.append((node[0] - 1, node[1]))
        if node[0] < self.rows - 1:
            neighbors.append((node[0] + 1, node[1]))
        if node[1] > 0:
            neighbors.append((node[0], node[1] - 1))
        if node[1] < self.cols - 1:
            neighbors.append((node[0], node[1] + 1))
        return neighbors

    # returns the direction to go from node1 to node2
    def get_direction(self, x1, y1, x2, y2):
        if x1 - x2 < 0:
            return 'D'
        if x1 - x2 > 0:
            return 'U'
        if y1 - y2 < 0:
            return 'R'
        if y1 - y2 > 0:
            return 'L'

    # returns a string which contain the direction to the goal with the optimal cost in shape of (U, D, R, L)
    def get_path(self, goal):
        if goal not in self.parent:
            return "there's no path to the goal!"
        # we backtrack the path from the goal to the start and store the path while backing

        # cur is the current node in the path
        cur = goal
        path = ""
        # prv is pointer to the next node in the path which is the previous node in the backtracking
        prv = (-1, -1)

        # loop until the cur have no parent
        while self.parent[cur] != cur:
            if prv != (-1, -1):
                # get the char that should be added in the string to move from cur to prv
                path += self.get_direction(cur[0], cur[1], prv[0], prv[1])
            prv = cur
            cur = self.parent[cur]
        path += self.get_direction(cur[0], cur[1], prv[0], prv[1])

        # reverse the path
        return path[::-1]

    # finds the best route to the goal using the A_star algorithm
    def find(self, start, goal):
        self.parent.clear()
        self.cost.clear()

        q = [(0, start)]
        heapq.heapify(q)
        self.parent[start] = start
        self.cost[start] = 0

        # loop until the queue is empty and there's no more nodes to explore
        while q:
            # get the node with least f where f = weight + heuristic
            currentNode = heapq.heappop(q)[1]

            # if we find the goal s
            if currentNode == goal:
                break

            for child in self.get_neighbors(currentNode):
                # calculating the cost for the childs of the current node
                child_new_cost = self.cost[currentNode] + self.grid[child[0]][child[1]]

                # explore the node if it's a new node or it's a visited node but with shorter path to it
                if (self.grid[child[0]][child[1]] >= 0) and (
                        child not in self.cost or child_new_cost < self.cost[child]):
                    self.cost[child] = child_new_cost

                    # f(node) = total cost + heuristic of this node
                    f_child = child_new_cost + self.heuristic(child, goal)
                    heapq.heappush(q, (f_child, child))
                    self.parent[child] = currentNode

        if goal not in self.cost:
            return -1
        return self.cost[goal]

    # asking if a node is free
    def is_free(self, node):
        print("is the node", node, " free? (yes/no)")
        state = input()
        if state == 'yes':
            return True
        return False

    def get_move(self, move):
        if move == 'D':
            return +1, 0
        if move == 'U':
            return -1, 0
        if move == 'L':
            return 0, -1
        if move == 'R':
            return 0, +1

    def Move(self, start, goal):
        total_cost = 0
        self.cost[start] = 0
        # first find a path from current node and starting moving through the path
        # if there is an obstacle in the path recalculate a new path and start moving again

        # current node of the agent
        currnet_node = start
        total_path = ""
        p = []
        p = currnet_node
        while currnet_node != goal:
            print("The agent current node : ", currnet_node)
            path_cost = self.find(currnet_node, goal)

            if path_cost == -1:
                return -1, None, None

            print("min cost to reach the goal from the current node: ", path_cost)
            path = self.get_path(goal)
            print("the path to reach the goal: ", path)
            print("Starting to move...")
            for move in path:
                x, y = self.get_move(move)
                next_node = (currnet_node[0] + x, currnet_node[1] + y)

                if not self.is_free(next_node):
                    total_cost += self.cost[currnet_node]
                    print("detected an obstacle!")
                    self.grid[next_node[0]][next_node[1]] = -2
                    break
                # if it's a free node move to it
                total_path += move
                currnet_node = next_node
                p += next_node

        total_cost += self.cost[goal]
        return total_cost, total_path, p


# -1 is an obstacle
Graph = [
    [1, 7, 6, 1.0],
    [20, 5, -1, 8],
    [6, 5, 8, 1],
    [7, 9, 9, 14]
]

agent = Agent(Graph)
Start = (0, 0)
Goal = (3, 3)
#Start = (2, 1)
#oal = (0, 3)
total_cost, total_path, p = agent.Move(Start, Goal)

# there is no path
if total_cost == -1:
    print("Can't reach the goal from the current node.")
    exit()

print("The agent has reached the Goal with total cost : ", total_cost)
print("The agent has reached the Goal with the path  : ", total_path)
print("The agent has reached the Goal with the path  : ", p)

new_p = []
for i in range(0, len(p), 2):
    x = p[i]
    y = p[i + 1]
    new_p.append((x, y))

print(new_p)
# Define the canvas dimensions
canvas_width = 500
canvas_height = 500

# Define the animation speed
animation_speed = 1000  # milliseconds

# Create the Tkinter window and canvas
root = tk.Tk()
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

# Draw the graph
for i in range(len(Graph)):
    for j in range(len(Graph[i])):
        x1 = j * (canvas_width / len(Graph[i]))
        y1 = i * (canvas_height / len(Graph))
        x2 = (j + 1) * (canvas_width / len(Graph[i]))
        y2 = (i + 1) * (canvas_height / len(Graph))
        canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")
        canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=str(Graph[i][j]))


# Define the function to draw the lines between the points
def draw_lines(points):
    if len(points) <= 1:
        return
    x1 = points[0][0] * (canvas_height / len(Graph)) + (canvas_height / len(Graph)) / 2
    y1 = points[0][1] * (canvas_width / len(Graph[0])) + (canvas_width / len(Graph[0])) / 2
    x2 = points[1][0] * (canvas_height / len(Graph)) + (canvas_height / len(Graph)) / 2
    y2 = points[1][1] * (canvas_width / len(Graph[0])) + (canvas_width / len(Graph[0])) / 2
    line = canvas.create_line(y2, x2, y1, x1)
    canvas.after(animation_speed)
    canvas.after(animation_speed, lambda: draw_lines(points[1:]))


# Draw the lines between the points
draw_lines(new_p)
root.mainloop()
