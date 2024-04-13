import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy
from rrt_utils import draw_cov, bhattacharyya_distance
import heapq
import time

image_counter = 1

cov_obs = np.array([[0.001, -0.000], [-0.000, 0.015]])
cov_rob = np.array([[0.0001, -0.000], [-0.000, 0.0015]])

move_obs = [0.2, 0]
show_animation = True

class Node():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.g = 0  # Cost to reach this node from the start
        self.h = 0  # Estimated cost to reach the goal from this node
        self.f = 0  # Total cost of this node: f = g + h + a1_cost + a2_cost + ... + an_cost
    
    def __lt__(self, other):
        # Define the custom comparison for the heap based on the 'first' attribute
        return self.f < other.f

class RRT():
    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=1.0, goal_sample_rate=50, max_iter=500, water_levels=None):
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.water_levels = water_levels
        self.minrand = rand_area[0]
        self.maxrand = rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = [self.start]  # Initialize node_list as an empty list

    def Astar(self, animation=True):
        open_set = []
        heapq.heappush(open_set, self.start)  # Add the start node to open_set
        closed_set = set()  # Set to store explored nodes
        
        while open_set:
            current_node = heapq.heappop(open_set)
            
            closed_set.add(current_node)

            # Generate neighboring nodes
            for _ in range(10):  # Sample random nodes multiple times for exploration
                if random.randint(0, 100) > self.goal_sample_rate:
                    rnd = [random.uniform(self.minrand, self.maxrand), random.uniform(self.minrand, self.maxrand)]
                else:
                    rnd = [self.end.x, self.end.y]
                
                sequence = []  # Sequence of positions for collision checking
                last_index = self.get_nearest_list_index(self.node_list, rnd)  # Get nearest node index
                while last_index is not None:
                    node = self.node_list[last_index]
                    sequence.append([node.x, node.y])
                    last_index = self.get_node_index(self.node_list,node.parent)

                theta_to_new_node = math.atan2(rnd[1] - current_node.y, rnd[0] - current_node.x)
                new_x = current_node.x + self.expand_dis * math.cos(theta_to_new_node)
                new_y = current_node.y + self.expand_dis * math.sin(theta_to_new_node)

                new_node = Node(new_x, new_y)
                new_node.parent = current_node
                new_node.g = current_node.g + self.expand_dis
                new_node.h = math.sqrt((new_node.x - self.end.x) ** 2 + (new_node.y - self.end.y) ** 2)
                # additional cost 1
                a1_cost = 0
                
                water_x = math.ceil(new_node.x)
                
                water_y = math.ceil(new_node.y)
                
                if self.water_levels[water_x][water_y] <= 3:
                    a1_cost = 100000/self.water_levels[water_x][water_y]
                else:
                    a1_cost = -10*self.water_levels[water_x][water_y]
                # additional cost 2
                a2_cost = 0
                if self.is_surrounded_by_obstacles(node.x, node.y):
                    a2_cost = new_node.h * 2

                
                new_node.f = new_node.g + new_node.h + a1_cost + a2_cost
                # print(water_x,water_y,self.water_levels[water_x][water_y],new_x,new_y,new_node.f, a1_cost, a2_cost)
                sequence.append([self.start.x, self.start.y])
                
                sequence_length = len(sequence)
                
                if not self.__collision_check(new_node, self.obstacle_list, sequence_length):
                    continue
                
                if self.distance(new_node,self.end) <= 0.25:
                    print("Goal!")
                    path = []
                    while current_node:
                        path.append([current_node.x, current_node.y])
                        current_node = current_node.parent
                    path.reverse()
                    return path
                if new_node not in closed_set and self.water_levels[water_x][water_y] > 3:
                    heapq.heappush(open_set, new_node)
                    self.node_list.append(new_node)
                elif self.water_levels[water_x][water_y] <= 3 and random.randint(0, 100) > 80:
                    heapq.heappush(open_set, new_node)
                    self.node_list.append(new_node)

            if animation:
                # Update the visualization (you can modify this part as needed)
                self.draw_graph(sequence_length,rnd)
        return None  # Failed to find the path
    
    def get_nearest_list_index(self, node_list, rnd):
        distance_list = [math.sqrt((node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2) for node in node_list]
        minind = distance_list.index(min(distance_list))
        return minind
    
    
    def distance(self, node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)    
    

    def get_node_index(self,node_list, node_val):

        for i, node in enumerate(node_list) :
            if node_val == node:
                return i
        return None      
      
    def is_surrounded_by_obstacles(self, x, y):
        # Check if the cell at (x, y) is surrounded by obstacles

        # Define the eight possible neighboring cell offsets (4 cardinal directions and 4 diagonals)
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        # Iterate through the neighboring cells and check if they are obstacles
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in self.obstacle_list:
                # At least one neighboring cell is not an obstacle
                return False

        # All neighboring cells are obstacles
        return True


    def __collision_check(self, node, obstacle_list, sequence_length):
        for (ox, oy, size) in obstacle_list:
            ox += sequence_length * move_obs[0]
            oy += sequence_length * move_obs[1]
            dx = ox - node.x
            dy = oy - node.y
            d = math.sqrt(dx ** 2 + dy ** 2)
            x_o = np.array([ox, oy])
            x_r = np.array([node.x, node.y])
            cov_robo = sequence_length * cov_rob
            bhatta_dist = bhattacharyya_distance(x_r, x_o, cov_robo, cov_obs)
            # print(d, bhatta_dist)
            if np.abs(bhatta_dist) <= 100:
                return False
        return True


    def draw_graph(self, sequence_length, rnd=None):
        global image_counter

        # plt.clf()
        # if rnd is not None:
        # 	plt.plot(rnd[0], rnd[1],'^k')
        # 	x_r = np.array([rnd[0],rnd[1]])
        # 	cov_robo = sequence_length*cov_rob
        # 	xr,yr = draw_cov(x_r, cov_robo, p=0.95)
        # 	plt.plot(xr,yr,'-r')
        # 	# time.sleep(2)

        for node in self.node_list:
            if node.parent is not None:
                plt.plot([node.x, self.node_list[self.get_node_index(self.node_list,node.parent)].x], [node.y, self.node_list[self.get_node_index(self.node_list,node.parent)].y], "-g")
        for (ox, oy, size) in self.obstacle_list:
            ox += sequence_length*move_obs[0]
            oy += sequence_length*move_obs[1]
            x_o = np.array([ox,oy])
            x,y = draw_cov(x_o, cov_obs, p=0.95)
            # plt.plot(x,y,'-b')
            plt.plot(ox, oy, "s", color='black', markersize=size*30)


        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([-15, 15, -15, 15])
        plt.grid(True)
        # plt.savefig('image%04d'%image_counter)
        image_counter += 1
        plt.pause(0.01)


def main():
    global image_counter, sx, sy, gx, gy, obstacleList

    water_levels = [[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [6,6,6,6,6,6,6,6,6,6,6,6,3,3,3,3,3,3,3,3,6,6,6,6,6,6,6,6,6,6],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,6,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,6,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,6,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                    ]
    # water_levels = [[3] * 30 for _ in range(30)]
    # for x in range(-10, 7):
    #     for y in range(-10, 13):
    #         water_levels[x][y] = 5

    # Set specific cells to values above 3
    start_time = time.time()
    rrt = RRT(start=[-10, -10], goal=[gx, gy], rand_area=[-15, 15], obstacle_list=obstacleList,water_levels=water_levels)
    path = rrt.Astar(animation=show_animation)

    end_time = time.time()
    global run_time, expanded_nodes, path_length
    run_time[4] = end_time - start_time
    expanded_nodes[4] = len(rrt.node_list)
    path_length[4] = len(path)
    	# Draw final path
    if show_animation:  # pragma: no cover
      # rrt.draw_graph()
        # length_path = len(path)
        # path = path[::-1]
        # print(path.shape)
        rrt.node_list =  list(reversed(rrt.node_list))
        i = 1
        for (x,y) in path:

            plt.clf()
            
            for node in rrt.node_list:
                if node.parent is not None:
                    plt.plot([node.x, rrt.node_list[rrt.get_node_index(rrt.node_list,node.parent)].x], [node.y, rrt.node_list[rrt.get_node_index(rrt.node_list,node.parent)].y], "-g")

            x_f = np.array([x,y])
            cov_robo = cov_rob*(i-1)
            xf,yf = draw_cov(x_f, cov_robo, p=0.95)
            plt.plot(xf,yf,'--m')
            # print(path)
            # path1 = list(reversed(path))
            # print(path1)
            plt.plot([x for (x, y) in (path[0:i+1])], [y for (x, y) in path[0:i+1]], '-r')

            for (ox, oy, size) in obstacleList:
                ox += i*move_obs[0]
                oy += i*move_obs[1]
                x_o = np.array([ox,oy])
                x,y = draw_cov(x_o, cov_obs, p=0.95)
                # plt.plot(x,y,'-b')
                plt.plot(ox, oy, "s", color='black', markersize=size*30)

            plt.grid(True)
            plt.axis([-15, 15, -15, 15])
            plt.plot(sx, sy, "xr")
            plt.plot(gx, gy, "xr")
            plt.pause(0.5)
            # plt.savefig('image%04d'%image_counter)
            # image_counter += 1
            i = i + 1

if __name__ == '__main__':
	main()

