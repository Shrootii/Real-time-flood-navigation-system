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
    
    def __lt__(self, other):
        # Define the custom comparison for the heap based on the 'first' attribute
        return self.g + self.h < other.h +other.g 

class RRT():
    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=1.0, goal_sample_rate=1, max_iter=500):
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
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
        # path = [[self.start.x,self.start.y]]
        # sequence_length = 0
        while open_set:
            current_node = heapq.heappop(open_set)
            # current_node = currTemp.y
            if self.distance(current_node ,self.end) < 0.5:
                print("Goal!")
                path = []
                while current_node:
                    path.append([current_node.x, current_node.y])
                    current_node = current_node.parent
                path.reverse()
                return path

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
                new_node.g = current_node.g + self.octile_distance(current_node, new_node)
                new_node.h = math.sqrt((new_node.x - self.end.x) ** 2 + (new_node.y - self.end.y) ** 2)

                
                
                sequence.append([self.start.x, self.start.y])
                # last_node = current_node
                # while last_node.parent :
                #     node = last_node
                #     sequence.append([node.x, node.y])
                #     last_node= node.parent
                sequence_length = len(sequence)
                # print(sequence_length)
                # print("\n\n")
                if not self.__collision_check(new_node, self.obstacle_list, sequence_length):
                    continue
               
                if self.distance(new_node ,self.end) < 0.5:
                    print("Goal!")
                    path = []
                    while new_node:
                        path.append([new_node.x, new_node.y])
                        new_node = new_node.parent
                    path.reverse()
                    return path
                # path.append(new_node)
                # Add the new node to open_set
                if new_node not in closed_set:
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
    
    def get_node_index(self,node_list, node_val):

        for i, node in enumerate(node_list) :
            if node_val == node:
                return i
        return None      
      
    def distance(self, node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
    
    def octile_distance(self,node1, node2):
        # Extract coordinates of the two nodes
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y

        # Calculate the absolute differences in coordinates
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

        # Calculate the diagonal and straight distance components
        min_d = min(dx, dy)
        max_d = max(dx, dy)

        # Calculate the octile distance using the diagonal and straight components
        return max_d + (1.414 - 2) * min_d
    
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


def main(gx=-5, gy=-10, sx=5, sy=-2):
    obstacleList = [
    (5, 5, 0.25),
    (3, 6, 0.5),
    (3, 8, 0.5),
    (3, 10, 0.5),
    (7, 5, 0.5),
    (9, 5, 0.5),
    (-10, 10, 0.5),
    (-7, 5, 0.5),
    (-9, -5, 0.5),
    (-10, -4, 0.5),
    (-6, 7, 0.5),
    (-11, 9, 0.5),
    (10, -4, 0.5),
    (6, -10, 0.5),
    (11, -9, 0.5),
    (0, 0, 0.5),
    (-7, 0, 0.5)]

    # global run_time, expanded_nodes, path_length
    start_time = time.time()
    rrt = RRT(start=[sx,sy], goal=[gx, gy], rand_area=[-15, 15], obstacle_list=obstacleList)
    path = rrt.Astar(animation=show_animation)
    end_time = time.time()

    # run_time[3] = end_time - start_time
    # expanded_nodes[3] = len(rrt.node_list)
    # path_length[3] = len(path)


    
    
    	# Draw final path
    if show_animation:  # pragma: no cover
      # rrt.draw_graph()
        # length_path = len(path)
        # path = path[::-1]
        # print(path.shape)

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
            plt.plot([x for (x, y) in path[0:i+1]], [y for (x, y) in path[0:i+1]], '-r')

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

