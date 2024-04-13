## self written rrt

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from rrt_utils import draw_cov, bhattacharyya_distance

image_counter = 1

cov_obs = np.array([[0.001,-0.000],[-0.000,0.015]])
cov_rob = np.array([[0.0001,-0.000],[-0.000,0.0015]])

move_obs = [0.2,0]

show_animation = True

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=1.0, goal_sample_rate=20, max_iter=500):
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.minrand = rand_area[0]
        self.maxrand = rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list

    def Planning(self, animation=True):
        self.node_list = [self.start]

        for _ in range(self.max_iter):
            if random.randint(0, 100) > self.goal_sample_rate:
                rnd = [random.uniform(self.minrand, self.maxrand), random.uniform(self.minrand, self.maxrand)]
            else:
                rnd = [self.end.x, self.end.y]

            nind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[nind]
            theta_to_new_node = math.atan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)
            new_node = Node(nearest_node.x + self.expand_dis * math.cos(theta_to_new_node),
                            nearest_node.y + self.expand_dis * math.sin(theta_to_new_node))
            new_node.parent = nind

            if not self.__collision_check(new_node, self.obstacle_list):
                continue

            self.node_list.append(new_node)
            dx = new_node.x - self.end.x
            dy = new_node.y - self.end.y
            d = math.sqrt(dx ** 2 + dy ** 2)

            if d <= self.expand_dis:
                print("Goal!")
                break

            if animation:
                self.draw_graph()

        path = [[self.end.x, self.end.y]]
        last_index = len(self.node_list) - 1
        while self.node_list[last_index].parent is not None:
            node = self.node_list[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])

        return path, self.node_list

    def get_nearest_list_index(self, node_list, rnd):
        distance_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in node_list]
        minind = distance_list.index(min(distance_list))
        return minind

    def __collision_check(self, node, obstacle_list):
        for (ox, oy, size) in obstacle_list:
            dx = ox - node.x
            dy = oy - node.y
            d = math.sqrt(dx ** 2 + dy ** 2)
            if d <= size:
                return False
        return True

    def draw_graph(self):
        plt.clf()
        for node in self.node_list:
            if node.parent is not None:
                plt.plot([node.x, self.node_list[node.parent].x], [node.y, self.node_list[node.parent].y], "-g")
        for (ox, oy, size) in self.obstacle_list:
            plt.plot(ox, oy, "s", color='black', markersize=size * 30)
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([-15, 15, -15, 15])
        plt.grid(True)
        plt.pause(0.01)


def main():
	#print("start " + __file__)
	

	gx=-4
	gy=2
	sx=5
	sy=-2
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
	

	# Set Initial parameters
	

	start_time = time.time()
	rrt = RRT(start=[sx, sy], goal=[gx, gy],rand_area=[-15, 15], obstacle_list=obstacleList)
	path, node_list = rrt.Planning(animation=show_animation)
	end_time = time.time()
	# global run_time, expanded_nodes, path_length
	run_time = end_time - start_time
	expanded_nodes = len(node_list)
	path_length = len(path)
	print("RRT took %0.2f seconds, expanded %d nodes, path length %d"%(run_time, expanded_nodes, path_length))
	# print("RRT took %0.2f seconds, expanded %d nodes, path length %d"%(run_time[0], expanded_nodes[0], path_length[0]))
	# Draw final path
	if show_animation:  # pragma: no cover
		# rrt.draw_graph()
		length_path = len(path)
		path = path[::-1]
		# print(path.shape)

		i = 1
		for (x,y) in path:

			plt.clf()

			for node in node_list:
				if node.parent is not None:
					plt.plot([node.x, node_list[node.parent].x], [node.y, node_list[node.parent].y], "-g")

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
			image_counter += 1
			i = i + 1


if __name__ == '__main__':
	main()
