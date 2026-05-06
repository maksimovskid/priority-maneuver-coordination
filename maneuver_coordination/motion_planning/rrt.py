"""

Path planning Sample Code with RRT

author: Atsushi Sakai(@Atsushi_twi)
modified by Daniel Maksimovski

"""

import math
import random

import matplotlib.pyplot as plt

# from new_map_with_obstacles import *
import timeit

show_animation = False


class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self, start, goal, obstacle_list, rand_area_x, rand_area_y,
                 expand_dis=4, path_resolution=1, goal_sample_rate=10, max_iter=1000):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y],...]
        randArea:Random Sampling Area [min,max]

        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand_x = rand_area_x[0]
        self.max_rand_x = rand_area_x[1]
        self.min_rand_y = rand_area_y[0]
        self.max_rand_y = rand_area_y[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """
        start = timeit.default_timer()
        self.node_list = [self.start]
        for i in range(self.max_iter):
            # print("Iter:", i, ", number of nodes:", len(self.node_list))

            rnd_node = self.get_random_node()  # random node in the freespace
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)  # find closest node in tree
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)  # compute position of new node

            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(rnd_node)
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            stop = timeit.default_timer()
            print('time: ', stop - start)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(random.uniform(self.min_rand_x, self.max_rand_x),
                            random.uniform(self.min_rand_y, self.max_rand_y))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy) in self.obstacle_list:
            plt.plot(ox, oy, '.k')

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 45, -2, 45])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)
                 ** 2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_collision(node, obstacleList):

        if node is None:
            return False

        for (ox, oy) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= 1.1:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


def main():
    """Run the standalone RRT demo."""
    print("start " + __file__)

    # ====Search Path with RRT====
    # points defined as obstacles along the road boundaries
    obstacleList = [
        (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (8, 6), (9, 6), (10, 6), (11, 6), (12, 6),
        (13, 6), (14, 6),
        (15, 6), (16, 6), (17, 6), (18, 6), (19, 6), (20, 6), (21, 6), (22, 6), (23, 6), (24, 6), (25, 6), (26, 6),
        (27, 6),
        (28, 6), (29, 6), (30, 6), (31, 6), (32, 6), (33, 6), (34, 6), (35, 6), (36, 6), (37, 6), (38, 6), (39, 6),
        (40, 6),
        (41, 6), (41, 7), (41, 8), (41, 9), (41, 10), (41, 11), (41, 12), (41, 13), (41, 14), (41, 15), (41, 16),
        (41, 17),
        (41, 18), (41, 19), (41, 20), (41, 21), (41, 22), (41, 23), (41, 24), (41, 25), (41, 26), (41, 27), (41, 28),
        (41, 29),
        (41, 30), (41, 31), (41, 32), (41, 33), (41, 34), (41, 35), (41, 36), (41, 37), (41, 38), (41, 39), (41, 40),
        (40, 39),
        (39, 39), (38, 39), (37, 39), (37, 38), (37, 37), (37, 36), (37, 35), (37, 34), (36, 34), (35, 34), (34, 34),
        (33, 34),
        (32, 34), (31, 34), (30, 34), (29, 34), (28, 34), (27, 34), (26, 34), (25, 34), (24, 34), (23, 34), (22, 34),
        (21, 34),
        (20, 34), (19, 34), (18, 34), (17, 34), (16, 34), (15, 34), (14, 34), (13, 34), (12, 34), (11, 34), (10, 34),
        (9, 34),
        (8, 34), (7, 34), (6, 34), (5, 34), (4, 34), (3, 34), (2, 34), (1, 34), (0, 34), (0, 33), (0, 32), (0, 31),
        (0, 30),
        (0, 29), (0, 28), (0, 27), (0, 26), (0, 25), (0, 24), (0, 23), (0, 22), (0, 21), (0, 20), (0, 19), (0, 18),
        (0, 17),
        (0, 16), (0, 15), (0, 14), (0, 13), (0, 12), (0, 11), (0, 10), (0, 9), (0, 8), (0, 7), (0, 6), (0, 5), (0, 4),
        (0, 3), (0, 2),
        (0, 1), (0, 0), (0, 14), (1, 14), (2, 14), (3, 14), (4, 14), (5, 14), (6, 14), (7, 14), (8, 14),
        (9, 14), (10, 14), (11, 14), (12, 14), (13, 14), (14, 14), (15, 14), (16, 14), (17, 14), (18, 14), (19, 14),
        (20, 14),
        (21, 14), (22, 14), (23, 14), (24, 14), (25, 14), (26, 14), (27, 14), (28, 14), (29, 14), (30, 14), (31, 14),
        (32, 14),
        (33, 14), (33, 14), (33, 15), (33, 16), (33, 17), (33, 18), (33, 19), (33, 20), (33, 21), (33, 22), (33, 23),
        (33, 24), (33, 25), (33, 25), (33, 25), (32, 26), (31, 26), (30, 26), (29, 26), (28, 26), (27, 26), (26, 26),
        (25, 26), (24, 26), (23, 26), (22, 26), (21, 26), (20, 26), (19, 26), (18, 26), (17, 26), (16, 26), (15, 26),
        (14, 26),
        (13, 26), (12, 26), (11, 26), (10, 26), (9, 26), (8, 26), (7, 26), (6, 26), (5, 26), (4, 26), (3, 26), (2, 26),
        (1, 26), (0, 26),
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
        (0, 10), (1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10), (11, 10),
        (12, 10), (13, 10), (14, 10), (15, 10), (16, 10), (17, 10), (18, 10), (19, 10), (20, 10), (21, 10), (22, 10),
        (23, 10),
        (24, 10), (25, 10), (26, 10), (27, 10), (28, 10), (29, 10), (30, 10), (31, 10), (32, 10), (33, 10), (34, 10),
        (35, 10),
        (36, 10),
        (37, 11), (37, 12), (37, 13), (37, 14), (37, 15), (37, 16), (37, 17), (37, 18), (37, 19), (37, 20),
        (37, 21), (37, 22), (37, 23), (37, 24), (37, 25), (37, 26), (37, 27), (37, 28), (37, 29),
        (0, 30), (1, 30), (2, 30), (3, 30), (4, 30), (5, 30), (6, 30), (7, 30), (8, 30), (9, 30), (10, 30), (11, 30),
        (12, 30), (13, 30), (14, 30), (15, 30), (16, 30), (17, 30), (18, 30), (19, 30), (20, 30), (21, 30), (22, 30),
        (23, 30),
        (24, 30), (25, 30), (26, 30), (27, 30), (28, 30), (29, 30), (30, 30), (31, 30), (32, 30), (33, 30), (34, 30),
        (35, 30),
        (36, 30),
        (38, 34), (39, 34), (40, 34), (41, 34),
    ]

    # Set Initial parameters
    rrt = RRT(start=[2, 8],
              goal=[5, 32],
              rand_area_x=[0, 40],
              rand_area_y=[0, 40],
              obstacle_list=obstacleList)
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("path is found")
        print("number of nodes in the final path: ", len(path))

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.show()

        if not show_animation:

            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)

            ax = plt.axes()
            ax.set_xlim(-5, 46)
            ax.set_ylim(-5, 41)
            ax.set_ylabel('vertical length (40 m)')
            ax.set_xlabel('horizontal length (41 m)')
            ax.set_title('parking lot with occupied spots')
            for car in cars:
                ax.add_patch(car)

            plt.plot(x1, y1, linestyle='-', color='k', lw=2.5)
            plt.plot(x2, y2, linestyle='-', color='k', lw=2)
            plt.plot(x3, y3, x4, y4, x5, y5, x6, y6, linestyle='-', color='y', lw=1.5)
            plt.plot(x7, y7, linestyle='--', color='y', lw=1)

            plt.show()


if __name__ == '__main__':
    main()
