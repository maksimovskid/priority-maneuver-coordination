"""
Path planning Sample Code with RRT and Reeds-Shepp steering.

author: Atsushi Sakai(@Atsushi_twi)
modified by: Daniel Maksimovski
"""

import copy
import math
import random
import timeit

import matplotlib.pyplot as plt
import numpy as np

from maneuver_coordination.motion_planning import reeds_shepp_path_planning
from maneuver_coordination.motion_planning.rrt import RRT

show_animation = False


class RRTReedsShepp(RRT):
    """
    Class for RRT planning with Reeds-Shepp steering.
    """

    class Node(RRT.Node):
        def __init__(self, x, y, yaw):
            super().__init__(x, y)
            self.cost = 0
            self.yaw = yaw
            self.path_yaw = []

    def __init__(self, start, goal, obstacle_list, rand_area, goal_sample_rate=10, max_iter=10):
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

        self.curvature = 0.18
        self.goal_yaw_th = np.deg2rad(30.0)
        self.goal_xy_th = 1.0

    def planning(self, animation=True, search_until_max_iter=False):
        _ = animation
        start = timeit.default_timer()
        self.node_list = [self.start]

        for i in range(self.max_iter):
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd)

            if new_node and self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            if (not search_until_max_iter) and new_node:
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    return self.generate_final_course(last_index)

            stop = timeit.default_timer()

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)

        return None

    def draw_graph(self, rnd=None):  # pragma: no cover
        plt.clf()
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy) in self.obstacle_list:
            plt.plot(ox, oy, ".k")

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([-2, 45, -2, 45])
        plt.axis("equal")
        plt.grid(False)
        self.plot_start_goal_arrow()
        plt.pause(0.01)

    def plot_start_goal_arrow(self):  # pragma: no cover
        reeds_shepp_path_planning.plot_arrow(self.start.x, self.start.y, self.start.yaw)
        reeds_shepp_path_planning.plot_arrow(self.end.x, self.end.y, self.end.yaw)

    def steer(self, from_node, to_node):
        px, py, pyaw, mode, course_lengths = reeds_shepp_path_planning.reeds_shepp_path_planning(
            from_node.x,
            from_node.y,
            from_node.yaw,
            to_node.x,
            to_node.y,
            to_node.yaw,
            self.curvature,
        )

        if len(px) <= 0:
            return None

        new_node = copy.deepcopy(from_node)
        new_node.x = px[-1]
        new_node.y = py[-1]
        new_node.yaw = pyaw[-1]
        new_node.path_x = px
        new_node.path_y = py
        new_node.path_yaw = pyaw
        new_node.cost += sum(abs(length) for length in course_lengths)
        new_node.parent = from_node
        return new_node

    def calc_new_cost(self, from_node, to_node):
        _, _, _, _, course_lengths = reeds_shepp_path_planning.reeds_shepp_path_planning(
            from_node.x,
            from_node.y,
            from_node.yaw,
            to_node.x,
            to_node.y,
            to_node.yaw,
            self.curvature,
        )
        return from_node.cost + sum(abs(length) for length in course_lengths)

    def get_random_node(self):
        if random.randint(0, 1) > self.goal_sample_rate:
            return self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(-math.pi / 4, math.pi / 4),
            )
        return self.Node(self.end.x, self.end.y, self.end.yaw)

    def search_best_goal_node(self):
        goal_indexes = []
        for i, node in enumerate(self.node_list):
            if self.calc_dist_to_goal(node.x, node.y) <= self.goal_xy_th:
                goal_indexes.append(i)

        if not goal_indexes:
            return None

        min_cost = min(self.node_list[i].cost for i in goal_indexes)
        for i in goal_indexes:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def generate_final_course(self, goal_index):
        path = [[self.end.x, self.end.y, self.end.yaw]]
        node = self.node_list[goal_index]
        while node.parent:
            for ix, iy, iyaw in zip(
                reversed(node.path_x),
                reversed(node.path_y),
                reversed(node.path_yaw),
            ):
                path.append([ix, iy, iyaw])
            node = node.parent
        path.append([self.start.x, self.start.y, self.start.yaw])
        return path
