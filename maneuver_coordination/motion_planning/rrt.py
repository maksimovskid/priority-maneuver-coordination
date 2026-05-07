"""
Path planning sample code with RRT.

author: Atsushi Sakai(@Atsushi_twi)
modified by Daniel Maksimovski
"""

import math
import random

import matplotlib.pyplot as plt

show_animation = False


class RRT:
    """Rapidly-exploring Random Tree planner for 2D point paths."""

    class Node:
        """RRT tree node with sampled path points back to its parent."""

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(
        self,
        start,
        goal,
        obstacle_list,
        rand_area_x,
        rand_area_y,
        expand_dis=4,
        path_resolution=1,
        goal_sample_rate=10,
        max_iter=1000,
    ):
        """
        Configure the RRT planner.

        start: start position [x, y]
        goal: goal position [x, y]
        obstacle_list: obstacle positions [(x, y), ...]
        rand_area_x: x sampling range [min, max]
        rand_area_y: y sampling range [min, max]
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
        """Run RRT path planning and return the final path if one is found."""
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:  # pragma: no cover
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """Extend from one node toward another by the configured step length."""
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
        """Reconstruct the final path from goal to start."""
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

    def calc_dist_to_goal(self, x, y):
        """Return Euclidean distance from a point to the goal."""
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        """Sample either a random free-space node or the goal node."""
        if random.randint(0, 100) > self.goal_sample_rate:
            return self.Node(
                random.uniform(self.min_rand_x, self.max_rand_x),
                random.uniform(self.min_rand_y, self.max_rand_y),
            )
        return self.Node(self.end.x, self.end.y)

    def draw_graph(self, rnd=None):  # pragma: no cover
        """Draw the current RRT tree and obstacle points."""
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

        for ox, oy in self.obstacle_list:
            plt.plot(ox, oy, ".k")

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        """Return the index of the tree node nearest to a sampled node."""
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in node_list]
        return dlist.index(min(dlist))

    @staticmethod
    def check_collision(node, obstacle_list):
        """Return True when the node path is clear of all point obstacles."""
        if node is None:
            return False

        for ox, oy in obstacle_list:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for dx, dy in zip(dx_list, dy_list)]
            if min(d_list) <= 1.1:
                return False

        return True

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """Return distance and heading from one node to another."""
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


def build_demo_obstacles():
    """Build a compact obstacle set for the standalone RRT demo."""
    obstacles = []

    # Outer rectangular boundary.
    obstacles.extend((x, 0) for x in range(0, 51))
    obstacles.extend((x, 30) for x in range(0, 51))
    obstacles.extend((0, y) for y in range(0, 31))
    obstacles.extend((50, y) for y in range(0, 31))

    # Two internal walls with openings to make the search non-trivial.
    obstacles.extend((20, y) for y in range(0, 22))
    obstacles.extend((35, y) for y in range(8, 31))

    return obstacles


def main():
    """Run a small standalone RRT demo."""
    print("start " + __file__)
    obstacle_list = build_demo_obstacles()
    rrt = RRT(
        start=[5, 5],
        goal=[45, 25],
        rand_area_x=[0, 50],
        rand_area_y=[0, 30],
        obstacle_list=obstacle_list,
    )
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
        return

    print("path is found")
    print("number of nodes in the final path:", len(path))

    if show_animation:  # pragma: no cover
        rrt.draw_graph()
        plt.plot([x for x, _ in path], [y for _, y in path], "-r")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
