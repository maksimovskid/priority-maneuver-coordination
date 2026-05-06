"""Low-level Matplotlib vehicle drawing primitives."""

import math

import matplotlib.pyplot as plt
import numpy as np

from maneuver_coordination.vehicle.model import LB, LENGTH, W


def _normalize_color(color: str) -> str:
    return color[1:] if color.startswith("-") else color


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="b", ec="k"):
    """Plot one or more vehicle heading arrows."""
    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw, length=length, width=width, fc=fc, ec=ec)
        return

    plt.arrow(
        x,
        y,
        length * math.cos(yaw),
        length * math.sin(yaw),
        fc=fc,
        ec=ec,
        head_width=width,
        head_length=width,
        alpha=0.4,
    )


def _vehicle_outline(x, y, yaw):
    outline = np.array(
        [[-LB, LENGTH - LB, LENGTH - LB, -LB, -LB], [W / 2, W / 2, -W / 2, -W / 2, W / 2]]
    )
    rotation = np.array(
        [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]
    )
    outline = (outline.T.dot(rotation)).T
    outline[0, :] += x
    outline[1, :] += y
    return outline


def _plot_vehicle(x, y, yaw, marker, arrow_color, truckcolor):
    outline = _vehicle_outline(x, y, yaw)
    plt.plot(np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), truckcolor)
    arrow_x = math.cos(yaw) * 1.5 + x
    arrow_y = math.sin(yaw) * 1.5 + y
    plot_arrow(arrow_x, arrow_y, yaw, fc=arrow_color)
    if marker:
        plt.plot(x, y, marker)


def plot_car(x, y, yaw):
    """Plot the default single-vehicle marker."""
    _plot_vehicle(x, y, yaw, marker="*b", arrow_color="b", truckcolor="-k")


def plot_cars(x, y, yaw, x2, y2, yaw2):
    """Plot two legacy vehicle markers for older demos."""
    _plot_vehicle(x, y, yaw, marker="*b", arrow_color="b", truckcolor="-b")
    _plot_vehicle(x2, y2, yaw2, marker="*r", arrow_color="r", truckcolor="-r")


def plot_car1(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
    """Plot vehicle slot 1 using the shared outline renderer."""
    _ = (steer, cabcolor)
    color = _normalize_color(truckcolor)
    _plot_vehicle(x, y, yaw, marker=None, arrow_color=color, truckcolor=truckcolor)


def plot_car2(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
    """Plot vehicle slot 2 using the shared outline renderer."""
    _ = (steer, cabcolor)
    color = _normalize_color(truckcolor)
    _plot_vehicle(x, y, yaw, marker=None, arrow_color=color, truckcolor=truckcolor)


def plot_car3(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
    """Plot vehicle slot 3 using the shared outline renderer."""
    _ = (steer, cabcolor)
    color = _normalize_color(truckcolor)
    _plot_vehicle(x, y, yaw, marker=None, arrow_color=color, truckcolor=truckcolor)


def plot_car4(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
    """Plot vehicle slot 4 using the shared outline renderer."""
    _ = (steer, cabcolor)
    color = _normalize_color(truckcolor)
    _plot_vehicle(x, y, yaw, marker=None, arrow_color=color, truckcolor=truckcolor)


def plot_car5(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
    """Plot vehicle slot 5 using the shared outline renderer."""
    _ = (steer, cabcolor)
    color = _normalize_color(truckcolor)
    _plot_vehicle(x, y, yaw, marker=None, arrow_color=color, truckcolor=truckcolor)
