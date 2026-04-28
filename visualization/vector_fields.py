"""Gradient vector-field plotting utilities."""

import numpy as np
import matplotlib.pyplot as plt
from core.gradients import Gradient
from visualization.contours import ContourPlotter


class VectorFieldPlotter:
    """Plot gradient vector fields."""

    @staticmethod
    def plot_gradient_field(f, x_range=(-5, 5), y_range=(-5, 5),
                            arrows_per_side=15, ax=None, title="Gradient Vector Field",
                            trajectories=None, normalize=True):
        """Plot -∇f as vector field.

        Args:
            f: scalar function
            x_range, y_range: ranges for axes
            arrows_per_side: number of arrows per axis
            ax: matplotlib axis (optional)
            title: plot title
            trajectories: list of Trajectory objects to overlay (optional)
            normalize: whether to normalize arrow lengths for visibility

        Returns:
            fig, ax if ax was None, else just ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            created_fig = True
        else:
            created_fig = False
            fig = ax.figure

        X, Y = ContourPlotter.create_mesh(x_range, y_range, resolution=150)
        Z = ContourPlotter.evaluate_function(f, X, Y)

        contourf = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
        ax.contour(X, Y, Z, levels=20, colors='black',
                   alpha=0.2, linewidths=0.5)
        plt.colorbar(contourf, ax=ax, label='f(x)')

        x_arrow = np.linspace(x_range[0], x_range[1], arrows_per_side)
        y_arrow = np.linspace(y_range[0], y_range[1], arrows_per_side)

        U = np.zeros((arrows_per_side, arrows_per_side))
        V = np.zeros((arrows_per_side, arrows_per_side))

        for i, x0 in enumerate(x_arrow):
            for j, y0 in enumerate(y_arrow):
                point = np.array([x0, y0])
                grad = Gradient.get_grad(f, point)

                U[j, i] = -grad[0]
                V[j, i] = -grad[1]

        if normalize:
            magnitudes = np.sqrt(U**2 + V**2)
            magnitudes = np.where(magnitudes == 0, 1, magnitudes)
            U = U / magnitudes
            V = V / magnitudes

        ax.quiver(x_arrow, y_arrow, U, V, alpha=0.6, scale=25, width=0.004)

        if trajectories is not None:
            if not isinstance(trajectories, list):
                trajectories = [trajectories]

            colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

            for traj, color in zip(trajectories, colors):
                path = traj.trajectory
                ax.plot(path[:, 0], path[:, 1], 'o-', color=color,
                        markersize=4, linewidth=2, label=traj.optimizer_name, alpha=0.8)

                ax.plot(path[0, 0], path[0, 1], 'o', color=color, markersize=10,
                        markeredgecolor='white', markeredgewidth=2)
                ax.plot(path[-1, 0], path[-1, 1], 's', color=color, markersize=10,
                        markeredgecolor='white', markeredgewidth=2)

            ax.legend(loc='best', fontsize=11)

        ax.set_xlabel('x₀', fontsize=12)
        ax.set_ylabel('x₁', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')

        if created_fig:
            return fig, ax
        else:
            return ax
