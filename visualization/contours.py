"""2D contour plotting helpers."""

import numpy as np
import matplotlib.pyplot as plt


class ContourPlotter:
    """Generate 2D contour plots of functions."""

    @staticmethod
    def create_mesh(x_range=(-5, 5), y_range=(-5, 5), resolution=200):
        """Create a mesh for contour plotting.

        Args:
            x_range: tuple (x_min, x_max)
            y_range: tuple (y_min, y_max)
            resolution: number of points per axis

        Returns:
            X, Y: meshgrid arrays
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        return X, Y

    @staticmethod
    def evaluate_function(f, X, Y):
        """Evaluate function on mesh.

        Args:
            f: scalar function f(x) where x = [x0, x1]
            X, Y: meshgrid arrays

        Returns:
            Z: function values on mesh
        """
        try:
            # Try vectorized evaluation first
            return f(np.stack([X, Y], axis=0))
        except (ValueError, TypeError, IndexError):
            # Fallback to loop if function doesn't support vectorization
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
            return Z

    @staticmethod
    def plot_contour(f, x_range=(-5, 5), y_range=(-5, 5),
                     trajectories=None, ax=None, title="Contour Plot",
                     levels=None, cmap='viridis'):
        """Plot contours with optional trajectory overlay.

        Args:
            f: scalar function
            x_range, y_range: ranges for axes
            trajectories: list of Trajectory objects (optional)
            ax: matplotlib axis (optional)
            title: plot title
            levels: number of contour levels (default: auto)
            cmap: colormap name

        Returns:
            fig, ax if ax was None, else just ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            created_fig = True
        else:
            created_fig = False
            fig = ax.figure

        X, Y = ContourPlotter.create_mesh(x_range, y_range)
        Z = ContourPlotter.evaluate_function(f, X, Y)

        if levels is None:
            contour = ax.contour(X, Y, Z, levels=20,
                                 colors='black', alpha=0.4, linewidths=0.5)
            contourf = ax.contourf(X, Y, Z, levels=20, cmap=cmap, alpha=0.8)
            plt.colorbar(contourf, ax=ax, label='f(x)')
        else:
            contour = ax.contour(X, Y, Z, levels=levels,
                                 colors='black', alpha=0.4)
            contourf = ax.contourf(
                X, Y, Z, levels=levels, cmap=cmap, alpha=0.8)
            plt.colorbar(contourf, ax=ax, label='f(x)')

        if trajectories is not None:
            if not isinstance(trajectories, list):
                trajectories = [trajectories]

            colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

            for traj, color in zip(trajectories, colors):
                path = traj.trajectory
                ax.plot(path[:, 0], path[:, 1], 'o-', color=color,
                        markersize=3, linewidth=1.5, label=traj.optimizer_name, alpha=0.7)

                ax.plot(path[0, 0], path[0, 1], 'o', color=color, markersize=8,
                        markeredgecolor='black', markeredgewidth=1.5)
                ax.plot(path[-1, 0], path[-1, 1], 's', color=color, markersize=8,
                        markeredgecolor='black', markeredgewidth=1.5)

            ax.legend(loc='best', fontsize=10)

        ax.set_xlabel('x₀', fontsize=12)
        ax.set_ylabel('x₁', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        if created_fig:
            return fig, ax
        else:
            return ax
