"""Multi-start and basin-of-attraction analyses for optimizer comparison."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fclusterdata
from dynamics.trajectory import TrajectoryRunner


class MultiStartAnalyzer:
    """Analyze optimizer behavior across multiple starting points."""

    @staticmethod
    def generate_grid_starts(x_range=(-5, 5), y_range=(-5, 5), n_per_axis=5):
        """Generate grid of starting points.

        Args:
            x_range, y_range: ranges for each dimension
            n_per_axis: number of points per axis

        Returns:
            array of starting points, shape (n_points, 2)
        """
        x_points = np.linspace(x_range[0], x_range[1], n_per_axis)
        y_points = np.linspace(y_range[0], y_range[1], n_per_axis)

        starts = []
        for x in x_points:
            for y in y_points:
                starts.append([x, y])

        return np.array(starts)

    @staticmethod
    def generate_random_starts(x_range=(-5, 5), y_range=(-5, 5), n_points=25,
                               seed=None):
        """Generate random starting points.

        Args:
            x_range, y_range: ranges for each dimension
            n_points: number of random points
            seed: random seed for reproducibility

        Returns:
            array of starting points, shape (n_points, 2)
        """
        if seed is not None:
            np.random.seed(seed)

        starts = np.column_stack([
            np.random.uniform(x_range[0], x_range[1], n_points),
            np.random.uniform(y_range[0], y_range[1], n_points)
        ])

        return starts

    @staticmethod
    def run_multistart(optimizer, function, starting_points,
                       steps=100, tol=1e-6):
        """Run optimization from multiple starting points.

        Args:
            optimizer: optimizer object
            function: Function2D object
            starting_points: array of starting points, shape (n_points, 2)
            steps: max iterations per run
            tol: convergence tolerance

        Returns:
            trajectories: list of Trajectory objects
        """
        trajectories = TrajectoryRunner.run_multistart(
            optimizer, function, starting_points, steps=steps, tol=tol
        )

        return trajectories

    @staticmethod
    def cluster_minima(trajectories, distance_threshold=0.5):
        """Cluster final points to identify unique minima.

        Args:
            trajectories: list of Trajectory objects
            distance_threshold: clustering threshold

        Returns:
            dict mapping cluster_id to list of trajectory indices
        """
        final_points = np.array([traj.x_final for traj in trajectories])

        if len(final_points) < 2:
            return {0: list(range(len(trajectories)))}

        try:
            clusters = fclusterdata(
                final_points,
                t=distance_threshold,
                criterion='distance',
                method='complete'
            )

            cluster_map = {}
            for idx, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_map:
                    cluster_map[cluster_id] = []
                cluster_map[cluster_id].append(idx)

            return cluster_map
        except:
            return {0: list(range(len(trajectories)))}

    @staticmethod
    def plot_multistart_trajectories(f, trajectories, x_range=(-5, 5),
                                     y_range=(-5, 5), ax=None,
                                     title="Multi-Start Analysis",
                                     color_by_cluster=True):
        """Plot all trajectories from multiple starting points.

        Args:
            f: scalar function
            trajectories: list of Trajectory objects
            x_range, y_range: axis ranges
            ax: matplotlib axis (optional)
            title: plot title
            color_by_cluster: whether to color by final point cluster

        Returns:
            fig, ax if ax was None else ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
            created_fig = True
        else:
            created_fig = False
            fig = ax.figure

        from visualization.contours import ContourPlotter
        X, Y = ContourPlotter.create_mesh(x_range, y_range, resolution=150)
        Z = ContourPlotter.evaluate_function(f, X, Y)

        contourf = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
        ax.contour(X, Y, Z, levels=20, colors='black',
                   alpha=0.2, linewidths=0.5)
        plt.colorbar(contourf, ax=ax, label='f(x)')

        if color_by_cluster:
            clusters = MultiStartAnalyzer.cluster_minima(trajectories)
            colors_list = plt.cm.tab20(np.linspace(0, 1, len(clusters)))

            for cluster_id, color in zip(sorted(clusters.keys()), colors_list):
                indices = clusters[cluster_id]

                for idx in indices:
                    traj = trajectories[idx]
                    path = traj.trajectory

                    ax.plot(path[:, 0], path[:, 1], 'o-', color=color,
                            markersize=2, linewidth=0.8, alpha=0.6)

                    ax.plot(path[-1, 0], path[-1, 1], 's', color=color,
                            markersize=6, markeredgecolor='black', markeredgewidth=0.5)

                final_points_cluster = np.array([
                    trajectories[i].x_final for i in indices
                ])
                center = np.mean(final_points_cluster, axis=0)
                ax.plot(center[0], center[1], '*', color=color, markersize=20,
                        markeredgecolor='black', markeredgewidth=1, zorder=100)
        else:
            for traj in trajectories:
                path = traj.trajectory
                ax.plot(path[:, 0], path[:, 1], 'o-', color='blue',
                        markersize=2, linewidth=0.8, alpha=0.5)

                ax.plot(path[0, 0], path[0, 1], 'o', color='green',
                        markersize=4, markeredgecolor='black', markeredgewidth=0.5)
                ax.plot(path[-1, 0], path[-1, 1], 's', color='red',
                        markersize=4, markeredgecolor='black', markeredgewidth=0.5)

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

    @staticmethod
    def plot_basin_of_attraction(f, optimizer, function, x_range=(-5, 5),
                                 y_range=(-5, 5), resolution=50, ax=None,
                                 n_steps=100, title="Basin of Attraction"):
        """Visualize basin of attraction by color-coding final minima.

        Args:
            f: scalar function
            optimizer: optimizer object
            function: Function2D object
            x_range, y_range: axis ranges
            resolution: grid resolution
            ax: matplotlib axis (optional)
            n_steps: max iterations per run
            title: plot title

        Returns:
            fig, ax if ax was None else ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            created_fig = True
        else:
            created_fig = False
            fig = ax.figure

        print("Computing basin of attraction (this may take a moment)...")

        x_pts = np.linspace(x_range[0], x_range[1], resolution)
        y_pts = np.linspace(y_range[0], y_range[1], resolution)
        X_grid, Y_grid = np.meshgrid(x_pts, y_pts)

        basin = np.zeros_like(X_grid)

        for i in range(resolution):
            for j in range(resolution):
                x0 = np.array([X_grid[i, j], Y_grid[i, j]])
                traj = TrajectoryRunner.run(optimizer, function, x0,
                                            steps=n_steps, tol=1e-6)
                basin[i, j] = traj.f_final

        contourf = ax.contourf(X_grid, Y_grid, basin,
                               levels=20, cmap='tab20', alpha=0.8)
        ax.contour(X_grid, Y_grid, basin, levels=20,
                   colors='black', alpha=0.3, linewidths=0.5)
        plt.colorbar(contourf, ax=ax, label='f(x_final)')

        ax.set_xlabel('x₀', fontsize=12)
        ax.set_ylabel('x₁', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_aspect('equal')

        if created_fig:
            return fig, ax
        else:
            return ax
