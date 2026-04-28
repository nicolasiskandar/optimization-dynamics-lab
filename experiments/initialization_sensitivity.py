"""Initialization-sensitivity analyses for non-convex optimization dynamics."""

import numpy as np
import matplotlib.pyplot as plt
from functions.nonconvex import NonConvex
from optimizers.gradient_descent import GradientDescent
from optimizers.momentum import MomentumGD
from dynamics.trajectory import TrajectoryRunner
from visualization.trajectories import VisualizationRunner
from experiments.compare_optimizers import MultiStartAnalyzer


class InitializationSensitivity:
    """Analyze how initial point affects convergence."""

    @staticmethod
    def distance_from_optimum(trajectory, x_opt):
        """Compute distance from optimum at each iteration.

        Args:
            trajectory: Trajectory object
            x_opt: optimal point

        Returns:
            array of distances
        """
        x_opt = np.array(x_opt)
        distances = np.array([
            np.linalg.norm(x - x_opt)
            for x in trajectory.trajectory
        ])
        return distances

    @staticmethod
    def convergence_rate_analysis(optimizer, function, starting_points,
                                  x_opt, n_steps=200):
        """Analyze convergence rates from different starting points.

        Args:
            optimizer: optimizer object
            function: Function2D object
            starting_points: list of starting points
            x_opt: optimal point
            n_steps: max iterations

        Returns:
            dict with convergence metrics
        """
        trajectories = []
        distances = []

        for x0 in starting_points:
            traj = TrajectoryRunner.run(optimizer, function, x0,
                                        steps=n_steps, tol=1e-8)
            trajectories.append(traj)

            dist = InitializationSensitivity.distance_from_optimum(traj, x_opt)
            distances.append(dist)

        return {
            'trajectories': trajectories,
            'distances': distances,
            'starting_points': starting_points
        }

    @staticmethod
    def plot_convergence_rates(results, ax=None, title="Convergence Rates"):
        """Plot convergence rates from different starting points.

        Args:
            results: dict from convergence_rate_analysis
            ax: matplotlib axis (optional)
            title: plot title

        Returns:
            fig, ax if ax was None else ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            created_fig = True
        else:
            created_fig = False
            fig = ax.figure

        distances = results['distances']

        colors = plt.cm.viridis(np.linspace(0, 1, len(distances)))

        for i, (dist, color) in enumerate(zip(distances, colors)):
            iterations = np.arange(len(dist))

            dist_nonzero = np.where(dist > 1e-12, dist, 1e-12)

            ax.semilogy(iterations, dist_nonzero, 'o-', markersize=2,
                        linewidth=1.5, color=color, alpha=0.7,
                        label=f'Start {i+1}')

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('||x_t - x*|| [log]', fontsize=12)
        
        from matplotlib.ticker import LogFormatter
        ax.yaxis.set_major_formatter(LogFormatter())
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', ncol=2)
        ax.grid(True, alpha=0.3, which='both')

        if created_fig:
            return fig, ax
        else:
            return ax

    @staticmethod
    def analyze_nonconvex_function(n_starts=16, save_dir=None):
        """Analyze initialization sensitivity on non-convex function.

        Args:
            n_starts: number of random starting points
            save_dir: optional directory to save figures

        Returns:
            dict with results
        """
        function = NonConvex()

        x_opt_1 = np.array([-np.pi, 0.0])

        x_range = (-3, 3)
        y_range = (-2, 4)
        starting_points = MultiStartAnalyzer.generate_random_starts(
            x_range, y_range, n_points=n_starts, seed=42
        )

        gd = GradientDescent(step_size=0.1)
        momentum = MomentumGD(step_size=0.1, beta=0.9)

        print(
            f"Running {n_starts} optimizations from different starting points...")

        results_gd = InitializationSensitivity.convergence_rate_analysis(
            gd, function, starting_points, x_opt_1, n_steps=300
        )

        results_momentum = InitializationSensitivity.convergence_rate_analysis(
            momentum, function, starting_points, x_opt_1, n_steps=300
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        InitializationSensitivity.plot_convergence_rates(
            results_gd, ax=axes[0, 0],
            title="GD: Distance to Optimum"
        )

        InitializationSensitivity.plot_convergence_rates(
            results_momentum, ax=axes[0, 1],
            title="Momentum: Distance to Optimum"
        )

        VisualizationRunner.plot_3d_surface(
            function.f,
            trajectories=results_gd['trajectories'][:5],
            x_range=x_range, y_range=y_range,
            title="GD: Multiple Starting Points", show=False
        )

        ax_loss = axes[1, 0]

        init_distances_gd = [
            np.linalg.norm(x0 - x_opt_1)
            for x0 in starting_points
        ]
        final_losses_gd = [
            traj.f_final
            for traj in results_gd['trajectories']
        ]
        final_losses_momentum = [
            traj.f_final
            for traj in results_momentum['trajectories']
        ]

        ax_loss.scatter(init_distances_gd, final_losses_gd, s=100, alpha=0.6,
                        label='GD', color='blue')
        ax_loss.scatter(init_distances_gd, final_losses_momentum, s=100, alpha=0.6,
                        label='Momentum', color='orange')

        ax_loss.set_xlabel('Distance from optimum at start', fontsize=12)
        ax_loss.set_ylabel('f(x_final)', fontsize=12)
        ax_loss.set_title('Final Loss vs Initial Distance',
                          fontsize=12, fontweight='bold')
        ax_loss.legend(fontsize=11)
        ax_loss.grid(True, alpha=0.3)

        ax_steps = axes[1, 1]

        n_steps_gd = [traj.n_steps for traj in results_gd['trajectories']]
        n_steps_momentum = [
            traj.n_steps for traj in results_momentum['trajectories']]

        ax_steps.scatter(init_distances_gd, n_steps_gd, s=100, alpha=0.6,
                         label='GD', color='blue', marker='o')
        ax_steps.scatter(init_distances_gd, n_steps_momentum, s=100, alpha=0.6,
                         label='Momentum', color='orange', marker='s')

        ax_steps.set_xlabel('Distance from optimum at start', fontsize=12)
        ax_steps.set_ylabel('Steps to convergence', fontsize=12)
        ax_steps.set_title(
            'Convergence Speed vs Initial Distance', fontsize=12, fontweight='bold')
        ax_steps.legend(fontsize=11)
        ax_steps.grid(True, alpha=0.3)

        plt.suptitle("Initialization Sensitivity Analysis (Non-Convex Function)",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_dir:
            fig.savefig(f"{save_dir}/initialization_sensitivity.png",
                        dpi=300, bbox_inches='tight')
            print(f"Saved to {save_dir}/initialization_sensitivity.png")

        return {
            'figure': fig,
            'results_gd': results_gd,
            'results_momentum': results_momentum,
            'init_distances': init_distances_gd,
            'final_losses_gd': final_losses_gd,
            'final_losses_momentum': final_losses_momentum
        }


if __name__ == "__main__":
    print("Analyzing initialization sensitivity...")

    results = InitializationSensitivity.analyze_nonconvex_function(
        n_starts=16, save_dir="."
    )

    print("✓ Initialization sensitivity analysis complete")
