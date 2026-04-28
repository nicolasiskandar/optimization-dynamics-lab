"""Failure-mode demonstrations for gradient and Newton-style optimizers."""

import matplotlib.pyplot as plt
from functions.ill_conditioned import IllConditioned
from functions.saddle import Saddle
from functions.nonconvex import NonConvex
from functions.plateau import Plateau
from optimizers.gradient_descent import GradientDescent
from optimizers.momentum import MomentumGD
from optimizers.newton import Newton
from dynamics.trajectory import TrajectoryRunner
from visualization.trajectories import VisualizationRunner
from visualization.loss import LossPlotter


class FailureModes:
    """Demonstrate and visualize optimization failure modes."""

    @staticmethod
    def oscillation_in_narrow_valley(save_path=None):
        """GD oscillating in ill-conditioned landscape (narrow valley).

        Args:
            save_path: optional path to save figure

        Returns:
            fig, trajectories
        """
        function = IllConditioned(a=1, b=100)

        gd_small_lr = GradientDescent(step_size=0.01)
        gd_medium_lr = GradientDescent(step_size=0.05)
        gd_large_lr = GradientDescent(step_size=0.15)

        x0 = [4.0, -3.0]

        trajs = TrajectoryRunner.run_comparison(
            [gd_small_lr, gd_medium_lr, gd_large_lr], function, x0,
            steps=300, tol=1e-8
        )

        trajs[0].optimizer_name = "GD (η=0.01, slow)"
        trajs[1].optimizer_name = "GD (η=0.05, oscillating)"
        trajs[2].optimizer_name = "GD (η=0.15, diverging)"

        fig = VisualizationRunner.create_comparison_figure(
            function.f, trajs, x_range=(-5, 5), y_range=(-5, 5),
            title="Failure Mode 1: Oscillation in Ill-Conditioned Landscape"
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, trajs

    @staticmethod
    def newton_at_saddle(save_path=None):
        """Newton's method failing at saddle point.

        Args:
            save_path: optional path to save figure

        Returns:
            fig, trajectories
        """
        function = Saddle()

        gd = GradientDescent(step_size=0.05)
        newton = Newton()

        x0 = [0.5, 0.5]

        trajs = TrajectoryRunner.run_comparison(
            [gd, newton], function, x0, steps=100, tol=1e-6
        )

        fig = VisualizationRunner.create_comparison_figure(
            function.f, trajs, x_range=(-2, 2), y_range=(-2, 2),
            title="Failure Mode 2: Newton's Method at Saddle Point"
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, trajs

    @staticmethod
    def gd_stuck_on_plateau(save_path=None):
        """GD stuck on flat region.

        Args:
            save_path: optional path to save figure

        Returns:
            fig, trajectory
        """
        function = Plateau()

        gd = GradientDescent(step_size=0.1)

        x0 = [0.0, 2.0]

        traj = TrajectoryRunner.run(gd, function, x0, steps=500, tol=1e-8)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        from visualization.contours import ContourPlotter
        ContourPlotter.plot_contour(
            function.f, x_range=(-5, 5), y_range=(-5, 5),
            trajectories=[traj], ax=axes[0, 0],
            title="Gradient Descent Stuck on Plateau"
        )

        ContourPlotter.plot_contour(
            function.f, x_range=(-0.5, 0.5), y_range=(1.0, 3.0),
            trajectories=[traj], ax=axes[0, 1],
            title="Zoomed: Slow Escape from Plateau"
        )

        LossPlotter.plot_loss_curves(
            [traj], ax=axes[1, 0], title="Loss vs Iteration")
        LossPlotter.plot_gradient_norm(
            [traj], ax=axes[1, 1], title="Gradient Norm vs Iteration")

        fig.suptitle("Failure Mode 3: GD Stuck on Plateau",
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, traj

    @staticmethod
    def large_step_divergence(save_path=None):
        """Large step size causing divergence.

        Args:
            save_path: optional path to save figure

        Returns:
            fig, trajectories
        """
        function = IllConditioned(a=1, b=50)

        gd_small = GradientDescent(step_size=0.05)
        gd_diverging = GradientDescent(step_size=0.3)

        x0 = [3.0, -2.0]

        trajs = TrajectoryRunner.run_comparison(
            [gd_small, gd_diverging], function, x0, steps=50, tol=1e-6
        )

        trajs[0].optimizer_name = "GD (η=0.05, stable)"
        trajs[1].optimizer_name = "GD (η=0.30, diverging)"

        fig = VisualizationRunner.create_comparison_figure(
            function.f, trajs, x_range=(-10, 10), y_range=(-10, 10),
            title="Failure Mode 3: Divergence with Large Step Size"
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, trajs

    @staticmethod
    def momentum_overshoot(save_path=None):
        """Momentum overshooting in narrow valley.

        Args:
            save_path: optional path to save figure

        Returns:
            fig, trajectories
        """
        function = IllConditioned(a=1, b=100)

        gd = GradientDescent(step_size=0.08)
        momentum_low = MomentumGD(step_size=0.08, beta=0.5)
        momentum_high = MomentumGD(step_size=0.08, beta=0.95)

        x0 = [4.0, -3.0]

        trajs = TrajectoryRunner.run_comparison(
            [gd, momentum_low, momentum_high], function, x0,
            steps=200, tol=1e-8
        )

        trajs[0].optimizer_name = "GD (no momentum)"
        trajs[1].optimizer_name = "Momentum (β=0.5)"
        trajs[2].optimizer_name = "Momentum (β=0.95, overshooting)"

        fig = VisualizationRunner.create_comparison_figure(
            function.f, trajs, x_range=(-5, 5), y_range=(-5, 5),
            title="Failure Mode 4: Momentum Overshoot in Valley"
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, trajs

    @staticmethod
    def nonconvex_local_minima(save_path=None):
        """GD stuck at local minimum, missing global minimum.

        Args:
            save_path: optional path to save figure

        Returns:
            fig, trajectories
        """
        function = NonConvex()

        gd1 = GradientDescent(step_size=0.1)
        gd2 = GradientDescent(step_size=0.1)

        x0_1 = [1.0, 2.0]
        x0_2 = [-1.5, 2.0]

        traj1 = TrajectoryRunner.run(gd1, function, x0_1, steps=200, tol=1e-6)
        traj2 = TrajectoryRunner.run(gd2, function, x0_2, steps=200, tol=1e-6)

        traj1.optimizer_name = "From x₀=(1.0, 2.0) [local min]"
        traj2.optimizer_name = "From x₀=(-1.5, 2.0) [global min]"

        fig = VisualizationRunner.create_comparison_figure(
            function.f, [traj1, traj2], x_range=(-3, 3), y_range=(-2, 4),
            title="Failure Mode 5: Local Minima in Non-Convex Landscape"
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, [traj1, traj2]

    @staticmethod
    def create_all_failure_demos(output_dir="failure_modes"):
        """Generate all failure mode demonstrations.

        Args:
            output_dir: directory to save figures

        Returns:
            dict of figures
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        demos = {}

        print("Generating failure modes demonstrations...")

        print("1. Oscillation in narrow valley...")
        fig1, _ = FailureModes.oscillation_in_narrow_valley(
            f"{output_dir}/01_narrow_valley.png"
        )
        demos['narrow_valley'] = fig1

        print("2. Newton at saddle point...")
        fig2, _ = FailureModes.newton_at_saddle(
            f"{output_dir}/02_saddle_newton.png"
        )
        demos['saddle_newton'] = fig2

        print("3. Plateau stalling...")
        fig3, _ = FailureModes.gd_stuck_on_plateau(
            f"{output_dir}/03_plateau.png"
        )
        demos['plateau'] = fig3

        print("4. Large step divergence...")
        fig4, _ = FailureModes.large_step_divergence(
            f"{output_dir}/04_divergence.png"
        )
        demos['divergence'] = fig4

        print("5. Momentum overshoot...")
        fig5, _ = FailureModes.momentum_overshoot(
            f"{output_dir}/05_momentum_overshoot.png"
        )
        demos['momentum_overshoot'] = fig5

        print("6. Local minima in non-convex...")
        fig6, _ = FailureModes.nonconvex_local_minima(
            f"{output_dir}/06_local_minima.png"
        )
        demos['local_minima'] = fig6

        print(f"✓ All failure modes saved to {output_dir}/")

        return demos


if __name__ == "__main__":
    FailureModes.create_all_failure_demos()
