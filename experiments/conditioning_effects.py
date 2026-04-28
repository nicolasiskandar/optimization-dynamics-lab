"""Experiments that measure optimization behavior across conditioning regimes."""

import numpy as np
import matplotlib.pyplot as plt
from functions.ill_conditioned import IllConditioned
from optimizers.gradient_descent import GradientDescent
from optimizers.momentum import MomentumGD
from dynamics.trajectory import TrajectoryRunner


class ConditioningEffects:
    """Analyze how conditioning (Hessian eigenvalue ratio) affects optimization."""

    @staticmethod
    def compare_conditioning_levels(step_size=0.05, n_steps=300):
        """Compare optimizers on quadratics with different conditioning.

        Args:
            step_size: step size for optimizers
            n_steps: max iterations

        Returns:
            dict of trajectories grouped by condition number
        """
        condition_levels = {
            'Well-conditioned (κ=2)': {'a': 1, 'b': 2},
            'Moderate (κ=10)': {'a': 1, 'b': 10},
            'Ill-conditioned (κ=100)': {'a': 1, 'b': 100},
            'Very ill-conditioned (κ=1000)': {'a': 1, 'b': 1000},
        }

        x0 = [5.0, -5.0]

        results = {}

        for name, params in condition_levels.items():
            function = IllConditioned(a=params['a'], b=params['b'])

            gd = GradientDescent(step_size=step_size)
            momentum = MomentumGD(step_size=step_size, beta=0.9)

            trajs = TrajectoryRunner.run_comparison(
                [gd, momentum], function, x0, steps=n_steps, tol=1e-6
            )

            cond_num = params['b'] / params['a']
            trajs[0].optimizer_name = f"GD"
            trajs[1].optimizer_name = f"Momentum"

            results[name] = {
                'trajectories': trajs,
                'condition_number': cond_num,
                'function': function
            }

        return results

    @staticmethod
    def plot_conditioning_comparison(results, save_path=None):
        """Create figure showing effect of conditioning on convergence.

        Args:
            results: dict from compare_conditioning_levels
            save_path: optional path to save figure

        Returns:
            fig with subplots
        """
        n_conditions = len(results)

        fig, axes = plt.subplots(2, n_conditions, figsize=(16, 10))

        for col, (cond_name, result_data) in enumerate(results.items()):
            trajs = result_data['trajectories']

            ax_loss = axes[0, col]
            for traj in trajs:
                f_vals = traj.diagnostics['function_values']
                ax_loss.semilogy(f_vals, 'o-', markersize=2, linewidth=1.5,
                                 label=traj.optimizer_name, alpha=0.8)

            ax_loss.set_title(f"{cond_name}\n(κ={result_data['condition_number']:.0f})",
                              fontsize=11, fontweight='bold')
            ax_loss.set_xlabel('Iteration')
            ax_loss.set_ylabel('f(x) [log]')
            
            from matplotlib.ticker import LogFormatter
            ax_loss.yaxis.set_major_formatter(LogFormatter())
            
            ax_loss.legend(fontsize=9)
            ax_loss.grid(True, alpha=0.3, which='both')

            ax_grad = axes[1, col]
            for traj in trajs:
                grad_norms = traj.diagnostics['gradient_norms']
                ax_grad.semilogy(grad_norms, 'o-', markersize=2, linewidth=1.5,
                                 label=traj.optimizer_name, alpha=0.8)

            ax_grad.set_xlabel('Iteration')
            ax_grad.set_ylabel('||∇f(x)|| [log]')
            
            from matplotlib.ticker import LogFormatter
            ax_grad.yaxis.set_major_formatter(LogFormatter())
            
            ax_grad.legend(fontsize=9)
            ax_grad.grid(True, alpha=0.3, which='both')

        plt.suptitle("Effect of Conditioning on Convergence\n(Higher κ = harder to optimize)",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig

    @staticmethod
    def eigenvalue_evolution(function, trajectory, ax=None, title="Hessian Eigenvalues"):
        """Plot how Hessian eigenvalues evolve during optimization.

        Args:
            function: Function2D object
            trajectory: Trajectory object
            ax: matplotlib axis (optional)
            title: plot title

        Returns:
            fig, ax if ax was None else ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            created_fig = True
        else:
            created_fig = False
            fig = ax.figure

        eigenvalues_list = trajectory.diagnostics['eigenvalues']

        n_iterations = len(eigenvalues_list)

        lambda_0 = [ev[0] if len(
            ev) > 0 else np.nan for ev in eigenvalues_list]
        lambda_1 = [ev[1] if len(
            ev) > 1 else np.nan for ev in eigenvalues_list]

        iterations = np.arange(n_iterations)

        ax.semilogy(iterations, np.abs(lambda_0), 'o-', label='λ₁',
                    markersize=3, linewidth=2, alpha=0.7)
        ax.semilogy(iterations, np.abs(lambda_1), 's-', label='λ₂',
                    markersize=3, linewidth=2, alpha=0.7)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('|λᵢ|', fontsize=12)
        
        from matplotlib.ticker import LogFormatter
        ax.yaxis.set_major_formatter(LogFormatter())
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')

        if created_fig:
            return fig, ax
        else:
            return ax

    @staticmethod
    def condition_number_metric(trajectory, ax=None):
        """Plot condition number evolution.

        Args:
            trajectory: Trajectory object
            ax: matplotlib axis (optional)

        Returns:
            fig, ax if ax was None else ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            created_fig = True
        else:
            created_fig = False
            fig = ax.figure

        cond_nums = trajectory.diagnostics['condition_numbers']

        iterations = np.arange(len(cond_nums))

        cond_nums_plot = np.array(
            [c if np.isfinite(c) else 1e10 for c in cond_nums])

        ax.semilogy(iterations, cond_nums_plot,
                    'o-', markersize=3, linewidth=2)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('κ(H) = λ_max / λ_min', fontsize=12)
        
        from matplotlib.ticker import LogFormatter
        ax.yaxis.set_major_formatter(LogFormatter())
        
        ax.set_title('Condition Number Evolution',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')

        if created_fig:
            return fig, ax
        else:
            return ax


if __name__ == "__main__":
    print("Comparing conditioning effects...")

    results = ConditioningEffects.compare_conditioning_levels(
        step_size=0.05, n_steps=300
    )

    fig = ConditioningEffects.plot_conditioning_comparison(
        results, save_path="conditioning_effects.png"
    )

    print("✓ Conditioning effects analysis complete")
