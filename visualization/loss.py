"""Convergence-metric plotting helpers."""

import numpy as np
import matplotlib.pyplot as plt


class LossPlotter:
    """Plot optimization metrics vs iteration."""

    @staticmethod
    def _to_positive_for_log(values, eps=1e-12):
        """Shift a series to strictly positive values for log plotting."""
        values = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            return values, False

        finite_values = values[finite_mask]
        min_val = np.min(finite_values)
        shifted = False

        if min_val <= 0:
            shift = -min_val + eps
            values = values + shift
            shifted = True

        values[finite_mask] = np.maximum(values[finite_mask], eps)
        return values, shifted

    @staticmethod
    def plot_loss_curves(trajectories, ax=None, title="Loss vs Iteration"):
        """Plot function value vs iteration for multiple trajectories.

        Args:
            trajectories: list of Trajectory objects
            ax: matplotlib axis (optional)
            title: plot title

        Returns:
            fig, ax if ax was None, else just ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            created_fig = True
        else:
            created_fig = False
            fig = ax.figure

        if not isinstance(trajectories, list):
            trajectories = [trajectories]

        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

        any_shifted = False
        for traj, color in zip(trajectories, colors):
            f_values = traj.diagnostics['function_values']
            f_values, shifted = LossPlotter._to_positive_for_log(f_values)
            any_shifted = any_shifted or shifted
            iterations = np.arange(len(f_values))
            ax.semilogy(iterations, f_values, 'o-', color=color,
                        markersize=4, linewidth=2, label=traj.optimizer_name, alpha=0.8)

        ax.set_xlabel('Iteration', fontsize=12)
        ylabel = 'f(x) [log scale]'
        if any_shifted:
            ylabel = 'f(x) [log scale, auto-shifted]'
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        from matplotlib.ticker import LogFormatterExponent
        ax.yaxis.set_major_formatter(LogFormatterExponent())
        
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')

        if created_fig:
            return fig, ax
        return ax

    @staticmethod
    def plot_gradient_norm(trajectories, ax=None, title="Gradient Norm vs Iteration"):
        """Plot ||∇f(x_t)|| vs iteration.

        Args:
            trajectories: list of Trajectory objects
            ax: matplotlib axis (optional)
            title: plot title

        Returns:
            fig, ax if ax was None, else just ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            created_fig = True
        else:
            created_fig = False
            fig = ax.figure

        if not isinstance(trajectories, list):
            trajectories = [trajectories]

        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

        for traj, color in zip(trajectories, colors):
            grad_norms = traj.diagnostics['gradient_norms']
            grad_norms, _ = LossPlotter._to_positive_for_log(grad_norms)
            iterations = np.arange(len(grad_norms))
            ax.semilogy(iterations, grad_norms, 'o-', color=color,
                        markersize=4, linewidth=2, label=traj.optimizer_name, alpha=0.8)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('||∇f(x)|| [log scale]', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        from matplotlib.ticker import LogFormatterExponent
        ax.yaxis.set_major_formatter(LogFormatterExponent())
        
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')

        if created_fig:
            return fig, ax
        return ax
