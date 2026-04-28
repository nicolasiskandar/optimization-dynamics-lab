"""High-level visualization workflows combining multiple plot types."""

import matplotlib.pyplot as plt
from visualization.contours import ContourPlotter
from visualization.loss import LossPlotter
from visualization.vector_fields import VectorFieldPlotter
from visualization.surfaces import Surface3DPlotter


class VisualizationRunner:
    """Orchestrate comprehensive visualization of optimization dynamics."""

    @staticmethod
    def plot_contour_with_trajectories(f, trajectories, x_range=(-5, 5),
                                       y_range=(-5, 5), title="Optimization Trajectories",
                                       save_path=None):
        """Create contour plot with trajectory overlays.

        Args:
            f: scalar function
            trajectories: list of Trajectory objects
            x_range, y_range: axis ranges
            title: plot title
            save_path: optional path to save figure

        Returns:
            fig, ax
        """
        fig, ax = ContourPlotter.plot_contour(
            f, x_range=x_range, y_range=y_range,
            trajectories=trajectories, title=title
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved contour plot to {save_path}")

        return fig, ax

    @staticmethod
    def plot_vector_field_with_trajectories(f, trajectories, x_range=(-5, 5),
                                            y_range=(-5, 5), title="Gradient Field",
                                            arrows_per_side=12, save_path=None):
        """Create vector field plot with trajectory overlays.

        Args:
            f: scalar function
            trajectories: list of Trajectory objects
            x_range, y_range: axis ranges
            title: plot title
            arrows_per_side: density of arrows
            save_path: optional path to save figure

        Returns:
            fig, ax
        """
        fig, ax = VectorFieldPlotter.plot_gradient_field(
            f, x_range=x_range, y_range=y_range,
            arrows_per_side=arrows_per_side, title=title,
            trajectories=trajectories
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved vector field plot to {save_path}")

        return fig, ax

    @staticmethod
    def plot_metrics(trajectories, save_dir=None):
        """Create plots for convergence metrics.

        Args:
            trajectories: list of Trajectory objects
            save_dir: optional directory to save figures

        Returns:
            dict of figures
        """
        figures = {}

        fig_loss, ax_loss = LossPlotter.plot_loss_curves(trajectories)
        figures['loss'] = (fig_loss, ax_loss)
        if save_dir:
            fig_loss.savefig(f"{save_dir}/loss_curve.png",
                             dpi=300, bbox_inches='tight')

        fig_grad, ax_grad = LossPlotter.plot_gradient_norm(trajectories)
        figures['gradient'] = (fig_grad, ax_grad)
        if save_dir:
            fig_grad.savefig(f"{save_dir}/gradient_norm.png",
                             dpi=300, bbox_inches='tight')

        return figures

    @staticmethod
    def plot_3d_surface(f, trajectories=None, x_range=(-5, 5), y_range=(-5, 5),
                        title="3D Surface", save_path=None, show=True):
        """Create 3D surface plot.

        Args:
            f: scalar function
            trajectories: list of Trajectory objects (optional)
            x_range, y_range: axis ranges
            title: plot title
            save_path: optional path to save HTML
            show: whether to display in browser

        Returns:
            plotly Figure
        """
        fig = Surface3DPlotter.plot_surface(
            f, x_range=x_range, y_range=y_range,
            trajectories=trajectories, title=title
        )

        if fig is None:
            print("3D plotting requires plotly. Skipping 3D visualization.")
            return None

        if save_path:
            Surface3DPlotter.save_surface(fig, save_path)

        if show:
            Surface3DPlotter.show_surface(fig)

        return fig

    @staticmethod
    def create_comparison_figure(f, trajectories, x_range=(-5, 5), y_range=(-5, 5),
                                 save_path=None, title=None):
        """Create a figure with contour, vector field, and metrics.

        Args:
            f: scalar function
            trajectories: list of Trajectory objects
            x_range, y_range: axis ranges
            save_path: optional path to save figure
            title: optional figure title

        Returns:
            fig with subplots
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        ContourPlotter.plot_contour(
            f, x_range=x_range, y_range=y_range,
            trajectories=trajectories, ax=axes[0, 0], title="Contour Plot"
        )

        VectorFieldPlotter.plot_gradient_field(
            f, x_range=x_range, y_range=y_range,
            ax=axes[0, 1], title="Gradient Vector Field",
            trajectories=trajectories, arrows_per_side=10
        )

        LossPlotter.plot_loss_curves(trajectories, ax=axes[1, 0])

        LossPlotter.plot_gradient_norm(trajectories, ax=axes[1, 1])

        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison figure to {save_path}")

        return fig
