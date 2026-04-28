"""3D surface visualization helpers for objective landscapes."""

import numpy as np
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class Surface3DPlotter:
    """Generate 3D surface plots using Plotly."""

    @staticmethod
    def plot_surface(f, x_range=(-5, 5), y_range=(-5, 5),
                     trajectories=None, title="3D Surface with Optimization Path",
                     resolution=100, height=800):
        """Create 3D surface plot with trajectory overlay.

        Args:
            f: scalar function
            x_range, y_range: ranges for axes
            trajectories: list of Trajectory objects (optional)
            title: plot title
            resolution: mesh resolution
            height: plot height in pixels

        Returns:
            plotly Figure object
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not installed. Install with: pip install plotly kaleido")
            return None

        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # Vectorized evaluation if possible, fallback to loop
        try:
            # Try to evaluate by passing the meshgrid directly if the function supports it
            # This requires the function to be compatible with numpy broadcasting
            Z = f(np.stack([X, Y], axis=0))
        except (ValueError, TypeError, IndexError):
            # Fallback to the robust but slower point-by-point evaluation
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

        fig = go.Figure()

        # Surface plot
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='f(x)', x=1.1),
            name='Surface',
            opacity=0.8,
            hovertemplate='x₀: %{x:.2f}<br>x₁: %{y:.2f}<br>f(x): %{z:.2f}<extra></extra>'
        ))

        if trajectories is not None:
            if not isinstance(trajectories, list):
                trajectories = [trajectories]

            colors = px.colors.qualitative.Plotly

            for idx, traj in enumerate(trajectories):
                path = traj.trajectory
                z_values = np.array(traj.diagnostics['function_values'])

                # Filter out points that are too far away or contain NaNs/Infs
                # to prevent Plotly rendering issues
                valid_mask = np.isfinite(path[:, 0]) & np.isfinite(path[:, 1]) & np.isfinite(z_values)
                
                # Further filter to keep points within a reasonable multiple of the axis ranges
                # to prevent extreme zooming
                margin = 2.0
                x_min, x_max = x_range[0] - margin, x_range[1] + margin
                y_min, y_max = y_range[0] - margin, y_range[1] + margin
                
                valid_mask &= (path[:, 0] >= x_min) & (path[:, 0] <= x_max)
                valid_mask &= (path[:, 1] >= y_min) & (path[:, 1] <= y_max)

                if not np.any(valid_mask):
                    continue

                safe_path = path[valid_mask]
                safe_z = z_values[valid_mask]

                color = colors[idx % len(colors)]

                fig.add_trace(go.Scatter3d(
                    x=safe_path[:, 0],
                    y=safe_path[:, 1],
                    z=safe_z,
                    mode='lines+markers',
                    name=traj.optimizer_name,
                    line=dict(color=color, width=4),
                    marker=dict(size=4, color=color),
                    hovertemplate='x₀: %{x:.2f}<br>x₁: %{y:.2f}<br>f(x): %{z:.2f}<extra></extra>'
                ))

                # Final point marker
                fig.add_trace(go.Scatter3d(
                    x=[safe_path[-1, 0]],
                    y=[safe_path[-1, 1]],
                    z=[safe_z[-1]],
                    mode='markers',
                    marker=dict(size=12, symbol='diamond', color=color,
                                line=dict(width=2, color='white')),
                    name=f'{traj.optimizer_name} (final)',
                    showlegend=False,
                    hovertemplate='FINAL<br>x₀: %{x:.2f}<br>x₁: %{y:.2f}<br>f(x): %{z:.2f}<extra></extra>'
                ))

        # Determine reasonable Z-axis range
        z_min, z_max = np.nanmin(Z), np.nanmax(Z)
        z_range = z_max - z_min
        if z_range == 0:
            z_range = 1.0
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(title='x₀', range=x_range),
                yaxis=dict(title='x₁', range=y_range),
                zaxis=dict(title='f(x)', range=[z_min - 0.1*z_range, z_max + 0.1*z_range]),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            height=height,
            width=1000,
            margin=dict(l=0, r=0, b=0, t=50),
            font=dict(size=12)
        )

        return fig

    @staticmethod
    def show_surface(fig):
        """Display 3D surface in browser."""
        if fig is not None:
            fig.show()

    @staticmethod
    def save_surface(fig, filename='surface.html'):
        """Save 3D surface to HTML file."""
        if fig is not None:
            fig.write_html(filename)
            print(f"Saved to {filename}")
