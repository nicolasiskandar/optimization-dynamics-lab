"""Streamlit application for interactive optimization-dynamics exploration."""

from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Disable mathtext for axes formatters to avoid \mathdefault errors in some environments
matplotlib.rcParams['axes.formatter.use_mathtext'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'

from dynamics.constrained import (
    LagrangeMultiplierMethod,
    PenaltyMethod,
    create_circle_constraint_problem,
    create_ellipse_constraint_problem,
)
from dynamics.trajectory import TrajectoryRunner
from experiments.compare_optimizers import MultiStartAnalyzer
from experiments.conditioning_effects import ConditioningEffects
from experiments.initialization_sensitivity import InitializationSensitivity
from experiments.failure_modes import FailureModes
from functions import IllConditioned, NonConvex, Quadratic, Saddle
from optimizers import (
    GradientDescent,
    GradientDescentWithLineSearch,
    MomentumGD,
    MomentumWithLineSearch,
    Newton,
)
from visualization.contours import ContourPlotter
from visualization.trajectories import VisualizationRunner


st.set_page_config(
    page_title="Optimization Dynamics Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Optimization Dynamics Laboratory")
st.markdown(
    """
Visualize optimization algorithms, constrained solvers, and experiment modules
from the codebase in one place.
"""
)


def build_function(name):
    """Return the selected 2D objective."""
    function_map = {
        "Quadratic": Quadratic(),
        "Ill-Conditioned Quadratic": IllConditioned(a=1, b=100),
        "Saddle Point": Saddle(),
        "Non-Convex": NonConvex(),
    }
    return function_map[name]


def build_unconstrained_optimizers(
    use_gd,
    use_momentum,
    use_newton,
    use_gd_line_search,
    use_momentum_line_search,
    step_size,
    beta,
    line_search_method,
    initial_step,
):
    """Create optimizer instances selected in the sidebar."""
    optimizers = []
    names = []

    if use_gd:
        optimizers.append(GradientDescent(step_size=step_size))
        names.append("Gradient Descent")

    if use_momentum:
        optimizers.append(MomentumGD(step_size=step_size, beta=beta))
        names.append(f"Momentum (β={beta})")

    if use_newton:
        optimizers.append(Newton())
        names.append("Newton's Method")

    if use_gd_line_search:
        optimizers.append(
            GradientDescentWithLineSearch(
                line_search_method=line_search_method,
                initial_step=initial_step,
            )
        )
        names.append(f"GD + LS ({line_search_method})")

    if use_momentum_line_search:
        optimizers.append(
            MomentumWithLineSearch(
                beta=beta,
                line_search_method=line_search_method,
                initial_step=initial_step,
            )
        )
        names.append(f"Momentum + LS ({line_search_method}, β={beta})")

    return optimizers, names


def plot_constrained_problem(problem, x_history, x_range, y_range, title):
    """Plot objective contours, constraint curve, and the optimization path."""
    X, Y = ContourPlotter.create_mesh(x_range, y_range, resolution=200)
    Z = ContourPlotter.evaluate_function(problem.f, X, Y)
    G = ContourPlotter.evaluate_function(problem.g, X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax0 = axes[0]
    contourf = ax0.contourf(X, Y, Z, levels=25, cmap="viridis", alpha=0.85)
    ax0.contour(X, Y, G, levels=[0], colors="tomato", linewidths=2.5)
    ax0.plot(x_history[:, 0], x_history[:, 1], "o-", color="white", linewidth=2, markersize=4)
    ax0.plot(x_history[0, 0], x_history[0, 1], "o", color="deepskyblue", markersize=8, markeredgecolor="black")
    ax0.plot(x_history[-1, 0], x_history[-1, 1], "s", color="gold", markersize=8, markeredgecolor="black")
    plt.colorbar(contourf, ax=ax0, label="f(x)")
    ax0.set_title("Objective + Constraint", fontweight="bold")
    ax0.set_xlabel("x₀")
    ax0.set_ylabel("x₁")
    ax0.set_xlim(x_range)
    ax0.set_ylim(y_range)
    ax0.set_aspect("equal")
    ax0.grid(True, alpha=0.2)

    ax1 = axes[1]
    iterations = np.arange(len(x_history))
    f_vals = np.array([problem.f(x) for x in x_history])
    g_vals = np.array([abs(problem.g(x)) for x in x_history])
    ax1.semilogy(iterations, np.maximum(f_vals, 1e-12), "o-", label="f(x)", linewidth=2)
    ax1.semilogy(iterations, np.maximum(g_vals, 1e-12), "s-", label="|g(x)|", linewidth=2)
    ax1.set_title("Convergence", fontweight="bold")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Value [log scale]")
    
    from matplotlib.ticker import LogFormatter
    ax1.yaxis.set_major_formatter(LogFormatter())
    
    ax1.legend()
    ax1.grid(True, alpha=0.3, which="both")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    return fig


def render_trajectory_dashboard(trajectories, optimizer_names, selected_function, x_range, y_range):
    """Render comparison plots, diagnostics, and exports for unconstrained runs."""
    fig = VisualizationRunner.create_comparison_figure(
        selected_function.f, trajectories, x_range=x_range, y_range=y_range
    )
    st.pyplot(fig, use_container_width=True)

    st.header("Diagnostics")
    cols = st.columns(len(trajectories))

    for col, traj, opt_name in zip(cols, trajectories, optimizer_names):
        with col:
            st.subheader(opt_name)
            left, right = st.columns(2)
            with left:
                st.metric("Steps", traj.n_steps)
                st.metric("f(x₀)", f"{traj.f_initial:.4f}")
                st.metric("||∇f(x₀)||", f"{traj.diagnostics['gradient_norms'][0]:.2e}")
            with right:
                st.metric("f(xf)", f"{traj.f_final:.4f}", delta=f"{traj.f_final - traj.f_initial:.4f}")
                st.metric("||∇f(xf)||", f"{traj.grad_norm_final:.2e}")
                cond_num_final = traj.diagnostics["condition_numbers"][-1]
                st.metric("κ(Hf)", f"{cond_num_final:.2e}" if np.isfinite(cond_num_final) else "∞")

            st.write("**Hessian eigenvalues (final):**")
            st.write(traj.diagnostics["eigenvalues"][-1])

    with st.expander("Iteration-by-Iteration Analysis", expanded=False):
        selected_traj_idx = st.selectbox(
            "Select trajectory to analyze:",
            range(len(trajectories)),
            format_func=lambda i: optimizer_names[i],
        )
        selected_traj = trajectories[selected_traj_idx]

        diag_data = {
            "Iteration": range(len(selected_traj.trajectory)),
            "x₀": selected_traj.trajectory[:, 0],
            "x₁": selected_traj.trajectory[:, 1],
            "f(x)": selected_traj.diagnostics["function_values"],
            "||∇f(x)||": selected_traj.diagnostics["gradient_norms"],
            "κ(H)": selected_traj.diagnostics["condition_numbers"],
        }
        st.dataframe(pd.DataFrame(diag_data), use_container_width=True, height=380)

        st.subheader("Hessian Eigenvalue Evolution")
        eigenvalues_over_time = selected_traj.diagnostics["eigenvalues"]
        fig_eigvals, ax = plt.subplots(figsize=(10, 4))
        for dim in range(2):
            eigvals_dim = [ev[dim] if len(ev) > dim else np.nan for ev in eigenvalues_over_time]
            ax.semilogy(range(len(eigenvalues_over_time)), np.abs(eigvals_dim), "o-", markersize=3, label=f"λ_{dim + 1}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("|λᵢ|")
        
        from matplotlib.ticker import LogFormatter
        ax.yaxis.set_major_formatter(LogFormatter())
        
        ax.set_title(f"Hessian Eigenvalues - {optimizer_names[selected_traj_idx]}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_eigvals, use_container_width=True)

    with st.expander("Optimizer Comparison", expanded=False):
        comparison_data = {
            "Optimizer": optimizer_names,
            "Steps": [traj.n_steps for traj in trajectories],
            "f(x₀)": [traj.f_initial for traj in trajectories],
            "f(xf)": [traj.f_final for traj in trajectories],
            "Δf": [traj.f_final - traj.f_initial for traj in trajectories],
            "||∇f(xf)||": [traj.grad_norm_final for traj in trajectories],
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    with st.expander("3D Surface Visualization", expanded=False):
        st.info("3D visualization requires Plotly.")
        try:
            fig_3d = VisualizationRunner.plot_3d_surface(
                selected_function.f,
                trajectories=trajectories,
                x_range=x_range,
                y_range=y_range,
                show=False,
            )
            if fig_3d is not None:
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("3D visualization not available in this environment.")
        except Exception as exc:
            st.error(f"Error creating 3D plot: {exc}")

    with st.expander("Export Results", expanded=False):
        all_results = []
        for traj, opt_name in zip(trajectories, optimizer_names):
            for it, (pos, f_val) in enumerate(zip(traj.trajectory, traj.diagnostics["function_values"])):
                all_results.append(
                    {
                        "Optimizer": opt_name,
                        "Iteration": it,
                        "x0": pos[0],
                        "x1": pos[1],
                        "f(x)": f_val,
                    }
                )

        df_export = pd.DataFrame(all_results)
        csv = df_export.to_csv(index=False)
        st.download_button(
            "Download as CSV",
            csv,
            f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
        )


def render_unconstrained_workspace():
    """Render the unconstrained optimization workspace."""
    st.sidebar.header("Optimization Controls")

    function_name = st.sidebar.selectbox(
        "Function to optimize",
        ["Quadratic", "Ill-Conditioned Quadratic", "Saddle Point", "Non-Convex"],
    )
    selected_function = build_function(function_name)

    st.sidebar.subheader("Axis Ranges")
    x_range = st.sidebar.slider("x₀ range", -10.0, 10.0, (-5.0, 5.0), step=0.5)
    y_range = st.sidebar.slider("x₁ range", -10.0, 10.0, (-5.0, 5.0), step=0.5)

    st.sidebar.subheader("Optimizer Settings")
    use_gd = st.sidebar.checkbox("Gradient Descent", value=True)
    use_momentum = st.sidebar.checkbox("Momentum GD", value=True)
    use_newton = st.sidebar.checkbox("Newton's Method", value=False, help="Newton can diverge on some functions")
    use_gd_line_search = st.sidebar.checkbox("GD + Line Search", value=False)
    use_momentum_line_search = st.sidebar.checkbox("Momentum + Line Search", value=False)

    if use_gd or use_momentum:
        step_size = st.sidebar.slider(
            "Fixed step size",
            0.001,
            0.5,
            0.1,
            step=0.01,
            help="Step size for fixed-step gradient-based methods",
        )
    else:
        step_size = 0.1

    if use_momentum or use_momentum_line_search:
        beta = st.sidebar.slider(
            "Momentum (β)",
            0.0,
            0.99,
            0.9,
            step=0.05,
            help="Higher values add more historical gradient",
        )
    else:
        beta = 0.9

    if use_gd_line_search or use_momentum_line_search:
        line_search_method = st.sidebar.selectbox(
            "Line search method",
            ["backtracking", "golden_section"],
        )
        initial_step = st.sidebar.slider(
            "Line-search initial step",
            0.1,
            5.0,
            1.0,
            step=0.1,
        )
    else:
        line_search_method = "backtracking"
        initial_step = 1.0

    n_steps = st.sidebar.slider("Max steps", 10, 500, 200, step=10)
    tol = st.sidebar.slider("Convergence tolerance", 1e-8, 1e-3, 1e-6, format="%.0e")

    st.sidebar.subheader("Initial Point")
    x0_mode = st.sidebar.radio("Starting point", ["Manual", "Random"])
    if x0_mode == "Manual":
        x0_0 = st.sidebar.slider("x₀[0]", x_range[0], x_range[1], 3.0, step=0.1)
        x0_1 = st.sidebar.slider("x₀[1]", y_range[0], y_range[1], -4.0, step=0.1)
        x0 = [x0_0, x0_1]
    else:
        x0 = [
            np.random.uniform(x_range[0], x_range[1]),
            np.random.uniform(y_range[0], y_range[1]),
        ]
        st.sidebar.write(f"Random start: ({x0[0]:.2f}, {x0[1]:.2f})")

    optimizers, optimizer_names = build_unconstrained_optimizers(
        use_gd,
        use_momentum,
        use_newton,
        use_gd_line_search,
        use_momentum_line_search,
        step_size,
        beta,
        line_search_method,
        initial_step,
    )

    if not optimizers:
        st.warning("Select at least one optimizer.")
        return

    with st.spinner("Running optimization..."):
        trajectories = TrajectoryRunner.run_comparison(
            optimizers,
            selected_function,
            x0,
            steps=n_steps,
            tol=tol,
        )

    render_trajectory_dashboard(trajectories, optimizer_names, selected_function, x_range, y_range)


def render_constrained_workspace():
    """Render the constrained optimization workspace."""
    st.sidebar.header("Constrained Controls")

    problem_name = st.sidebar.selectbox(
        "Problem",
        ["Linear constraint on circle objective", "Ellipse constraint"],
    )
    if problem_name == "Linear constraint on circle objective":
        problem = create_circle_constraint_problem()
        x0_default = [1.5, 1.5]
        x_range = (-1.0, 3.0)
        y_range = (-1.0, 3.0)
    else:
        problem = create_ellipse_constraint_problem()
        x0_default = [1.5, 0.5]
        x_range = (-3.0, 3.0)
        y_range = (-2.5, 2.5)

    method = st.sidebar.selectbox("Method", ["Lagrange Multipliers", "Penalty Method"])
    step_size = st.sidebar.slider("Primal step size", 0.001, 0.2, 0.01, step=0.001, format="%.3f")
    steps = st.sidebar.slider("Max steps", 10, 500, 150, step=10)
    tol = st.sidebar.slider("Tolerance", 1e-8, 1e-3, 1e-6, format="%.0e")

    x0_0 = st.sidebar.slider("x₀[0]", x_range[0], x_range[1], float(x0_default[0]), step=0.1)
    x0_1 = st.sidebar.slider("x₀[1]", y_range[0], y_range[1], float(x0_default[1]), step=0.1)
    x0 = np.array([x0_0, x0_1], dtype=float)

    if method == "Lagrange Multipliers":
        lambda0 = st.sidebar.slider("Initial λ", -5.0, 5.0, 1.0, step=0.1)
        solver = LagrangeMultiplierMethod(step_size=step_size)
        with st.spinner("Running Lagrange multiplier method..."):
            x_opt, lambda_opt, history = solver.optimize(
                problem.f,
                problem.g,
                x0,
                lambda0=lambda0,
                steps=steps,
                tol=tol,
            )
        x_history, dual_history = history
        dual_name = "λ"
        dual_label = "Multiplier"
        dual_final = lambda_opt
    else:
        rho_init = st.sidebar.slider("Initial ρ", 0.1, 10.0, 1.0, step=0.1)
        solver = PenaltyMethod(step_size=step_size)
        with st.spinner("Running penalty method..."):
            x_opt, rho_final, history = solver.optimize(
                problem.f,
                problem.g,
                x0,
                rho_init=rho_init,
                steps=steps,
                tol=tol,
            )
        x_history, dual_history = history
        dual_name = "ρ"
        dual_label = "Penalty"
        dual_final = rho_final

    figure = plot_constrained_problem(
        problem,
        x_history,
        x_range,
        y_range,
        f"{problem.name} - {method}",
    )
    st.pyplot(figure, use_container_width=True)

    final_constraint = problem.g(x_opt)
    final_objective = problem.f(x_opt)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("x*", f"({x_opt[0]:.4f}, {x_opt[1]:.4f})")
    m2.metric("f(x*)", f"{final_objective:.6f}")
    m3.metric("|g(x*)|", f"{abs(final_constraint):.2e}")
    m4.metric(dual_label, f"{dual_final:.4f}")

    summary_df = pd.DataFrame(
        [
            {"Metric": "Iterations", "Value": len(x_history)},
            {"Metric": "Objective", "Value": final_objective},
            {"Metric": "Constraint violation", "Value": abs(final_constraint)},
            {"Metric": dual_name, "Value": dual_final},
            {"Metric": "Feasible", "Value": problem.is_feasible(x_opt)},
        ]
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with st.expander("Trajectory details", expanded=False):
        history_df = pd.DataFrame(
            {
                "Iteration": range(len(x_history)),
                "x₀": x_history[:, 0],
                "x₁": x_history[:, 1],
                "f(x)": [problem.f(x) for x in x_history],
                "|g(x)|": [abs(problem.g(x)) for x in x_history],
                dual_name: dual_history,
            }
        )
        st.dataframe(history_df, use_container_width=True, height=360)


def render_experiments_workspace():
    """Render the experiment and analysis workspace."""
    st.sidebar.header("Experiment Controls")
    experiment = st.sidebar.selectbox(
        "Experiment",
        [
            "Failure Modes",
            "Conditioning Effects",
            "Multi-Start Analysis",
            "Initialization Sensitivity",
        ],
    )

    if experiment == "Failure Modes":
        demo = st.sidebar.selectbox(
            "Demo",
            [
                "Oscillation in narrow valley",
                "Newton at saddle point",
                "GD stuck on plateau",
                "Large step divergence",
                "Momentum overshoot",
                "Non-convex local minima",
            ],
        )

        demo_map = {
            "Oscillation in narrow valley": FailureModes.oscillation_in_narrow_valley,
            "Newton at saddle point": FailureModes.newton_at_saddle,
            "GD stuck on plateau": FailureModes.gd_stuck_on_plateau,
            "Large step divergence": FailureModes.large_step_divergence,
            "Momentum overshoot": FailureModes.momentum_overshoot,
            "Non-convex local minima": FailureModes.nonconvex_local_minima,
        }

        fig, result = demo_map[demo]()
        st.pyplot(fig, use_container_width=True)

        if isinstance(result, list):
            comparison = pd.DataFrame(
                {
                    "Trajectory": [traj.optimizer_name for traj in result],
                    "Steps": [traj.n_steps for traj in result],
                    "f(xf)": [traj.f_final for traj in result],
                    "||∇f(xf)||": [traj.grad_norm_final for traj in result],
                }
            )
            st.dataframe(comparison, use_container_width=True, hide_index=True)
        else:
            st.write(result.summary())

    elif experiment == "Conditioning Effects":
        step_size = st.sidebar.slider("Step size", 0.001, 0.2, 0.05, step=0.001, format="%.3f")
        n_steps = st.sidebar.slider("Max steps", 50, 500, 200, step=10)

        results = ConditioningEffects.compare_conditioning_levels(
            step_size=step_size,
            n_steps=n_steps,
        )
        fig = ConditioningEffects.plot_conditioning_comparison(results)
        st.pyplot(fig, use_container_width=True)

        rows = []
        for cond_name, data in results.items():
            for traj in data["trajectories"]:
                rows.append(
                    {
                        "Conditioning": cond_name,
                        "Optimizer": traj.optimizer_name,
                        "Steps": traj.n_steps,
                        "f(xf)": traj.f_final,
                        "||∇f(xf)||": traj.grad_norm_final,
                    }
                )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    elif experiment == "Multi-Start Analysis":
        optimizer_name = st.sidebar.selectbox(
            "Optimizer",
            ["Gradient Descent", "Momentum GD", "GD + Line Search"],
        )
        n_points = st.sidebar.slider("Starting points", 4, 64, 16, step=1)
        x_range = st.sidebar.slider("x₀ range", -5.0, 5.0, (-3.0, 3.0), step=0.5)
        y_range = st.sidebar.slider("x₁ range", -5.0, 5.0, (-2.0, 4.0), step=0.5)
        n_steps = st.sidebar.slider("Max steps", 50, 500, 200, step=10)
        tol = st.sidebar.slider("Tolerance", 1e-8, 1e-3, 1e-6, format="%.0e")
        seed = st.sidebar.number_input("Seed", value=42, step=1)
        color_by_cluster = st.sidebar.checkbox("Color by cluster", value=True)

        optimizer_map = {
            "Gradient Descent": GradientDescent(step_size=0.1),
            "Momentum GD": MomentumGD(step_size=0.1, beta=0.9),
            "GD + Line Search": GradientDescentWithLineSearch(
                line_search_method="backtracking",
                initial_step=1.0,
            ),
        }

        function = NonConvex()
        starts = MultiStartAnalyzer.generate_random_starts(
            x_range=x_range,
            y_range=y_range,
            n_points=n_points,
            seed=seed,
        )

        with st.spinner("Running multi-start analysis..."):
            trajectories = MultiStartAnalyzer.run_multistart(
                optimizer_map[optimizer_name],
                function,
                starts,
                steps=n_steps,
                tol=tol,
            )

        fig, _ = MultiStartAnalyzer.plot_multistart_trajectories(
            function.f,
            trajectories,
            x_range=x_range,
            y_range=y_range,
            title=f"Multi-Start Analysis - {optimizer_name}",
            color_by_cluster=color_by_cluster,
        )
        st.pyplot(fig, use_container_width=True)

        cluster_map = MultiStartAnalyzer.cluster_minima(trajectories)
        st.metric("Clusters found", len(cluster_map))

        rows = []
        for idx, traj in enumerate(trajectories):
            rows.append(
                {
                    "Run": idx,
                    "x0": traj.x0[0],
                    "x1": traj.x0[1],
                    "xf0": traj.x_final[0],
                    "xf1": traj.x_final[1],
                    "f(xf)": traj.f_final,
                    "Steps": traj.n_steps,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    elif experiment == "Initialization Sensitivity":
        n_starts = st.sidebar.slider("Starting points", 4, 32, 16, step=1)

        with st.spinner("Running initialization sensitivity analysis..."):
            results = InitializationSensitivity.analyze_nonconvex_function(n_starts=n_starts, save_dir=None)

        st.pyplot(results["figure"], use_container_width=True)

        summary = pd.DataFrame(
            {
                "Metric": [
                    "GD mean final loss",
                    "Momentum mean final loss",
                    "Shared mean start distance",
                ],
                "Value": [
                    float(np.mean(results["final_losses_gd"])),
                    float(np.mean(results["final_losses_momentum"])),
                    float(np.mean(results["init_distances"])),
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)


st.sidebar.markdown("---")
workspace = st.sidebar.radio(
    "Workspace",
    ["Optimization", "Constrained", "Experiments"],
)
st.sidebar.write("**Optimization Dynamics Lab**")

if workspace == "Optimization":
    render_unconstrained_workspace()
elif workspace == "Constrained":
    render_constrained_workspace()
else:
    render_experiments_workspace()
