#!/usr/bin/env python3
"""
Optimization Dynamics Lab - Comprehensive Demo

Demonstrates:
1. Basic optimization on different function types
2. Failure modes
3. Conditioning effects
4. Multi-start analysis
5. Constrained optimization
"""

from experiments.conditioning_effects import ConditioningEffects
from experiments.compare_optimizers import MultiStartAnalyzer
from dynamics.constrained import (
    LagrangeMultiplierMethod,
    PenaltyMethod,
    create_circle_constraint_problem,
    create_ellipse_constraint_problem
)
from dynamics.trajectory import TrajectoryRunner
from optimizers.gradient_descent_line_search import GradientDescentWithLineSearch
from optimizers import GradientDescent, MomentumGD, Newton, SGD
from functions import Quadratic, IllConditioned, Saddle, NonConvex
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def demo_basic_optimization():
    """Demo 1: Basic optimization on different function types."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Optimization on Different Functions")
    print("="*70)

    functions = [
        ("Quadratic", Quadratic()),
        ("Ill-Conditioned", IllConditioned(a=1, b=100)),
        ("Saddle Point", Saddle()),
        ("Non-Convex", NonConvex()),
    ]

    x0 = [3.0, -4.0]
    optimizers = [
        GradientDescent(step_size=0.1),
        MomentumGD(step_size=0.1, beta=0.9),
        Newton(),
        SGD(step_size=0.01, batch_size=10)
    ]

    for func_name, function in functions:
        print(f"\nOptimizing {func_name}...")

        try:
            trajectories = TrajectoryRunner.run_comparison(
                optimizers, function, x0, steps=200, tol=1e-6
            )

            for traj in trajectories:
                print(f"  {traj.optimizer_name:20s}: {traj.n_steps:3d} steps, "
                      f"f(x) = {traj.f_final:10.6f}, ||∇f|| = {traj.grad_norm_final:.2e}")
        except Exception as e:
            print(f"  Error: {e}")


def demo_line_search():
    """Demo 2: Comparing fixed step size vs. line search."""
    print("\n" + "="*70)
    print("DEMO 2: Fixed Step Size vs. Line Search")
    print("="*70)

    function = IllConditioned(a=1, b=100)
    x0 = [4.0, -3.0]

    print(f"\nOptimizing ill-conditioned function...")

    # Fixed step
    gd_fixed = GradientDescent(step_size=0.05)

    # Line search variants
    gd_backtrack = GradientDescentWithLineSearch(
        line_search_method='backtracking', initial_step=1.0
    )
    gd_golden = GradientDescentWithLineSearch(
        line_search_method='golden_section', initial_step=1.0
    )

    optimizers = [gd_fixed, gd_backtrack, gd_golden]
    trajectories = TrajectoryRunner.run_comparison(
        optimizers, function, x0, steps=300, tol=1e-8
    )

    trajectories[0].optimizer_name = "GD (fixed η=0.05)"
    trajectories[1].optimizer_name = "GD (backtracking)"
    trajectories[2].optimizer_name = "GD (golden section)"

    for traj in trajectories:
        print(f"  {traj.optimizer_name:25s}: {traj.n_steps:3d} steps, "
              f"f(x) = {traj.f_final:.6f}")


def demo_failure_modes():
    """Demo 3: Visualize failure modes."""
    print("\n" + "="*70)
    print("DEMO 3: Failure Modes Analysis")
    print("="*70)

    print("\nDemonstrating optimization failure modes:")
    print("  1. Oscillation in narrow valley")
    print("  2. Newton failing at saddle point")
    print("  3. Divergence with large step size")
    print("  4. Momentum overshoot")
    print("  5. Local minima traps")

    # Just print info since visualization creation takes time
    print("\nTo visualize failure modes, run:")
    print("  python -c \"from experiments.failure_modes import FailureModes; ")
    print("             FailureModes.create_all_failure_demos()\"")


def demo_multistart():
    """Demo 4: Multi-start analysis."""
    print("\n" + "="*70)
    print("DEMO 4: Multi-Start Analysis")
    print("="*70)

    function = NonConvex()

    print("\nRunning gradient descent from 16 random starting points...")

    gd = GradientDescent(step_size=0.1)

    starts = MultiStartAnalyzer.generate_random_starts(
        x_range=(-3, 3), y_range=(-2, 4), n_points=16, seed=42
    )

    trajectories = MultiStartAnalyzer.run_multistart(
        gd, function, starts, steps=200, tol=1e-6
    )

    # Analyze clusters
    clusters = MultiStartAnalyzer.cluster_minima(trajectories)

    print(f"\nFound {len(clusters)} distinct minima regions:")
    for cluster_id, indices in clusters.items():
        final_points = np.array([trajectories[i].x_final for i in indices])
        center = np.mean(final_points, axis=0)
        f_center = function.f(center)

        print(f"  Cluster {cluster_id}: {len(indices)} trajectories, "
              f"center=({center[0]:.2f}, {center[1]:.2f}), f={f_center:.4f}")


def demo_conditioning():
    """Demo 5: Conditioning effects."""
    print("\n" + "="*70)
    print("DEMO 5: Conditioning Effects on Convergence")
    print("="*70)

    print("\nComparing convergence on quadratics with different condition numbers:")

    results = ConditioningEffects.compare_conditioning_levels(
        step_size=0.05, n_steps=200
    )

    for cond_name, result_data in results.items():
        trajs = result_data['trajectories']
        print(f"\n{cond_name}:")
        for traj in trajs:
            print(f"  {traj.optimizer_name:15s}: {traj.n_steps:3d} steps")


def demo_constrained_optimization():
    """Demo 6: Constrained optimization."""
    print("\n" + "="*70)
    print("DEMO 6: Constrained Optimization (Lagrange Multipliers)")
    print("="*70)

    # Problem 1: Circle constraint
    problem = create_circle_constraint_problem()
    print(f"\n{problem.name}")
    print("  Minimize x² + y² subject to x + y = 2")
    print("  Expected solution: x = y = 1, f = 2")

    lagrange = LagrangeMultiplierMethod(step_size=0.01)
    x0 = np.array([1.5, 1.5])

    try:
        x_opt, lambda_opt, history = lagrange.optimize(
            problem.f, problem.g, x0, lambda0=1.0, steps=100
        )

        print(f"\n  Solution found:")
        print(f"    x = ({x_opt[0]:.4f}, {x_opt[1]:.4f})")
        print(f"    f(x) = {problem.f(x_opt):.6f}")
        print(f"    g(x) = {problem.g(x_opt):.6e} (should be ≈ 0)")
        print(f"    λ = {lambda_opt:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Problem 2: Ellipse constraint
    problem2 = create_ellipse_constraint_problem()
    print(f"\n{problem2.name}")
    print("  Find point on ellipse x²/4 + y² = 1 closest to (2, 0)")

    # Use penalty method
    x0_2 = np.array([1.5, 0.5])

    try:
        penalty = PenaltyMethod(step_size=0.01)
        x_opt, rho_final, history = penalty.optimize(
            problem2.f, problem2.g, x0_2, rho_init=1.0, steps=100
        )

        print(f"\n  Solution found:")
        print(f"    x = ({x_opt[0]:.4f}, {x_opt[1]:.4f})")
        print(f"    f(x) = {problem2.f(x_opt):.6f}")
        print(f"    g(x) = {problem2.g(x_opt):.6e} (should be ≈ 0)")
    except Exception as e:
        print(f"  Error: {e}")


def demo_diagnostics():
    """Demo 7: Detailed diagnostics."""
    print("\n" + "="*70)
    print("DEMO 7: Detailed Diagnostics")
    print("="*70)

    function = IllConditioned(a=1, b=100)
    gd = GradientDescent(step_size=0.05)
    x0 = [3.0, -4.0]

    print("\nRunning GD on ill-conditioned quadratic...")
    trajectory = TrajectoryRunner.run(gd, function, x0, steps=100, tol=1e-6)

    print("\nInitial diagnostics:")
    print(f"  x₀ = {trajectory.trajectory[0]}")
    print(f"  f(x₀) = {trajectory.f_initial:.6f}")
    print(f"  ||∇f(x₀)|| = {trajectory.diagnostics['gradient_norms'][0]:.2e}")

    print("\nFinal diagnostics:")
    print(f"  x_final = {trajectory.x_final}")
    print(f"  f(x_final) = {trajectory.f_final:.6f}")
    print(f"  ||∇f(x_final)|| = {trajectory.grad_norm_final:.2e}")

    eigenvalues_final = trajectory.diagnostics['eigenvalues'][-1]
    cond_num_final = trajectory.diagnostics['condition_numbers'][-1]

    print(f"\n  Hessian eigenvalues (final): {eigenvalues_final}")
    print(f"  Condition number (final): {cond_num_final:.2e}")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "Optimization Dynamics Laboratory" + " "*21 + "║")
    print("║" + " "*12 + "Comprehensive Demonstration Script" + " "*24 + "║")
    print("╚" + "="*68 + "╝")

    try:
        demo_basic_optimization()
        demo_line_search()
        demo_failure_modes()
        demo_multistart()
        demo_conditioning()
        demo_constrained_optimization()
        demo_diagnostics()

        print("\n" + "="*70)
        print("✓ All demos completed successfully!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run the interactive Streamlit app:")
        print("     streamlit run app.py")
        print("\n  2. Generate failure mode visualizations:")
        print("     python -c \"from experiments.failure_modes import FailureModes;")
        print("                FailureModes.create_all_failure_demos()\"")
        print()

    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
