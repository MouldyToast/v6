"""
Diffusion V7 Evaluation Module

Contains metrics and visualization for trajectory evaluation.
"""

from .metrics import (
    compute_endpoint_from_trajectory,
    compute_goal_distance_error,
    compute_goal_angle_error,
    compute_realism_metrics,
    compute_diversity_metrics,
    evaluate_trajectories,
    print_metrics_report,
    angular_distance,
)

from .visualize import (
    plot_trajectory,
    plot_multiple_trajectories,
    plot_trajectory_comparison,
    plot_feature_distributions,
    plot_diversity_by_condition,
    plot_goal_accuracy_scatter,
    create_evaluation_report,
)

__all__ = [
    # Metrics
    'compute_endpoint_from_trajectory',
    'compute_goal_distance_error',
    'compute_goal_angle_error',
    'compute_realism_metrics',
    'compute_diversity_metrics',
    'evaluate_trajectories',
    'print_metrics_report',
    'angular_distance',
    # Visualization
    'plot_trajectory',
    'plot_multiple_trajectories',
    'plot_trajectory_comparison',
    'plot_feature_distributions',
    'plot_diversity_by_condition',
    'plot_goal_accuracy_scatter',
    'create_evaluation_report',
]
