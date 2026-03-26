"""
CLI Application for OpenEnv Job Scheduling Environment
Provides interactive evaluation and testing interface
"""

import sys
import json
import argparse
from job_scheduling_env import JobSchedulingEnv
from task_graders import TaskGrader
from baseline_inference import BaselineAgents, BaselineEvaluator


def print_banner(title):
    """Print formatted title banner."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def cmd_demo():
    """Run demo episode with random agent."""
    print_banner("DEMO: Random Agent Playing 20 Steps")
    
    env = JobSchedulingEnv(seed=42)
    obs = env.reset()
    
    print(f"Initial state:")
    print(f"  Queue size: {obs.queue_size}")
    print(f"  Available CPU: {obs.available_resources.available_cpu}/{env.num_cpus}")
    print(f"  Available Memory: {obs.available_resources.available_memory}/{env.num_memory}")
    print()
    
    total_reward = 0
    for step in range(20):
        action = -1 if len(obs.job_queue) == 0 else step % max(1, len(obs.job_queue))
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward
        
        print(f"Step {step + 1:2d}: Action={action:2d}, Reward={result.reward:6.3f}, "
              f"Queue={obs.queue_size:2d}, Running={len(obs.running_jobs):2d}, "
              f"CPU={obs.available_resources.available_cpu:2d}/{env.num_cpus}")
    
    metrics = env.get_metrics()
    print(f"\nEpisode complete:")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Jobs completed: {metrics['total_jobs_completed']}")
    print(f"  Avg completion time: {metrics['avg_completion_time']:.2f}")
    print(f"  Throughput: {metrics['throughput']:.4f}")


def cmd_baseline():
    """Run baseline agent comparison."""
    print_banner("BASELINE AGENT COMPARISON (5 episodes each)")
    
    agents = {
        'random': BaselineAgents.random_agent,
        'fcfs': BaselineAgents.fcfs_agent,
        'sjf': BaselineAgents.greedy_shortest,
        'priority': BaselineAgents.greedy_priority,
        'resource_aware': BaselineAgents.greedy_resource_aware,
        'balanced': BaselineAgents.balanced_agent,
    }
    
    evaluator = BaselineEvaluator(seed=42)
    results = {}
    
    for agent_name, agent_fn in agents.items():
        result = evaluator.evaluate_agent(agent_fn, agent_name, num_episodes=3)
        results[agent_name] = result
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Agent':<20} {'Jobs':<10} {'Avg Time':<12} {'Throughput':<12} {'Reward':<10}")
    print("-" * 64)
    
    for agent_name in sorted(results.keys()):
        stats = results[agent_name]['aggregate']
        print(f"{agent_name:<20} {stats['total_jobs_completed']:<10.1f} "
              f"{stats['avg_completion_time']:<12.2f} {stats['throughput']:<12.4f} "
              f"{stats['total_reward']:<10.2f}")


def cmd_task(task_id, agent_name='balanced'):
    """Evaluate agent on specific task."""
    print_banner(f"TASK EVALUATION: {task_id.upper()}")
    
    agents = {
        'random': BaselineAgents.random_agent,
        'fcfs': BaselineAgents.fcfs_agent,
        'sjf': BaselineAgents.greedy_shortest,
        'priority': BaselineAgents.greedy_priority,
        'resource_aware': BaselineAgents.greedy_resource_aware,
        'balanced': BaselineAgents.balanced_agent,
    }
    
    if agent_name not in agents:
        print(f"ERROR: Unknown agent '{agent_name}'")
        print(f"Available agents: {', '.join(agents.keys())}")
        return
    
    print(f"Evaluating agent: {agent_name}")
    print(f"Task: {task_id}")
    print()
    
    result = TaskGrader.evaluate_agent(
        agents[agent_name],
        task_id,
        num_episodes=3,
        seed=42
    )
    
    print(f"Score: {result.score:.3f}")
    print(f"Passed: {'✅ YES' if result.passed else '❌ NO'}")
    print(f"Difficulty: {result.difficulty}")
    print()
    print("Feedback:")
    print(f"  {result.feedback}")
    print()
    print("Metrics:")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


def cmd_evaluate():
    """Evaluate on all tasks."""
    print_banner("FULL EVALUATION: All Tasks")
    
    tasks = ['easy', 'medium', 'hard']
    all_results = {}
    
    for task_id in tasks:
        print(f"Evaluating {task_id.upper()} task...")
        result = TaskGrader.evaluate_agent(
            BaselineAgents.balanced_agent,
            task_id,
            num_episodes=2,
            seed=42
        )
        all_results[task_id] = result
        print(f"  Score: {result.score:.3f}, Passed: {result.passed}\n")
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Task':<12} {'Score':<10} {'Passed':<10} {'Difficulty':<20}")
    print("-" * 52)
    
    total_score = 0
    for task_id in tasks:
        result = all_results[task_id]
        passed = "✅ YES" if result.passed else "❌ NO"
        print(f"{task_id:<12} {result.score:<10.3f} {passed:<10} {result.difficulty:<20}")
        total_score += result.score
    
    print("-" * 52)
    print(f"{'Average':<12} {total_score/len(tasks):<10.3f}")


def cmd_list():
    """List available commands and agents."""
    print_banner("AVAILABLE COMMANDS")
    
    print("Commands:")
    print("  demo          - Run demo episode (20 steps)")
    print("  baseline      - Compare all baseline agents")
    print("  task <id>     - Evaluate balanced agent on task (easy|medium|hard)")
    print("  evaluate      - Full evaluation on all tasks")
    print()
    
    print("Baseline Agents:")
    print("  random        - Purely random scheduling")
    print("  fcfs          - First-Come-First-Served")
    print("  sjf           - Shortest Job First")
    print("  priority      - Highest Priority First")
    print("  resource_aware - Resource utilization optimized")
    print("  balanced      - Priority + Resource awareness (recommended)")
    print()
    
    print("Tasks:")
    print("  easy          - Small queue (5 jobs, easy)")
    print("  medium        - Balanced load (15 jobs, medium)")
    print("  hard          - Peak load (25+ jobs, hard)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OpenEnv Job Scheduling Environment - CLI Interface"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Demo command
    subparsers.add_parser('demo', help='Run demo episode')
    
    # Baseline command
    subparsers.add_parser('baseline', help='Compare baseline agents')
    
    # Task command
    task_parser = subparsers.add_parser('task', help='Evaluate on specific task')
    task_parser.add_argument('task_id', choices=['easy', 'medium', 'hard'], help='Task difficulty')
    task_parser.add_argument('--agent', default='balanced', help='Agent to evaluate')
    
    # Evaluate command
    subparsers.add_parser('evaluate', help='Full evaluation on all tasks')
    
    # List command
    subparsers.add_parser('list', help='List commands and agents')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        cmd_demo()
    elif args.command == 'baseline':
        cmd_baseline()
    elif args.command == 'task':
        cmd_task(args.task_id, args.agent)
    elif args.command == 'evaluate':
        cmd_evaluate()
    elif args.command == 'list':
        cmd_list()
    else:
        parser.print_help()
        print("\n" + "=" * 70)
        print("Quick start: python app.py list")
        print("=" * 70)


if __name__ == "__main__":
    try:
        if len(sys.argv) == 1:
            # Default to demo if no args
            cmd_demo()
        else:
            main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
