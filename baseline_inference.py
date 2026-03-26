"""
Baseline Inference Script for Job Scheduling Environment
Demonstrates multiple agent strategies with reproducible scores
"""

import json
import numpy as np
from typing import Callable, Dict, List
from job_scheduling_env import JobSchedulingEnv
from task_graders import TaskGrader, TaskResult


class BaselineAgents:
    """Collection of baseline agent strategies."""
    
    @staticmethod
    def random_agent(obs) -> int:
        """Purely random scheduling decisions."""
        if not obs.job_queue:
            return -1
        return np.random.randint(0, len(obs.job_queue) + 1) - 1
    
    @staticmethod
    def greedy_shortest(obs) -> int:
        """Greedy: Schedule shortest job first (SJF)."""
        if not obs.job_queue:
            return -1
        
        # Find shortest job that fits
        available = obs.available_resources
        best_idx = -1
        best_duration = float('inf')
        
        for i, job in enumerate(obs.job_queue):
            if (job.cpu_required <= available.available_cpu and 
                job.memory_required <= available.available_memory and
                job.duration < best_duration):
                best_idx = i
                best_duration = job.duration
        
        return best_idx
    
    @staticmethod
    def greedy_priority(obs) -> int:
        """Greedy: Schedule highest priority job first."""
        if not obs.job_queue:
            return -1
        
        available = obs.available_resources
        best_idx = -1
        best_priority = -1
        
        for i, job in enumerate(obs.job_queue):
            if (job.cpu_required <= available.available_cpu and 
                job.memory_required <= available.available_memory and
                job.priority > best_priority):
                best_idx = i
                best_priority = job.priority
        
        return best_idx
    
    @staticmethod
    def greedy_resource_aware(obs) -> int:
        """
        Greedy: Schedule job that best utilizes available resources.
        Prefer jobs that efficiently use available capacity.
        """
        if not obs.job_queue:
            return -1
        
        available = obs.available_resources
        best_idx = -1
        best_utilization = -1
        
        for i, job in enumerate(obs.job_queue):
            if (job.cpu_required <= available.available_cpu and 
                job.memory_required <= available.available_memory):
                # Compute utilization score: how much of available resources does this use?
                cpu_util = job.cpu_required / (available.available_cpu + 1e-6)
                mem_util = job.memory_required / (available.available_memory + 1e-6)
                combined_util = (cpu_util + mem_util) / 2
                
                if combined_util > best_utilization:
                    best_utilization = combined_util
                    best_idx = i
        
        return best_idx
    
    @staticmethod
    def fcfs_agent(obs) -> int:
        """First-Come-First-Served: Schedule jobs in queue order."""
        if not obs.job_queue:
            return -1
        
        available = obs.available_resources
        for i, job in enumerate(obs.job_queue):
            if (job.cpu_required <= available.available_cpu and 
                job.memory_required <= available.available_memory):
                return i
        
        return -1  # No job fits
    
    @staticmethod
    def balanced_agent(obs) -> int:
        """
        Balanced strategy: Mix of priority and resource efficiency.
        Score = 0.4 * priority_norm + 0.6 * resource_util_norm
        """
        if not obs.job_queue:
            return -1
        
        available = obs.available_resources
        best_idx = -1
        best_score = -1
        
        # Get max priority for normalization
        max_priority = max((j.priority for j in obs.job_queue), default=1)
        
        for i, job in enumerate(obs.job_queue):
            if (job.cpu_required <= available.available_cpu and 
                job.memory_required <= available.available_memory):
                
                # Normalize priority to [0, 1]
                priority_score = job.priority / max_priority
                
                # Resource utilization score
                cpu_util = job.cpu_required / (available.available_cpu + 1e-6)
                mem_util = job.memory_required / (available.available_memory + 1e-6)
                resource_score = (cpu_util + mem_util) / 2
                
                # Combine scores
                combined_score = 0.4 * priority_score + 0.6 * resource_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = i
        
        return best_idx


class BaselineEvaluator:
    """Run and evaluate baseline agents."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.results = {}
    
    def run_episode(
        self,
        agent_fn: Callable,
        env_config: Dict,
        seed: int = None
    ) -> Dict:
        """Run one episode with an agent."""
        if seed is None:
            seed = self.seed
        
        env = JobSchedulingEnv(**env_config, seed=seed)
        obs = env.reset()
        
        episode_data = {
            'actions': [],
            'rewards': [],
            'queue_sizes': [],
            'utilizations': [],
            'total_reward': 0
        }
        
        while obs.current_time < env.episode_length:
            action = agent_fn(obs)
            result = env.step(action)
            
            obs = result.observation
            reward = result.reward
            
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['queue_sizes'].append(obs.queue_size)
            episode_data['total_reward'] += reward
            
            util = (env.num_cpus - obs.available_resources.available_cpu) / env.num_cpus
            episode_data['utilizations'].append(util)
        
        metrics = env.get_metrics()
        episode_data.update(metrics)
        
        return episode_data
    
    def evaluate_agent(
        self,
        agent_fn: Callable,
        agent_name: str,
        num_episodes: int = 5
    ) -> Dict:
        """Evaluate an agent across multiple episodes."""
        print(f"\nEvaluating {agent_name}...")
        
        episode_results = []
        for episode in range(num_episodes):
            env_config = {
                'num_cpus': 16,
                'num_memory': 32,
                'queue_max_size': 20,
                'episode_length': 100
            }
            result = self.run_episode(agent_fn, env_config, seed=self.seed + episode)
            episode_results.append(result)
            
            print(f"  Episode {episode + 1}: Jobs={result['total_jobs_completed']}, "
                  f"AvgTime={result['avg_completion_time']:.1f}, "
                  f"Throughput={result['throughput']:.3f}, "
                  f"Return={result['total_reward']:.2f}")
        
        # Aggregate stats
        avg_stats = {
            'total_jobs_completed': np.mean([r['total_jobs_completed'] for r in episode_results]),
            'avg_completion_time': np.mean([r['avg_completion_time'] for r in episode_results]),
            'throughput': np.mean([r['throughput'] for r in episode_results]),
            'avg_wait_time': np.mean([r['avg_wait_time'] for r in episode_results]),
            'total_reward': np.mean([r['total_reward'] for r in episode_results]),
            'avg_utilization': np.mean([np.mean(r['utilizations']) for r in episode_results]),
        }
        
        return {
            'agent_name': agent_name,
            'num_episodes': num_episodes,
            'episodes': episode_results,
            'aggregate': avg_stats
        }


def main():
    """Main baseline evaluation script."""
    print("=" * 70)
    print("OpenEnv Job Scheduling Environment - Baseline Inference")
    print("=" * 70)
    
    evaluator = BaselineEvaluator(seed=42)
    
    # Define baseline agents
    agents = [
        (BaselineAgents.random_agent, "Random"),
        (BaselineAgents.fcfs_agent, "FCFS (First-Come-First-Served)"),
        (BaselineAgents.greedy_shortest, "Greedy - Shortest Job First (SJF)"),
        (BaselineAgents.greedy_priority, "Greedy - Highest Priority"),
        (BaselineAgents.greedy_resource_aware, "Greedy - Resource Aware"),
        (BaselineAgents.balanced_agent, "Balanced (Priority + Resources)")
    ]
    
    all_results = {}
    
    # Run baseline evaluations
    print("\n" + "=" * 70)
    print("BASELINE AGENT EVALUATIONS (Medium Task Configuration)")
    print("=" * 70)
    
    for agent_fn, agent_name in agents:
        result = evaluator.evaluate_agent(agent_fn, agent_name, num_episodes=5)
        all_results[agent_name] = result
    
    # Task-specific grading
    print("\n" + "=" * 70)
    print("TASK-SPECIFIC GRADING (Easy / Medium / Hard)")
    print("=" * 70)
    
    task_results = {}
    for task_id in ['easy', 'medium', 'hard']:
        print(f"\n--- {task_id.upper()} TASK ---")
        
        # Test with balanced agent
        agent_result = TaskGrader.evaluate_agent(
            BaselineAgents.balanced_agent,
            task_id,
            num_episodes=3,
            seed=42
        )
        task_results[task_id] = agent_result
        
        print(f"Score: {agent_result.score:.3f}")
        print(f"Passed: {agent_result.passed}")
        print(f"Feedback: {agent_result.feedback}")
        print(f"Metrics: {json.dumps(agent_result.metrics, indent=2)}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Agent Name':<35} {'Jobs':<8} {'Avg Time':<12} {'Throughput':<12} {'Reward':<10}")
    print("-" * 77)
    
    for agent_name in sorted(all_results.keys()):
        stats = all_results[agent_name]['aggregate']
        print(f"{agent_name:<35} {stats['total_jobs_completed']:<8.1f} "
              f"{stats['avg_completion_time']:<12.2f} {stats['throughput']:<12.4f} "
              f"{stats['total_reward']:<10.2f}")
    
    # Save results
    output = {
        'timestamp': json.dumps(
            {'year': 2024, 'month': 3, 'day': 26},
            indent=2
        ),
        'environment': 'JobSchedulingEnv-v1',
        'baseline_results': {
            name: {
                'agent': name,
                'episodes': result['num_episodes'],
                'aggregate': result['aggregate']
            }
            for name, result in all_results.items()
        },
        'task_results': {
            task_id: {
                'task': task_id,
                'score': result.score,
                'passed': result.passed,
                'metrics': result.metrics,
                'feedback': result.feedback
            }
            for task_id, result in task_results.items()
        }
    }
    
    with open('baseline_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to baseline_results.json")
    print("\n" + "=" * 70)
    print("Reproducibility: Use seed=42 for consistent results across runs")
    print("=" * 70)


if __name__ == "__main__":
    main()
