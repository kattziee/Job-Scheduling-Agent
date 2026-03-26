"""
OpenEnv Task Graders for Job Scheduling Environment
Three difficulty levels with normalized scoring (0.0-1.0)
"""

from dataclasses import dataclass
from typing import Dict, Callable, Tuple
import numpy as np
from job_scheduling_env import JobSchedulingEnv


@dataclass
class TaskResult:
    """Result from task evaluation."""
    task_id: str
    difficulty: str
    score: float  # 0.0-1.0
    metrics: Dict
    passed: bool
    feedback: str


class TaskGrader:
    """Agent grader for job scheduling tasks."""
    
    @staticmethod
    def grade_easy(metrics: Dict, episode_info: Dict) -> TaskResult:
        """
        Task 1 (Easy): Small Queue Management
        
        Goal: Minimize average job completion time
        Focus: Learning basic scheduling decisions
        Success: Process 5+ jobs with avg completion time < 20 steps
        
        Scoring:
        - Job completion efficiency: 40%
        - Queue management: 30%
        - Resource utilization: 30%
        """
        job_completion = metrics.get('total_jobs_completed', 0)
        avg_completion_time = metrics.get('avg_completion_time', float('inf'))
        throughput = metrics.get('throughput', 0.0)
        
        scores = {}
        
        # Completion efficiency (0-1.0)
        # Target: 8+ jobs in 50 steps, completion time < 20
        if job_completion >= 8:
            scores['completion'] = 1.0
        elif job_completion >= 5:
            scores['completion'] = 0.5 + (job_completion - 5) / 6 * 0.5
        else:
            scores['completion'] = max(0.0, job_completion / 5 * 0.5)
        
        # Penalty for slow completion
        if avg_completion_time < 20:
            time_score = 1.0
        elif avg_completion_time < 30:
            time_score = 0.5 + (30 - avg_completion_time) / 10 * 0.5
        else:
            time_score = max(0.0, 1.0 - (avg_completion_time - 30) / 50)
        
        scores['completion'] = scores['completion'] * 0.7 + time_score * 0.3
        
        # Queue management (0-1.0)
        # Target: Keep queue size < 8 on average
        avg_queue_size = episode_info.get('avg_queue_size', 20)
        if avg_queue_size < 8:
            scores['queue'] = 1.0
        elif avg_queue_size < 12:
            scores['queue'] = 0.5 + (12 - avg_queue_size) / 4 * 0.5
        else:
            scores['queue'] = max(0.0, 1.0 - (avg_queue_size - 12) / 20)
        
        # Resource utilization (0-1.0)
        # Target: 60-80% utilization
        util = episode_info.get('avg_utilization', 0.0)
        if 0.6 <= util <= 0.8:
            scores['utilization'] = 1.0
        elif 0.4 <= util < 0.6:
            scores['utilization'] = 0.5 + (util - 0.4) / 0.2 * 0.5
        else:
            scores['utilization'] = max(0.0, util / 0.6 * 0.5)
        
        # Weighted score
        final_score = (
            scores['completion'] * 0.4 +
            scores['queue'] * 0.3 +
            scores['utilization'] * 0.3
        )
        
        passed = final_score >= 0.6 and job_completion >= 5
        
        return TaskResult(
            task_id="easy",
            difficulty="Easy",
            score=float(np.clip(final_score, 0.0, 1.0)),
            metrics={
                'jobs_completed': job_completion,
                'avg_completion_time': avg_completion_time,
                'avg_queue_size': avg_queue_size,
                'utilization': util,
                **scores
            },
            passed=passed,
            feedback=f"Easy task: Completed {job_completion} jobs (target: 5+), "
                     f"avg time {avg_completion_time:.1f} (target: <20), "
                     f"score {final_score:.3f}"
        )
    
    @staticmethod
    def grade_medium(metrics: Dict, episode_info: Dict) -> TaskResult:
        """
        Task 2 (Medium): Balanced Load Scheduling
        
        Goal: Optimize throughput while maintaining fairness
        Focus: Balancing multiple objectives under pressure
        Success: Process 15+ jobs with high throughput and fair scheduling
        
        Scoring:
        - Throughput efficiency: 40%
        - Fairness (priority handling): 35%
        - Resource efficiency: 25%
        """
        job_completion = metrics.get('total_jobs_completed', 0)
        throughput = metrics.get('throughput', 0.0)
        avg_wait_time = metrics.get('avg_wait_time', float('inf'))
        
        scores = {}
        
        # Throughput efficiency (0-1.0)
        # Target: 15+ jobs in 100 steps = throughput 0.15+
        if throughput >= 0.18:
            scores['throughput'] = 1.0
        elif throughput >= 0.12:
            scores['throughput'] = 0.5 + (throughput - 0.12) / 0.06 * 0.5
        else:
            scores['throughput'] = max(0.0, throughput / 0.12 * 0.5)
        
        # Fairness in wait times (0-1.0)
        # Target: Keep average wait time < 15 steps
        if avg_wait_time < 15:
            scores['fairness'] = 1.0
        elif avg_wait_time < 25:
            scores['fairness'] = 0.5 + (25 - avg_wait_time) / 10 * 0.5
        else:
            scores['fairness'] = max(0.0, 1.0 - (avg_wait_time - 25) / 50)
        
        # Resource efficiency (0-1.0)
        util = episode_info.get('avg_utilization', 0.0)
        idle_penalty = episode_info.get('idle_time_ratio', 0.3)
        
        if 0.65 <= util <= 0.85 and idle_penalty < 0.25:
            scores['efficiency'] = 1.0
        elif util >= 0.55:
            scores['efficiency'] = 0.4 + (util - 0.55) / 0.30 * 0.4 - idle_penalty * 0.2
        else:
            scores['efficiency'] = max(0.0, util / 0.65 * 0.5 - idle_penalty * 0.1)
        
        final_score = (
            scores['throughput'] * 0.4 +
            scores['fairness'] * 0.35 +
            scores['efficiency'] * 0.25
        )
        
        passed = final_score >= 0.65 and job_completion >= 12
        
        return TaskResult(
            task_id="medium",
            difficulty="Medium",
            score=float(np.clip(final_score, 0.0, 1.0)),
            metrics={
                'jobs_completed': job_completion,
                'throughput': throughput,
                'avg_wait_time': avg_wait_time,
                'utilization': util,
                **scores
            },
            passed=passed,
            feedback=f"Medium task: Completed {job_completion} jobs (target: 15+), "
                     f"throughput {throughput:.3f} (target: 0.15+), "
                     f"wait time {avg_wait_time:.1f} (target: <15), "
                     f"score {final_score:.3f}"
        )
    
    @staticmethod
    def grade_hard(metrics: Dict, episode_info: Dict) -> TaskResult:
        """
        Task 3 (Hard): Peak Load Management
        
        Goal: Handle variable, high-load scenarios with resource contention
        Focus: Robust scheduling under adversarial conditions
        Success: Process 25+ jobs while maintaining queue < 25
        
        Scoring:
        - Peak throughput: 35%
        - Queue stability: 35%
        - Resource optimization: 30%
        """
        job_completion = metrics.get('total_jobs_completed', 0)
        throughput = metrics.get('throughput', 0.0)
        max_queue_size = episode_info.get('max_queue_size', 100)
        avg_queue_size = episode_info.get('avg_queue_size', 50)
        
        scores = {}
        
        # Peak throughput (0-1.0)
        # Target: 25+ jobs in 150 steps = throughput 0.167+
        if throughput >= 0.20:
            scores['peak_throughput'] = 1.0
        elif throughput >= 0.13:
            scores['peak_throughput'] = 0.3 + (throughput - 0.13) / 0.07 * 0.7
        else:
            scores['peak_throughput'] = max(0.0, throughput / 0.13 * 0.3)
        
        # Queue stability under pressure (0-1.0)
        # Target: Max queue < 25, average < 15
        if max_queue_size <= 25 and avg_queue_size <= 15:
            scores['stability'] = 1.0
        elif max_queue_size <= 30:
            queue_factor = (30 - max_queue_size) / 5 * 0.5
            avg_factor = max(0, 1.0 - (avg_queue_size - 15) / 20)
            scores['stability'] = 0.5 + min(queue_factor, avg_factor) * 0.5
        else:
            scores['stability'] = max(0.0, 1.0 - (max_queue_size - 30) / 50)
        
        # Resource optimization under contention (0-1.0)
        util = episode_info.get('avg_utilization', 0.0)
        waste = episode_info.get('resource_waste_ratio', 0.5)
        fragmentation = episode_info.get('fragmentation_ratio', 0.3)
        
        if util >= 0.75 and waste < 0.15 and fragmentation < 0.2:
            scores['optimization'] = 1.0
        else:
            opt_score = (util / 0.85) * 0.4
            opt_score += max(0, 1.0 - waste / 0.3) * 0.35
            opt_score += max(0, 1.0 - fragmentation / 0.4) * 0.25
            scores['optimization'] = opt_score
        
        final_score = (
            scores['peak_throughput'] * 0.35 +
            scores['stability'] * 0.35 +
            scores['optimization'] * 0.30
        )
        
        passed = final_score >= 0.60 and job_completion >= 20 and max_queue_size <= 25
        
        return TaskResult(
            task_id="hard",
            difficulty="Hard",
            score=float(np.clip(final_score, 0.0, 1.0)),
            metrics={
                'jobs_completed': job_completion,
                'throughput': throughput,
                'max_queue_size': max_queue_size,
                'avg_queue_size': avg_queue_size,
                'utilization': util,
                'resource_waste': waste,
                **scores
            },
            passed=passed,
            feedback=f"Hard task: Completed {job_completion} jobs (target: 25+), "
                     f"max queue {max_queue_size} (target: <25), "
                     f"throughput {throughput:.3f} (target: 0.167+), "
                     f"utilization {util:.2%}, score {final_score:.3f}"
        )
    
    @staticmethod
    def evaluate_agent(agent_fn: Callable, task_id: str, num_episodes: int = 3, seed: int = 42) -> TaskResult:
        """
        Evaluate an agent policy on a task.
        
        Args:
            agent_fn: Function that takes observation and returns action
            task_id: "easy", "medium", or "hard"
            num_episodes: Number of episodes to average
            seed: Random seed for reproducibility
        
        Returns:
            TaskResult with comprehensive evaluation
        """
        # Task configurations
        task_configs = {
            'easy': {
                'queue_max_size': 10,
                'episode_length': 50,
                'num_cpus': 8,
                'num_memory': 16
            },
            'medium': {
                'queue_max_size': 20,
                'episode_length': 100,
                'num_cpus': 16,
                'num_memory': 32
            },
            'hard': {
                'queue_max_size': 30,
                'episode_length': 150,
                'num_cpus': 16,
                'num_memory': 32
            }
        }
        
        config = task_configs.get(task_id, task_configs['medium'])
        
        all_metrics = {
            'total_jobs_completed': [],
            'avg_completion_time': [],
            'avg_wait_time': [],
            'throughput': []
        }
        
        all_episode_info = {
            'avg_queue_size': [],
            'max_queue_size': [],
            'avg_utilization': [],
            'idle_time_ratio': [],
            'resource_waste_ratio': [],
            'fragmentation_ratio': []
        }
        
        for episode in range(num_episodes):
            env = JobSchedulingEnv(
                num_cpus=config['num_cpus'],
                num_memory=config['num_memory'],
                queue_max_size=config['queue_max_size'],
                episode_length=config['episode_length'],
                seed=seed + episode
            )
            
            obs = env.reset()
            total_reward = 0
            queue_sizes = []
            utilizations = []
            
            for step in range(config['episode_length']):
                # Get action from agent
                action = agent_fn(obs)
                
                # Step environment
                result = env.step(action)
                obs = result.observation
                total_reward += result.reward
                
                # Track metrics
                queue_sizes.append(obs.queue_size)
                util = (env.num_cpus - obs.available_resources.available_cpu) / env.num_cpus
                utilizations.append(util)
            
            metrics = env.get_metrics()
            all_metrics['total_jobs_completed'].append(metrics['total_jobs_completed'])
            all_metrics['avg_completion_time'].append(metrics['avg_completion_time'])
            all_metrics['avg_wait_time'].append(metrics['avg_wait_time'])
            all_metrics['throughput'].append(metrics['throughput'])
            
            all_episode_info['avg_queue_size'].append(np.mean(queue_sizes))
            all_episode_info['max_queue_size'].append(max(queue_sizes))
            all_episode_info['avg_utilization'].append(np.mean(utilizations))
            all_episode_info['idle_time_ratio'].append(1.0 - np.mean(utilizations))
            all_episode_info['resource_waste_ratio'].append(max(0, 1.0 - np.mean(utilizations)))
            all_episode_info['fragmentation_ratio'].append(np.std(utilizations) / (np.mean(utilizations) + 1e-6))
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        avg_episode_info = {k: np.mean(v) for k, v in all_episode_info.items()}
        
        # Grade based on task difficulty
        if task_id == 'easy':
            return TaskGrader.grade_easy(avg_metrics, avg_episode_info)
        elif task_id == 'medium':
            return TaskGrader.grade_medium(avg_metrics, avg_episode_info)
        elif task_id == 'hard':
            return TaskGrader.grade_hard(avg_metrics, avg_episode_info)
        else:
            raise ValueError(f"Unknown task_id: {task_id}")


# Example usage
if __name__ == "__main__":
    # Random baseline agent
    def random_agent(obs):
        return np.random.randint(-1, len(obs.job_queue) + 1)
    
    # Greedy agent: schedule highest priority job
    def greedy_priority_agent(obs):
        if not obs.job_queue:
            return -1
        best_idx = 0
        best_priority = obs.job_queue[0].priority
        for i, job in enumerate(obs.job_queue):
            if job.priority > best_priority:
                best_priority = job.priority
                best_idx = i
        return best_idx
    
    print("=== Easy Task ===")
    result_easy = TaskGrader.evaluate_agent(random_agent, 'easy', num_episodes=2)
    print(f"Score: {result_easy.score:.3f}, Passed: {result_easy.passed}")
    print(f"Feedback: {result_easy.feedback}")
    
    print("\n=== Medium Task ===")
    result_medium = TaskGrader.evaluate_agent(greedy_priority_agent, 'medium', num_episodes=2)
    print(f"Score: {result_medium.score:.3f}, Passed: {result_medium.passed}")
    print(f"Feedback: {result_medium.feedback}")
    
    print("\n=== Hard Task ===")
    result_hard = TaskGrader.evaluate_agent(random_agent, 'hard', num_episodes=1)
    print(f"Score: {result_hard.score:.3f}, Passed: {result_hard.passed}")
    print(f"Feedback: {result_hard.feedback}")
