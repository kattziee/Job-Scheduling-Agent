"""
Job Scheduling Optimization Environment - OpenEnv Specification
Real-world task: Minimize job completion time and resource waste in a data center

Environment: Agents learn to schedule jobs on limited resources to optimize throughput
and latency while maintaining fairness.

Typed models and full OpenEnv spec with step()/reset()/state() API.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import json


# ============================================================================
# TYPED MODELS (OpenEnv Specification)
# ============================================================================

@dataclass
class Job:
    """Represents a job in the queue."""
    job_id: int
    arrival_time: float
    cpu_required: int
    memory_required: int
    duration: int  # Expected execution time
    priority: int  # 1-5, where 5 is highest
    
    def to_dict(self) -> Dict:
        return {
            'job_id': self.job_id,
            'arrival_time': self.arrival_time,
            'cpu_required': self.cpu_required,
            'memory_required': self.memory_required,
            'duration': self.duration,
            'priority': self.priority
        }


@dataclass
class Resource:
    """Current system resources."""
    total_cpu: int
    total_memory: int
    available_cpu: int
    available_memory: int
    
    def to_dict(self) -> Dict:
        return {
            'total_cpu': self.total_cpu,
            'total_memory': self.total_memory,
            'available_cpu': self.available_cpu,
            'available_memory': self.available_memory
        }


@dataclass
class ObservationSpace:
    """Observation space specification."""
    queue_size: int
    current_time: float
    available_resources: Resource
    job_queue: List[Job]
    running_jobs: Dict[int, Job]
    completed_jobs: int
    
    def to_dict(self) -> Dict:
        return {
            'queue_size': self.queue_size,
            'current_time': self.current_time,
            'available_resources': self.available_resources.to_dict(),
            'job_queue': [j.to_dict() for j in self.job_queue],
            'running_jobs': {k: v.to_dict() for k, v in self.running_jobs.items()},
            'completed_jobs': self.completed_jobs
        }


@dataclass
class ActionSpace:
    """Action space specification."""
    valid_actions: List[int]  # Job indices that can be scheduled
    
    def to_dict(self) -> Dict:
        return {'valid_actions': self.valid_actions}


@dataclass
class StepReturn:
    """Return value from step() method."""
    observation: ObservationSpace
    reward: float
    done: bool
    truncated: bool
    info: Dict


@dataclass
class Metadata:
    """Environment metadata."""
    name: str = "JobSchedulingEnv-v1"
    version: str = "1.0.0"
    description: str = "Real-world job scheduling optimization task"
    author: str = "AI Agent Lab"
    created: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# JOB SCHEDULING ENVIRONMENT
# ============================================================================

class JobSchedulingEnv:
    """
    OpenEnv-compliant job scheduling environment.
    
    Real-world task: Schedule jobs on limited resources to minimize:
    - Average job completion time
    - Resource idle time
    - Job wait time
    
    Action: Select which job from queue to schedule next
    Observation: Current queue, resources, job details
    Reward: Composite signal based on throughput, latency, and resource utilization
    """
    
    def __init__(
        self,
        num_cpus: int = 16,
        num_memory: int = 32,
        queue_max_size: int = 20,
        episode_length: int = 100,
        seed: Optional[int] = None
    ):
        self.num_cpus = num_cpus
        self.num_memory = num_memory
        self.queue_max_size = queue_max_size
        self.episode_length = episode_length
        self.seed_value = seed
        if seed is not None:
            np.random.seed(seed)
        
        self.metadata = Metadata()
        
        # State variables
        self.current_time = 0.0
        self.job_queue: List[Job] = []
        self.running_jobs: Dict[int, Tuple[Job, int]] = {}  # job_id -> (job, remaining_time)
        self.completed_jobs: List[Job] = []
        self.total_jobs_created = 0
        self.job_counter = 0
        self.step_count = 0
        
        # Metrics
        self.total_wait_time = 0.0
        self.total_completion_time = 0.0
        self.total_cpu_util = 0.0
        self.total_memory_util = 0.0
        
    def reset(self) -> ObservationSpace:
        """Reset environment to initial state."""
        self.current_time = 0.0
        self.job_queue = []
        self.running_jobs = {}
        self.completed_jobs = []
        self.job_counter = 0
        self.step_count = 0
        self.total_wait_time = 0.0
        self.total_completion_time = 0.0
        self.total_cpu_util = 0.0
        self.total_memory_util = 0.0
        
        # Generate initial jobs
        for _ in range(5):
            self._generate_job()
        
        return self.state()
    
    def _generate_job(self):
        """Generate a random job."""
        job = Job(
            job_id=self.job_counter,
            arrival_time=self.current_time + np.random.exponential(scale=2.0),
            cpu_required=np.random.randint(1, self.num_cpus + 1),
            memory_required=np.random.randint(1, self.num_memory + 1),
            duration=np.random.randint(5, 20),
            priority=np.random.randint(1, 6)
        )
        self.job_counter += 1
        return job
    
    def step(self, action: int) -> StepReturn:
        """
        Execute one step of the environment.
        
        Args:
            action: Index of job to schedule from the queue (or -1 to skip)
        
        Returns:
            StepReturn with observation, reward, done, truncated, and info
        """
        # Clamp action
        if action < 0 or action >= len(self.job_queue):
            action = -1  # Skip
        
        # Update time
        self.current_time += 1.0
        
        # Admit jobs that have arrived
        self.job_queue.extend([j for j in [self._generate_job()] if j.arrival_time <= self.current_time])
        
        # Schedule selected job if possible
        scheduled_job = None
        if action >= 0 and action < len(self.job_queue):
            job = self.job_queue[action]
            resource = self._get_available_resources()
            
            if resource.available_cpu >= job.cpu_required and resource.available_memory >= job.memory_required:
                self.job_queue.pop(action)
                self.running_jobs[job.job_id] = (job, job.duration)
                scheduled_job = job
        
        # Execute running jobs
        completed_this_step = []
        job_ids_to_remove = []
        for job_id, (job, remaining_time) in list(self.running_jobs.items()):
            remaining_time -= 1
            if remaining_time <= 0:
                self.running_jobs.pop(job_id)
                self.completed_jobs.append(job)
                completed_this_step.append(job)
                job_ids_to_remove.append(job_id)
                
                # Accumulate metrics
                wait_time = job.duration  # Simplified
                self.total_wait_time += wait_time
                self.total_completion_time += (self.current_time - job.arrival_time + job.duration)
            else:
                self.running_jobs[job_id] = (job, remaining_time)
        
        # Calculate reward
        reward = self._calculate_reward(scheduled_job, completed_this_step)
        
        # Check if done
        self.step_count += 1
        done = self.step_count >= self.episode_length
        
        # Info
        resource = self._get_available_resources()
        info = {
            'scheduled': scheduled_job.job_id if scheduled_job else -1,
            'completed': [j.job_id for j in completed_this_step],
            'queue_size': len(self.job_queue),
            'running_jobs': len(self.running_jobs),
            'total_completed': len(self.completed_jobs),
            'cpu_utilization': (self.num_cpus - resource.available_cpu) / self.num_cpus,
            'memory_utilization': (self.num_memory - resource.available_memory) / self.num_memory,
        }
        
        return StepReturn(
            observation=self.state(),
            reward=reward,
            done=done,
            truncated=False,
            info=info
        )
    
    def state(self) -> ObservationSpace:
        """
        Get current state observation.
        
        Returns:
            ObservationSpace with complete environment state
        """
        resource = self._get_available_resources()
        return ObservationSpace(
            queue_size=len(self.job_queue),
            current_time=self.current_time,
            available_resources=resource,
            job_queue=self.job_queue.copy(),
            running_jobs={jid: job for jid, (job, _) in self.running_jobs.items()},
            completed_jobs=len(self.completed_jobs)
        )
    
    def _get_available_resources(self) -> Resource:
        """Get current available resources."""
        used_cpu = sum(job.cpu_required for job, _ in self.running_jobs.values())
        used_memory = sum(job.memory_required for job, _ in self.running_jobs.values())
        
        return Resource(
            total_cpu=self.num_cpus,
            total_memory=self.num_memory,
            available_cpu=self.num_cpus - used_cpu,
            available_memory=self.num_memory - used_memory
        )
    
    def _calculate_reward(self, scheduled_job: Optional[Job], completed_jobs: List[Job]) -> float:
        """
        Calculate reward based on scheduling efficiency.
        
        Components:
        - Scheduling reward: Positive for scheduling jobs
        - Completion reward: Positive for completing jobs quickly
        - Efficiency penalty: Penalty for resource waste
        """
        reward = 0.0
        
        # Reward for scheduling
        if scheduled_job:
            reward += 0.1  # Base scheduling reward
        
        # Reward for completing jobs quickly
        for job in completed_jobs:
            completion_time = self.current_time - job.arrival_time
            if completion_time < job.duration + 5:
                reward += 0.5  # Fast completion bonus
            else:
                reward += 0.2  # Standard completion reward
        
        # Penalty for queue buildup
        if len(self.job_queue) > self.queue_max_size * 0.8:
            reward -= 0.1
        
        # Resource utilization bonus
        resource = self._get_available_resources()
        utilization = (self.num_cpus - resource.available_cpu) / self.num_cpus
        if utilization > 0.7:
            reward += 0.05
        
        return reward
    
    def get_metrics(self) -> Dict:
        """Get episode metrics."""
        total_jobs = len(self.completed_jobs)
        if total_jobs == 0:
            return {
                'total_jobs_completed': 0,
                'avg_completion_time': 0.0,
                'avg_wait_time': 0.0,
                'throughput': 0.0
            }
        
        return {
            'total_jobs_completed': total_jobs,
            'avg_completion_time': self.total_completion_time / total_jobs,
            'avg_wait_time': self.total_wait_time / total_jobs,
            'throughput': total_jobs / (self.current_time + 1e-6)
        }


if __name__ == "__main__":
    # Quick test
    env = JobSchedulingEnv(seed=42)
    obs = env.reset()
    print(f"Initial observation: Queue size={obs.queue_size}, Resources={obs.available_resources.available_cpu}/{env.num_cpus} CPU")
    
    total_reward = 0
    for step in range(20):
        action = np.random.randint(-1, len(obs.job_queue))
        result = env.step(action)
        obs = result.observation
        reward = result.reward
        total_reward += reward
        print(f"Step {step+1}: Action={action}, Reward={reward:.3f}, Queue={obs.queue_size}, Running={len(obs.running_jobs)}")
    
    metrics = env.get_metrics()
    print(f"\nMetrics: {metrics}")
