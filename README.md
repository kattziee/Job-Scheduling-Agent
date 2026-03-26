# Job Scheduling Optimization Environment (OpenEnv v1.0)

A **real-world production-grade reinforcement learning environment** for data center job scheduling optimization. Agents learn to make scheduling decisions that optimize throughput, latency, and resource utilization under realistic constraints.

## 🎯 Overview

**Real-World Task**: Minimize job completion time and resource waste in a virtualized data center by intelligently scheduling jobs on limited compute and memory resources.

**Problem**: Given a queue of arrived jobs with varying resource requirements (CPU, memory), execution times, and priorities, select which job to execute next to optimize system performance.

**Why It Matters**: 
- Modern cloud platforms (AWS, GCP, Azure) use sophisticated scheduling algorithms
- Improves data center efficiency and reduces operational costs
- Natural testbed for RL agents learning operational policies
- Extends to real-world problems: Kubernetes scheduling, batch processing, edge computing

## 📁 Project Structure

```
job-scheduling-env/
├── job_scheduling_env.py      # Core environment (typed models + API)
├── task_graders.py             # Task definitions & agent graders
├── baseline_inference.py        # Baseline agents & evaluation script
├── openenv.yaml                 # OpenEnv specification
├── requirements.txt             # Python dependencies
├── Dockerfile                   # HuggingFace Spaces deployment
└── README.md                    # This file
```

## 📋 Environment Specification (OpenEnv v1.0)

### State Space (Observation)

```python
{
    "queue_size": int,              # Number of jobs waiting
    "current_time": float,          # Simulation timestep
    "available_resources": {
        "available_cpu": int,       # Free CPU cores
        "available_memory": int     # Free memory GB
    },
    "job_queue": [                  # Jobs waiting for scheduling
        {
            "job_id": int,
            "cpu_required": int,    # 1-16 cores
            "memory_required": int, # 1-32 GB
            "duration": int,        # Execution time steps (5-20)
            "priority": int,        # 1-5 priority level
            "arrival_time": float   # When job arrived
        }
    ],
    "running_jobs": {...},          # Currently executing jobs
    "completed_jobs": int           # Career total completions
}
```

### Action Space

**Discrete** action with 21 possible values:
- `-1` (or value 20): No-op, skip scheduling
- `0-19`: Schedule the job at queue index `[0, 19]`

**Constraint**: Action is only valid if the job fits in available resources (CPU, memory).

### Reward Function

Composite reward signal (range: -1.0 to 1.0) designed to encourage:

1. **Scheduling Efficiency** (20%): Reward for successfully getting jobs running
2. **Completion Speed** (50%): Higher reward for fast job completions
3. **Queue Management** (20%): Penalty for queue buildup
4. **Resource Utilization** (10%): Bonus for high utilization (>70%)

```
reward = w₁·scheduling + w₂·completion - w₃·queue_penalty + w₄·utilization_bonus
```

## 📦 Installation

### Requirements
- Python 3.8+
- NumPy

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline inference
python baseline_inference.py

# Run specific task evaluation
python -c "from task_graders import TaskGrader; TaskGrader.evaluate_agent(lambda obs: -1, 'easy')"
```

### Docker (HuggingFace Spaces)

```bash
# Build
docker build -t job-scheduling-env .

# Run
docker run -it job-scheduling-env
```

## 🎓 Tasks & Evaluation

Three difficulty levels with increasing challenge:

### Task 1: Easy - Small Queue Management
- **Configuration**: 8 CPU, 16 GB memory, 50 steps
- **Goal**: Minimize average job completion time
- **Success**: Process 5+ jobs with avg completion < 20 steps
- **Baseline Score**: 0.70 (requires score ≥ 0.60 to pass)

**Metrics**:
- `completion_efficiency`: Job completion rate and speed (40%)
- `queue_management`: Keep queue < 8 jobs (30%)
- `resource_utilization`: Target 60-80% utilization (30%)

### Task 2: Medium - Balanced Load Scheduling
- **Configuration**: 16 CPU, 32 GB memory, 100 steps
- **Goal**: Optimize throughput while maintaining fairness
- **Success**: Process 15+ jobs with high throughput and fair waiting
- **Baseline Score**: 0.65 (requires score ≥ 0.65 to pass)

**Metrics**:
- `throughput`: Target 0.15+ jobs/step (40%)
- `fairness`: Keep avg wait time < 15 steps (35%)
- `efficiency`: Resource utilization & low idle time (25%)

### Task 3: Hard - Peak Load Management
- **Configuration**: 16 CPU, 32 GB memory, 150 steps, variable load
- **Goal**: Handle high-load scenarios with resource contention
- **Success**: Process 25+ jobs, max queue size ≤ 25
- **Baseline Score**: 0.60 (requires score ≥ 0.60 to pass)

**Metrics**:
- `peak_throughput`: Target 0.167+ jobs/step (35%)
- `queue_stability`: Max queue < 25, avg queue < 15 (35%)
- `optimization`: Resource efficiency under pressure (30%)

## 🤖 Usage Examples

### Basic Environment Loop

```python
from job_scheduling_env import JobSchedulingEnv

# Create environment
env = JobSchedulingEnv(
    num_cpus=16,
    num_memory=32,
    episode_length=100,
    seed=42
)

# Reset for new episode
observation = env.reset()

# Interact
for step in range(100):
    # Your agent selects action
    action = select_action(observation)
    
    # Step environment
    result = env.step(action)
    
    # Unpack return values
    next_observation = result.observation
    reward = result.reward
    done = result.done
    truncated = result.truncated
    info = result.info
    
    observation = next_observation
    
    if done:
        break

# Get metrics
metrics = env.get_metrics()
print(f"Completed jobs: {metrics['total_jobs_completed']}")
print(f"Avg completion time: {metrics['avg_completion_time']:.2f}")
print(f"Throughput: {metrics['throughput']:.3f}")
```

### Using Task Graders

```python
from task_graders import TaskGrader
import numpy as np

# Define your agent policy
def my_agent(observation):
    # Your scheduling logic here
    if not observation.job_queue:
        return -1
    return np.random.randint(0, len(observation.job_queue))

# Evaluate on tasks
for task in ['easy', 'medium', 'hard']:
    result = TaskGrader.evaluate_agent(
        my_agent,
        task_id=task,
        num_episodes=3,
        seed=42
    )
    
    print(f"Task: {task}")
    print(f"  Score: {result.score:.3f}")
    print(f"  Passed: {result.passed}")
    print(f"  Metrics: {result.metrics}")
```

### Baseline Agents

```python
from baseline_inference import BaselineAgents, BaselineEvaluator

# Predefined baseline strategies
agents = {
    'random': BaselineAgents.random_agent,
    'fcfs': BaselineAgents.fcfs_agent,
    'sjf': BaselineAgents.greedy_shortest,
    'priority': BaselineAgents.greedy_priority,
    'resource_aware': BaselineAgents.greedy_resource_aware,
    'balanced': BaselineAgents.balanced_agent,
}

# Evaluate baseline
evaluator = BaselineEvaluator(seed=42)
result = evaluator.evaluate_agent(agents['balanced'], "Balanced Agent", num_episodes=5)

print(result['aggregate'])
# Output: {
#   'total_jobs_completed': 18.6,
#   'avg_completion_time': 12.4,
#   'throughput': 0.186,
#   'avg_utilization': 0.72,
#   ...
# }
```

## 📊 Expected Baseline Performance

Using balanced heuristic agent (`priority + resource_awareness`):

| Task | Jobs/Episode | Avg Time | Throughput | Score | Status |
|------|--------------|----------|------------|-------|--------|
| Easy | 8-10 | 15-18 | 0.16-0.20 | 0.70-0.75 | ✅ Pass |
| Medium | 16-20 | 12-16 | 0.16-0.20 | 0.65-0.72 | ✅ Pass |
| Hard | 20-28 | 14-18 | 0.15-0.20 | 0.60-0.68 | ✅ Pass |

*Baseline scores are reproducible with seed=42*

## 🔧 API Reference

### JobSchedulingEnv

```python
class JobSchedulingEnv:
    def __init__(
        self,
        num_cpus: int = 16,
        num_memory: int = 32,
        queue_max_size: int = 20,
        episode_length: int = 100,
        seed: Optional[int] = None
    )
    
    def reset() -> ObservationSpace
        """Reset environment, return initial observation"""
    
    def step(action: int) -> StepReturn
        """Execute action, return (observation, reward, done, truncated, info)"""
    
    def state() -> ObservationSpace
        """Get current state without stepping"""
    
    def get_metrics() -> Dict
        """Get episode performance metrics"""
```

### TaskGrader

```python
class TaskGrader:
    @staticmethod
    def evaluate_agent(
        agent_fn: Callable,
        task_id: str,          # "easy", "medium", "hard"
        num_episodes: int = 3,
        seed: int = 42
    ) -> TaskResult
    
    # Returns TaskResult with:
    # - score: float [0.0, 1.0]
    # - passed: bool
    # - metrics: Dict
    # - feedback: str
```


## 🎯 Key Features

✅ **Production-Grade**
- Full typed models (dataclasses)
- Comprehensive OpenEnv specification
- Reproducible randomization

✅ **Real-World**
- Realistic job characteristics (CPU, memory, duration, priority)
- Resource constraints and allocation logic
- Dynamic job arrivals and variable load

✅ **Educational**
- 3 difficulty levels (easy → hard)
- Multiple baseline strategies for comparison
- Detailed metrics and feedback

✅ **Extensible**
- Easy to add custom constraints
- Support for different resource types
- Pluggable reward functions

✅ **Reproducible**
- Deterministic with seed control
- Consistent baseline scores
- JSON results export

## 💡 Learning Resources

### Agent Development Tips

1. **Start Simple**: Use FCFS or priority-based heuristics as baseline
2. **Greedy Algorithms**: Resource-aware scheduling often outperforms naive approaches
3. **State Representation**: Consider normalizing state features for RL
4. **Action Masking**: Enforce resource constraints in action selection
5. **Reward Shaping**: Partial progress signals help convergence

### Advanced Variations
- Multi-agent scheduling (competing agents)
- Dynamic resource prices (incentive-based)
- Job preemption (early termination allowed)
- Service level agreements (SLA compliance)
- Energy-aware scheduling (power constraints)

## 📚 References

- OpenAI Gym: https://gymnasium.farama.org/
- OpenEnv Specification: https://huggingface.co/spaces/openenv/docs
- Scheduling Theory: https://en.wikipedia.org/wiki/Job_shop_scheduling
- Cloud Scheduling: https://kubernetes.io/docs/concepts/scheduling-eviction/

## 📝 License

Apache 2.0

## 🙋 Support

For issues or questions:
1. Check the examples in this README
2. Review baseline_inference.py for working implementations
3. Examine openenv.yaml for specification details

## 📈 Version History

- **v1.0** (2024-03-26): Initial release
  - Job scheduling optimization environment
  - 3 graded tasks (easy, medium, hard)
  - 6 baseline agent strategies
  - Full OpenEnv specification
  - HuggingFace Spaces ready

---
