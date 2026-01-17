# Depth Limited Search: Professional Lab Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Theoretical Foundations](#theoretical-foundations)
3. [Algorithm Implementation](#algorithm-implementation)
4. [Practical Examples](#practical-examples)
5. [Industry Applications](#industry-applications)
6. [Performance Analysis](#performance-analysis)
7. [Advanced Topics](#advanced-topics)
8. [Lab Exercises](#lab-exercises)
9. [Best Practices](#best-practices)
10. [References](#references)

---

## Introduction

Depth Limited Search (DLS) is a systematic search algorithm that explores a graph or tree up to a specified depth limit. It serves as a crucial intermediate step between Depth First Search (DFS) and Iterative Deepening Search (IDS), offering controlled exploration with bounded memory requirements.

### Key Characteristics
- **Memory Efficiency**: O(b×d) where b is branching factor and d is depth limit
- **Time Complexity**: O(b^d) in worst case
- **Completeness**: Complete if solution depth ≤ limit
- **Optimality**: Not optimal (may find longer path first)

### When to Use DLS
- Large search spaces with unknown solution depth
- Memory-constrained environments
- Real-time systems requiring bounded computation
- Hierarchical problem domains

---

## Theoretical Foundations

### 1. Search Space Representation

A search space can be represented as:
- **Tree**: Acyclic structure with single root
- **Graph**: May contain cycles and multiple parents

### 2. Depth Limit Selection

The choice of depth limit critically impacts performance:

```
Optimal Limit ≈ Solution Depth + Safety Margin
```

Factors influencing limit selection:
- Domain knowledge
- Available computational resources
- Time constraints
- Solution depth distribution

### 3. Mathematical Analysis

#### Memory Requirements
```
Space Complexity = O(b × d)
Where:
- b = average branching factor
- d = depth limit
```

#### Time Complexity
```
Time Complexity = O(b^d)
Worst case: explores all nodes to depth d
```

---

## Algorithm Implementation

### Basic DLS Algorithm

```python
def depth_limited_search(node, goal, limit, depth=0):
    """
    Basic Depth Limited Search implementation
    
    Args:
        node: Current node in search tree
        goal: Goal state or condition
        limit: Maximum depth to explore
        depth: Current depth (default: 0)
    
    Returns:
        Solution path or None if not found
    """
    if goal_test(node):
        return [node]
    
    if depth == limit:
        return None  # Cutoff occurred
    
    for child in expand_node(node):
        result = depth_limited_search(child, goal, limit, depth + 1)
        if result is not None:
            return [node] + result
    
    return None
```

### Enhanced DLS with Cutoff Detection

```python
def depth_limited_search_enhanced(node, goal, limit, depth=0):
    """
    Enhanced DLS with cutoff detection and path tracking
    """
    if goal_test(node):
        return {'path': [node], 'found': True, 'cutoff': False}
    
    if depth == limit:
        return {'path': None, 'found': False, 'cutoff': True}
    
    cutoff_occurred = False
    
    for child in expand_node(node):
        result = depth_limited_search_enhanced(child, goal, limit, depth + 1)
        
        if result['found']:
            result['path'] = [node] + result['path']
            return result
        
        if result['cutoff']:
            cutoff_occurred = True
    
    if cutoff_occurred:
        return {'path': None, 'found': False, 'cutoff': True}
    else:
        return {'path': None, 'found': False, 'cutoff': False}
```

---

## Practical Examples

### Example 1: Maze Navigation

```python
class MazeNode:
    def __init__(self, position, maze, parent=None):
        self.position = position
        self.maze = maze
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
    
    def get_neighbors(self):
        """Generate valid neighboring positions"""
        x, y = self.position
        neighbors = []
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        
        for dx, dy in moves:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < len(self.maze[0]) and 
                0 <= new_y < len(self.maze) and 
                self.maze[new_y][new_x] != 1):  # 1 represents wall
                neighbors.append(MazeNode((new_x, new_y), self.maze, self))
        
        return neighbors
    
    def is_goal(self, goal_position):
        return self.position == goal_position

def solve_maze_dls(maze, start, goal, depth_limit):
    """
    Solve maze using Depth Limited Search
    """
    start_node = MazeNode(start, maze)
    return depth_limited_search(start_node, goal, depth_limit)

# Example maze (0 = path, 1 = wall, 2 = start, 3 = goal)
maze = [
    [2, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 3, 1]
]
```

### Example 2: Game Tree Analysis

```python
class GameNode:
    def __init__(self, state, player, parent=None, move=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.move = move
        self.depth = 0 if parent is None else parent.depth + 1
    
    def generate_moves(self):
        """Generate all possible moves from current state"""
        moves = []
        # Implementation depends on specific game rules
        return moves
    
    def apply_move(self, move):
        """Apply move and return new state"""
        # Game-specific implementation
        pass
    
    def is_terminal(self):
        """Check if current state is terminal"""
        # Game-specific implementation
        pass
    
    def evaluate(self):
        """Evaluate current state (for games with utility function)"""
        # Game-specific evaluation function
        pass

def game_tree_dls(root_state, max_depth, player):
    """
    Analyze game tree using Depth Limited Search
    """
    root = GameNode(root_state, player)
    best_move = None
    best_value = float('-inf')
    
    for move in root.generate_moves():
        new_state = root.apply_move(move)
        child = GameNode(new_state, 3 - player, root, move)  # Switch player
        
        value = minimax_dls(child, max_depth - 1, False)
        
        if value > best_value:
            best_value = value
            best_move = move
    
    return best_move, best_value

def minimax_dls(node, depth, is_maximizing):
    """Minimax with depth limit"""
    if depth == 0 or node.is_terminal():
        return node.evaluate()
    
    if is_maximizing:
        value = float('-inf')
        for move in node.generate_moves():
            new_state = node.apply_move(move)
            child = GameNode(new_state, 3 - node.player, node, move)
            value = max(value, minimax_dls(child, depth - 1, False))
        return value
    else:
        value = float('inf')
        for move in node.generate_moves():
            new_state = node.apply_move(move)
            child = GameNode(new_state, 3 - node.player, node, move)
            value = min(value, minimax_dls(child, depth - 1, True))
        return value
```

---

## Industry Applications

### 1. Robotics Path Planning

```python
class RobotNode:
    def __init__(self, position, orientation, obstacles, parent=None):
        self.position = position
        self.orientation = orientation
        self.obstacles = obstacles
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
    
    def get_valid_moves(self):
        """Generate collision-free moves"""
        moves = []
        step_size = 1.0  # Movement step size
        
        # Forward, backward, left, right movements
        directions = [
            (step_size, 0),   # Forward
            (-step_size, 0),  # Backward
            (0, step_size),   # Left
            (0, -step_size)    # Right
        ]
        
        for dx, dy in directions:
            new_pos = (self.position[0] + dx, self.position[1] + dy)
            if self.is_valid_position(new_pos):
                moves.append(RobotNode(new_pos, self.orientation, 
                                     self.obstacles, self))
        
        return moves
    
    def is_valid_position(self, position):
        """Check if position is valid (no collision)"""
        x, y = position
        for obs_x, obs_y, obs_size in self.obstacles:
            if (abs(x - obs_x) < obs_size and 
                abs(y - obs_y) < obs_size):
                return False
        return True
    
    def distance_to_goal(self, goal):
        """Calculate Manhattan distance to goal"""
        return (abs(self.position[0] - goal[0]) + 
                abs(self.position[1] - goal[1]))

def robot_path_planning(start, goal, obstacles, depth_limit):
    """
    Plan robot path using DLS
    """
    start_node = RobotNode(start, 0, obstacles)
    
    def dls_with_heuristic(node, goal, limit, depth=0):
        if node.distance_to_goal(goal) < 0.5:  # Reached goal
            path = []
            current = node
            while current:
                path.append(current.position)
                current = current.parent
            return list(reversed(path))
        
        if depth == limit:
            return None
        
        # Sort moves by heuristic (closer to goal first)
        moves = node.get_valid_moves()
        moves.sort(key=lambda m: m.distance_to_goal(goal))
        
        for move in moves:
            result = dls_with_heuristic(move, goal, limit, depth + 1)
            if result is not None:
                return result
        
        return None
    
    return dls_with_heuristic(start_node, goal, depth_limit)
```

### 2. Network Routing

```python
class NetworkNode:
    def __init__(self, node_id, network_graph, parent=None):
        self.node_id = node_id
        self.network = network_graph
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.path_cost = 0 if parent is None else parent.path_cost + 1
    
    def get_neighbors(self):
        """Get connected nodes with edge costs"""
        return self.network.get(self.node_id, [])
    
    def is_goal(self, target_id):
        return self.node_id == target_id

def network_routing_dls(network, source, target, max_hops):
    """
    Find routing path with maximum hop limit
    """
    start_node = NetworkNode(source, network)
    
    def dls_routing(node, target, limit, depth=0):
        if node.is_goal(target):
            path = []
            current = node
            while current:
                path.append(current.node_id)
                current = current.parent
            return list(reversed(path))
        
        if depth == limit:
            return None
        
        for neighbor_id, cost in node.get_neighbors():
            neighbor = NetworkNode(neighbor_id, network, node)
            result = dls_routing(neighbor, target, limit, depth + 1)
            if result is not None:
                return result
        
        return None
    
    return dls_routing(start_node, target, max_hops)

# Example network topology
network_topology = {
    'A': [('B', 1), ('C', 2)],
    'B': [('A', 1), ('D', 3), ('E', 1)],
    'C': [('A', 2), ('F', 4)],
    'D': [('B', 3), ('G', 2)],
    'E': [('B', 1), ('G', 5)],
    'F': [('C', 4), ('G', 1)],
    'G': [('D', 2), ('E', 5), ('F', 1)]
}
```

---

## Performance Analysis

### 1. Benchmarking Framework

```python
import time
import random
from collections import defaultdict

class DLSBenchmark:
    def __init__(self):
        self.results = defaultdict(list)
    
    def generate_random_tree(self, branching_factor, max_depth, current_depth=0):
        """Generate random tree for testing"""
        if current_depth >= max_depth:
            return None
        
        node_id = random.randint(1000, 9999)
        children = []
        
        for _ in range(branching_factor):
            child = self.generate_random_tree(branching_factor, max_depth, current_depth + 1)
            if child:
                children.append(child)
        
        return {
            'id': node_id,
            'children': children,
            'depth': current_depth
        }
    
    def benchmark_dls(self, tree, goal_id, depth_limits):
        """Benchmark DLS performance across different depth limits"""
        for limit in depth_limits:
            start_time = time.time()
            nodes_explored = 0
            
            def count_nodes(node):
                nonlocal nodes_explored
                if node is None:
                    return None
                nodes_explored += 1
                return node
            
            result = depth_limited_search(tree, goal_id, limit)
            end_time = time.time()
            
            self.results[limit].append({
                'time': end_time - start_time,
                'nodes_explored': nodes_explored,
                'found': result is not None,
                'depth_limit': limit
            })
    
    def analyze_results(self):
        """Analyze benchmark results"""
        analysis = {}
        
        for limit, runs in self.results.items():
            times = [run['time'] for run in runs]
            nodes = [run['nodes_explored'] for run in runs]
            success_rate = sum(1 for run in runs if run['found']) / len(runs)
            
            analysis[limit] = {
                'avg_time': sum(times) / len(times),
                'avg_nodes': sum(nodes) / len(nodes),
                'success_rate': success_rate,
                'time_std': self.std_dev(times),
                'nodes_std': self.std_dev(nodes)
            }
        
        return analysis
    
    def std_dev(self, values):
        """Calculate standard deviation"""
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
```

### 2. Memory Profiling

```python
import sys
import tracemalloc

class DLSMemoryProfiler:
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
    
    def profile_dls_memory(self, start_node, goal, depth_limit):
        """Profile memory usage during DLS execution"""
        tracemalloc.start()
        
        # Track memory at different depths
        memory_by_depth = {}
        
        def dls_with_memory_tracking(node, goal, limit, depth=0):
            # Record current memory usage
            current, peak = tracemalloc.get_traced_memory()
            memory_by_depth[depth] = current
            
            if goal_test(node):
                return [node]
            
            if depth == limit:
                return None
            
            for child in expand_node(node):
                result = dls_with_memory_tracking(child, goal, limit, depth + 1)
                if result is not None:
                    return [node] + result
            
            return None
        
        result = dls_with_memory_tracking(start_node, goal, depth_limit)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'result': result,
            'peak_memory': peak,
            'memory_by_depth': memory_by_depth,
            'max_depth_memory': max(memory_by_depth.values()) if memory_by_depth else 0
        }
```

---

## Advanced Topics

### 1. Iterative Deepening DLS

```python
def iterative_deepening_dls(start, goal, max_depth):
    """
    Iterative Deepening Depth Limited Search
    Combines benefits of DLS with completeness of BFS
    """
    for depth in range(max_depth + 1):
        result = depth_limited_search(start, goal, depth)
        if result is not None:
            return result, depth
    
    return None, max_depth

def bidirectional_dls(start, goal, max_depth):
    """
    Bidirectional Depth Limited Search
    Search from both start and goal simultaneously
    """
    from collections import deque
    
    # Forward search
    forward_frontier = [(start, 0)]
    forward_visited = {start: None}
    
    # Backward search
    backward_frontier = [(goal, 0)]
    backward_visited = {goal: None}
    
    for depth in range(max_depth // 2 + 1):
        # Expand forward frontier
        new_forward = []
        for node, current_depth in forward_frontier:
            if current_depth < depth:
                for child in expand_node(node):
                    if child not in forward_visited:
                        forward_visited[child] = node
                        new_forward.append((child, current_depth + 1))
        
        forward_frontier = new_forward
        
        # Check for intersection
        for node in forward_visited:
            if node in backward_visited:
                # Reconstruct path
                path_forward = []
                current = node
                while current is not None:
                    path_forward.append(current)
                    current = forward_visited[current]
                
                path_backward = []
                current = node
                while current is not None:
                    path_backward.append(current)
                    current = backward_visited[current]
                
                return list(reversed(path_forward[:-1])) + path_backward
        
        # Expand backward frontier
        new_backward = []
        for node, current_depth in backward_frontier:
            if current_depth < depth:
                for child in expand_node(node):
                    if child not in backward_visited:
                        backward_visited[child] = node
                        new_backward.append((child, current_depth + 1))
        
        backward_frontier = new_backward
    
    return None
```

### 2. Parallel DLS Implementation

```python
import concurrent.futures
import threading

class ParallelDLS:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.solution_found = threading.Event()
        self.solution_lock = threading.Lock()
        self.solution = None
    
    def parallel_dls(self, start, goal, depth_limit):
        """
        Parallel Depth Limited Search using multiple threads
        """
        children = expand_node(start)
        
        if not children:
            return None
        
        # Distribute children among workers
        chunk_size = max(1, len(children) // self.num_workers)
        chunks = [children[i:i + chunk_size] 
                 for i in range(0, len(children), chunk_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for chunk in chunks:
                future = executor.submit(
                    self._search_chunk, chunk, goal, depth_limit - 1
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                if self.solution_found.is_set():
                    break
                
                result = future.result()
                if result is not None:
                    with self.solution_lock:
                        if not self.solution_found.is_set():
                            self.solution = [start] + result
                            self.solution_found.set()
        
        return self.solution
    
    def _search_chunk(self, chunk, goal, depth_limit):
        """Search a chunk of nodes"""
        for child in chunk:
            if self.solution_found.is_set():
                break
            
            result = depth_limited_search(child, goal, depth_limit)
            if result is not None:
                with self.solution_lock:
                    if not self.solution_found.is_set():
                        self.solution = result
                        self.solution_found.set()
                        return result
        
        return None
```

### 3. Dynamic Depth Limit Adjustment

```python
class AdaptiveDLS:
    def __init__(self, initial_depth=5, max_depth=20):
        self.initial_depth = initial_depth
        self.max_depth = max_depth
        self.success_history = []
        self.failure_history = []
    
    def adaptive_dls(self, start, goal):
        """
        Adaptive DLS with dynamic depth limit adjustment
        """
        current_depth = self.initial_depth
        
        while current_depth <= self.max_depth:
            start_time = time.time()
            result = depth_limited_search(start, goal, current_depth)
            end_time = time.time()
            
            if result is not None:
                self.success_history.append({
                    'depth': current_depth,
                    'time': end_time - start_time,
                    'solution_length': len(result)
                })
                return result
            else:
                self.failure_history.append({
                    'depth': current_depth,
                    'time': end_time - start_time
                })
                
                # Adjust depth based on performance
                current_depth = self._adjust_depth(current_depth)
        
        return None
    
    def _adjust_depth(self, current_depth):
        """Dynamically adjust depth limit"""
        if len(self.failure_history) < 3:
            return current_depth + 1
        
        # Check if we're consistently failing at current depth
        recent_failures = self.failure_history[-3:]
        avg_time = sum(f['time'] for f in recent_failures) / 3
        
        # If search is taking too long, increase depth more aggressively
        if avg_time > 5.0:  # 5 seconds threshold
            return min(current_depth + 3, self.max_depth)
        else:
            return min(current_depth + 1, self.max_depth)
```

---

## Lab Exercises

### Exercise 1: Basic DLS Implementation

**Objective**: Implement and test basic DLS algorithm

**Tasks**:
1. Implement the `depth_limited_search` function
2. Create a simple tree structure for testing
3. Test with different depth limits
4. Analyze results and document findings

**Expected Outcomes**:
- Understanding of DLS behavior
- Recognition of cutoff conditions
- Analysis of depth limit impact

### Exercise 2: Maze Solving

**Objective**: Apply DLS to solve maze navigation problems

**Tasks**:
1. Implement maze representation
2. Create maze generation algorithm
3. Apply DLS to find paths
4. Compare performance with different depth limits

**Expected Outcomes**:
- Practical DLS application
- Performance optimization techniques
- Real-world problem solving

### Exercise 3: Performance Analysis

**Objective**: Analyze DLS performance characteristics

**Tasks**:
1. Create benchmarking framework
2. Test with varying branching factors
3. Measure time and space complexity
4. Generate performance graphs

**Expected Outcomes**:
- Empirical complexity analysis
- Performance optimization insights
- Comparative analysis with other algorithms

### Exercise 4: Advanced Applications

**Objective**: Implement advanced DLS variants

**Tasks**:
1. Implement Iterative Deepening DLS
2. Create parallel DLS implementation
3. Develop adaptive depth adjustment
4. Compare performance metrics

**Expected Outcomes**:
- Advanced algorithm understanding
- Parallel programming skills
- Adaptive algorithm design

---

## Best Practices

### 1. Algorithm Design

- **Modular Design**: Separate search logic from problem representation
- **Efficient Data Structures**: Use appropriate data structures for node representation
- **Memory Management**: Implement proper memory cleanup for large searches
- **Error Handling**: Handle edge cases and invalid inputs gracefully

### 2. Performance Optimization

- **Heuristic Ordering**: Order child nodes by heuristic estimates
- **Pruning**: Implement domain-specific pruning when possible
- **Memoization**: Cache results of expensive computations
- **Parallelization**: Utilize multiple cores for independent branches

### 3. Code Quality

- **Documentation**: Provide clear documentation for all functions
- **Testing**: Implement comprehensive unit and integration tests
- **Logging**: Add appropriate logging for debugging and monitoring
- **Profiling**: Profile code to identify performance bottlenecks

### 4. Industry Considerations

- **Scalability**: Design for large-scale problems
- **Maintainability**: Write clean, maintainable code
- **Extensibility**: Design for easy extension and modification
- **Robustness**: Handle unexpected inputs and edge cases

---

## References

### Academic Papers
1. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
2. Pearl, J. (1984). *Heuristics: Intelligent Search Strategies for Computer Problem Solving*. Addison-Wesley.
3. Korf, R. E. (1985). "Depth-First Iterative Deepening: An Optimal Admissible Tree Search". *Artificial Intelligence*, 27(1), 97-109.

### Technical Resources
1. Wikipedia contributors. (2023). "Depth-limited search". In *Wikipedia, The Free Encyclopedia*.
2. Stanford University. (2022). *CS221: Artificial Intelligence: Principles and Techniques*.
3. MIT OpenCourseWare. (2021). *6.034: Artificial Intelligence*.

### Industry Applications
1. LaValle, S. M. (2006). *Planning Algorithms*. Cambridge University Press.
2. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
3. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

---

## Appendix

### A. Common Pitfalls and Solutions

| Pitfall | Description | Solution |
|---------|-------------|----------|
| Infinite recursion | No depth limit or incorrect limit checking | Verify depth limit implementation |
| Memory leaks | Improper cleanup of node references | Implement proper garbage collection |
| Poor performance | Inefficient node expansion | Optimize data structures and algorithms |
| Incorrect results | Goal test or neighbor generation errors | Thoroughly test individual components |

### B. Performance Metrics

| Metric | Description | Typical Values |
|--------|-------------|----------------|
| Time complexity | O(b^d) where b is branching factor, d is depth | Exponential growth |
| Space complexity | O(b×d) | Linear in depth |
| Success rate | Probability of finding solution | Depends on depth limit |
| Average path length | Expected solution length | Varies by problem domain |

### C. Comparison with Other Algorithms

| Algorithm | Time | Space | Complete | Optimal |
|-----------|------|-------|----------|---------|
| BFS | O(b^d) | O(b^d) | Yes | Yes |
| DFS | O(b^m) | O(bm) | No | No |
| DLS | O(b^d) | O(bd) | If d ≤ limit | No |
| IDS | O(b^d) | O(bd) | Yes | Yes |
| A* | O(b^d) | O(b^d) | Yes | Yes (with admissible heuristic) |

---
### Author: 

**Khansa Younas**

*This lab manual provides a comprehensive foundation for understanding and implementing Depth Limited Search in professional and industrial contexts. The examples and exercises are designed to build practical skills while maintaining theoretical rigor.*

