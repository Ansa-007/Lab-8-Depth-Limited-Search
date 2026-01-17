"""
Depth Limited Search Implementation
Professional and Industry Level Implementation
"""

import time
import sys
import threading
import concurrent.futures
from collections import defaultdict, deque
from typing import Any, List, Optional, Dict, Tuple, Callable
import random
import tracemalloc


class Node:
    """Base class for search nodes"""
    
    def __init__(self, state: Any, parent: Optional['Node'] = None, depth: int = 0):
        self.state = state
        self.parent = parent
        self.depth = depth
        self.children = []
    
    def __eq__(self, other):
        return self.state == other.state if isinstance(other, Node) else False
    
    def __hash__(self):
        return hash(self.state)
    
    def get_path(self) -> List[Any]:
        """Get path from root to this node"""
        path = []
        current = self
        while current:
            path.append(current.state)
            current = current.parent
        return list(reversed(path))


class DepthLimitedSearch:
    """Professional Depth Limited Search Implementation"""
    
    def __init__(self, 
                 expand_func: Callable[[Node], List[Node]],
                 goal_test_func: Callable[[Node], bool],
                 max_depth: int = 10):
        self.expand = expand_func
        self.goal_test = goal_test_func
        self.max_depth = max_depth
        self.nodes_explored = 0
        self.cutoff_occurred = False
    
    def search(self, initial_state: Any) -> Optional[List[Any]]:
        """
        Perform depth limited search
        
        Args:
            initial_state: Initial state to search from
            
        Returns:
            List of states representing path to goal, or None if not found
        """
        self.nodes_explored = 0
        self.cutoff_occurred = False
        
        start_node = Node(initial_state)
        result = self._dls_recursive(start_node)
        
        if result is not None:
            return result.get_path()
        elif self.cutoff_occurred:
            return None  # Cutoff occurred
        else:
            return None  # No solution within depth limit
    
    def _dls_recursive(self, node: Node) -> Optional[Node]:
        """Recursive DLS implementation"""
        self.nodes_explored += 1
        
        if self.goal_test(node):
            return node
        
        if node.depth >= self.max_depth:
            self.cutoff_occurred = True
            return None
        
        for child in self.expand(node):
            result = self._dls_recursive(child)
            if result is not None:
                return result
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            'nodes_explored': self.nodes_explored,
            'max_depth': self.max_depth,
            'cutoff_occurred': self.cutoff_occurred
        }


class IterativeDeepeningDLS:
    """Iterative Deepening Depth Limited Search"""
    
    def __init__(self, 
                 expand_func: Callable[[Node], List[Node]],
                 goal_test_func: Callable[[Node], bool],
                 max_depth: int = 20):
        self.expand = expand_func
        self.goal_test = goal_test_func
        self.max_depth = max_depth
        self.total_nodes_explored = 0
    
    def search(self, initial_state: Any) -> Tuple[Optional[List[Any]], int]:
        """
        Perform iterative deepening search
        
        Returns:
            Tuple of (path to goal, depth at which solution was found)
        """
        for depth in range(self.max_depth + 1):
            dls = DepthLimitedSearch(self.expand, self.goal_test, depth)
            result = dls.search(initial_state)
            
            self.total_nodes_explored += dls.nodes_explored
            
            if result is not None:
                return result, depth
        
        return None, self.max_depth


class ParallelDLS:
    """Parallel Depth Limited Search Implementation"""
    
    def __init__(self, 
                 expand_func: Callable[[Node], List[Node]],
                 goal_test_func: Callable[[Node], bool],
                 max_depth: int = 10,
                 num_workers: int = 4):
        self.expand = expand_func
        self.goal_test = goal_test_func
        self.max_depth = max_depth
        self.num_workers = num_workers
        self.solution_found = threading.Event()
        self.solution_lock = threading.Lock()
        self.solution = None
    
    def search(self, initial_state: Any) -> Optional[List[Any]]:
        """Perform parallel DLS"""
        start_node = Node(initial_state)
        children = self.expand(start_node)
        
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
                    self._search_chunk, chunk, self.max_depth - 1
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                if self.solution_found.is_set():
                    break
                
                result = future.result()
                if result is not None:
                    with self.solution_lock:
                        if not self.solution_found.is_set():
                            self.solution = [initial_state] + result
                            self.solution_found.set()
        
        return self.solution
    
    def _search_chunk(self, chunk: List[Node], depth_limit: int) -> Optional[List[Any]]:
        """Search a chunk of nodes"""
        for child in chunk:
            if self.solution_found.is_set():
                break
            
            dls = DepthLimitedSearch(self.expand, self.goal_test, depth_limit)
            result = dls.search(child.state)
            
            if result is not None:
                with self.solution_lock:
                    if not self.solution_found.is_set():
                        self.solution = result
                        self.solution_found.set()
                        return result
        
        return None


class AdaptiveDLS:
    """Adaptive Depth Limited Search with dynamic depth adjustment"""
    
    def __init__(self, 
                 expand_func: Callable[[Node], List[Node]],
                 goal_test_func: Callable[[Node], bool],
                 initial_depth: int = 5,
                 max_depth: int = 20):
        self.expand = expand_func
        self.goal_test = goal_test_func
        self.initial_depth = initial_depth
        self.max_depth = max_depth
        self.success_history = []
        self.failure_history = []
    
    def search(self, initial_state: Any) -> Optional[List[Any]]:
        """Perform adaptive DLS"""
        current_depth = self.initial_depth
        
        while current_depth <= self.max_depth:
            start_time = time.time()
            dls = DepthLimitedSearch(self.expand, self.goal_test, current_depth)
            result = dls.search(initial_state)
            end_time = time.time()
            
            if result is not None:
                self.success_history.append({
                    'depth': current_depth,
                    'time': end_time - start_time,
                    'solution_length': len(result),
                    'nodes_explored': dls.nodes_explored
                })
                return result
            else:
                self.failure_history.append({
                    'depth': current_depth,
                    'time': end_time - start_time,
                    'nodes_explored': dls.nodes_explored
                })
                
                # Adjust depth based on performance
                current_depth = self._adjust_depth(current_depth)
        
        return None
    
    def _adjust_depth(self, current_depth: int) -> int:
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


class DLSBenchmark:
    """Benchmarking framework for DLS variants"""
    
    def __init__(self):
        self.results = defaultdict(list)
    
    def generate_random_tree(self, branching_factor: int, max_depth: int, 
                           current_depth: int = 0) -> Optional[Dict]:
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
    
    def benchmark_algorithm(self, algorithm_class, initial_state: Any, 
                           goal_state: Any, depth_limits: List[int],
                           **kwargs) -> Dict:
        """Benchmark a specific algorithm"""
        results = {}
        
        for limit in depth_limits:
            # Create expand and goal test functions for tree
            def expand_func(node):
                if isinstance(node.state, dict):
                    children = node.state.get('children', [])
                    return [Node(child, node, node.depth + 1) for child in children]
                return []
            
            def goal_test_func(node):
                return (isinstance(node.state, dict) and 
                       node.state.get('id') == goal_state)
            
            # Initialize algorithm
            if algorithm_class == DepthLimitedSearch:
                algorithm = algorithm_class(expand_func, goal_test_func, limit)
                start_time = time.time()
                result = algorithm.search(initial_state)
                end_time = time.time()
                
                results[limit] = {
                    'time': end_time - start_time,
                    'nodes_explored': algorithm.nodes_explored,
                    'found': result is not None,
                    'solution_length': len(result) if result else 0,
                    'cutoff_occurred': algorithm.cutoff_occurred
                }
            
            elif algorithm_class == IterativeDeepeningDLS:
                algorithm = algorithm_class(expand_func, goal_test_func, limit)
                start_time = time.time()
                result, found_depth = algorithm.search(initial_state)
                end_time = time.time()
                
                results[limit] = {
                    'time': end_time - start_time,
                    'nodes_explored': algorithm.total_nodes_explored,
                    'found': result is not None,
                    'solution_length': len(result) if result else 0,
                    'found_depth': found_depth
                }
        
        return results
    
    def compare_algorithms(self, algorithms: List, initial_state: Any,
                          goal_state: Any, depth_limits: List[int]) -> Dict:
        """Compare multiple algorithms"""
        comparison = {}
        
        for algorithm_class in algorithms:
            algorithm_name = algorithm_class.__name__
            comparison[algorithm_name] = self.benchmark_algorithm(
                algorithm_class, initial_state, goal_state, depth_limits
            )
        
        return comparison


class MemoryProfiler:
    """Memory profiling for DLS algorithms"""
    
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
    
    def profile_dls_memory(self, expand_func: Callable[[Node], List[Node]],
                          goal_test_func: Callable[[Node], bool],
                          initial_state: Any, depth_limit: int) -> Dict:
        """Profile memory usage during DLS execution"""
        tracemalloc.start()
        
        # Track memory at different depths
        memory_by_depth = {}
        nodes_explored = 0
        
        def dls_with_memory_tracking(node: Node, depth: int = 0) -> Optional[Node]:
            nonlocal nodes_explored
            
            # Record current memory usage
            current, peak = tracemalloc.get_traced_memory()
            memory_by_depth[depth] = current
            nodes_explored += 1
            
            if goal_test_func(node):
                return node
            
            if depth >= depth_limit:
                return None
            
            for child in expand_func(node):
                result = dls_with_memory_tracking(child, depth + 1)
                if result is not None:
                    return result
            
            return None
        
        start_node = Node(initial_state)
        result = dls_with_memory_tracking(start_node)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'result': result.get_path() if result else None,
            'peak_memory': peak,
            'current_memory': current,
            'memory_by_depth': memory_by_depth,
            'max_depth_memory': max(memory_by_depth.values()) if memory_by_depth else 0,
            'nodes_explored': nodes_explored
        }


# Utility functions for common use cases

def create_maze_node(position: Tuple[int, int], maze: List[List[int]], 
                    parent: Optional[Node] = None) -> Node:
    """Create a maze navigation node"""
    return Node(position, parent)


def expand_maze_node(node: Node, maze: List[List[int]]) -> List[Node]:
    """Expand maze node with valid moves"""
    x, y = node.state
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    neighbors = []
    
    for dx, dy in moves:
        new_x, new_y = x + dx, y + dy
        if (0 <= new_x < len(maze[0]) and 
            0 <= new_y < len(maze) and 
            maze[new_y][new_x] != 1):  # 1 represents wall
            neighbors.append(Node((new_x, new_y), node, node.depth + 1))
    
    return neighbors


def maze_goal_test(goal_position: Tuple[int, int]) -> Callable[[Node], bool]:
    """Create maze goal test function"""
    def goal_test(node: Node) -> bool:
        return node.state == goal_position
    return goal_test


def solve_maze_dls(maze: List[List[int]], start: Tuple[int, int], 
                  goal: Tuple[int, int], depth_limit: int) -> Optional[List[Tuple[int, int]]]:
    """Solve maze using Depth Limited Search"""
    expand_func = lambda node: expand_maze_node(node, maze)
    goal_test_func = maze_goal_test(goal)
    
    dls = DepthLimitedSearch(expand_func, goal_test_func, depth_limit)
    return dls.search(start)


# Example usage and testing
if __name__ == "__main__":
    # Example maze (0 = path, 1 = wall, 2 = start, 3 = goal)
    maze = [
        [2, 0, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 3, 1]
    ]
    
    # Find start and goal positions
    start_pos = None
    goal_pos = None
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if maze[y][x] == 2:
                start_pos = (x, y)
            elif maze[y][x] == 3:
                goal_pos = (x, y)
    
    if start_pos and goal_pos:
        print(f"Solving maze from {start_pos} to {goal_pos}")
        
        # Try different depth limits
        for depth in range(5, 16):
            solution = solve_maze_dls(maze, start_pos, goal_pos, depth)
            if solution:
                print(f"Solution found at depth {depth}: {solution}")
                print(f"Path length: {len(solution)}")
                break
        else:
            print("No solution found within depth limit")
    
    # Benchmark example
    print("\nRunning benchmark...")
    benchmark = DLSBenchmark()
    
    # Generate test tree
    test_tree = benchmark.generate_random_tree(3, 4)
    
    # Benchmark different algorithms
    algorithms = [DepthLimitedSearch, IterativeDeepeningDLS]
    results = benchmark.compare_algorithms(algorithms, test_tree, 5000, [3, 4, 5])
    
    print("Benchmark Results:")
    for alg_name, alg_results in results.items():
        print(f"\n{alg_name}:")
        for depth, stats in alg_results.items():
            print(f"  Depth {depth}: Time={stats['time']:.4f}s, "
                  f"Nodes={stats['nodes_explored']}, "
                  f"Found={stats['found']}")
