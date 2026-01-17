"""
Performance Analysis and Optimization for Depth Limited Search
Professional and Industry Level Performance Tools
"""

import time
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Callable
import threading
import concurrent.futures
from collections import defaultdict
import json
import csv
from dataclasses import dataclass
from enum import Enum


class PerformanceMetric(Enum):
    """Performance metrics for analysis"""
    TIME = "time"
    MEMORY = "memory"
    NODES_EXPLORED = "nodes_explored"
    SOLUTION_LENGTH = "solution_length"
    SUCCESS_RATE = "success_rate"
    CPU_USAGE = "cpu_usage"


@dataclass
class PerformanceResult:
    """Container for performance measurement results"""
    algorithm: str
    depth_limit: int
    metric: PerformanceMetric
    value: float
    timestamp: float
    additional_info: Dict[str, Any] = None


class PerformanceAnalyzer:
    """Comprehensive performance analysis for DLS algorithms"""
    
    def __init__(self, output_dir: str = "performance_results"):
        self.output_dir = output_dir
        self.results = []
        self.process = psutil.Process(os.getpid())
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def measure_execution_time(self, func: Callable, *args, **kwargs) -> Tuple[float, Any]:
        """Measure execution time of a function"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time, result
    
    def measure_memory_usage(self, func: Callable, *args, **kwargs) -> Tuple[float, Any]:
        """Measure memory usage of a function"""
        import tracemalloc
        
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return peak / (1024 * 1024), result  # Convert to MB
    
    def measure_cpu_usage(self, func: Callable, *args, **kwargs) -> Tuple[float, Any]:
        """Measure CPU usage during function execution"""
        # Start CPU monitoring
        cpu_usage = []
        monitoring = True
        
        def monitor_cpu():
            while monitoring:
                cpu_usage.append(psutil.cpu_percent(interval=0.1))
        
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        monitoring = False
        monitor_thread.join()
        
        avg_cpu = np.mean(cpu_usage) if cpu_usage else 0
        return avg_cpu, result
    
    def comprehensive_benchmark(self, algorithms: List[Dict], 
                               test_cases: List[Dict], 
                               depth_limits: List[int]) -> List[PerformanceResult]:
        """Run comprehensive benchmark across multiple algorithms and test cases"""
        results = []
        
        for algorithm_config in algorithms:
            algorithm_name = algorithm_config['name']
            algorithm_class = algorithm_config['class']
            algorithm_params = algorithm_config.get('params', {})
            
            for test_case in test_cases:
                test_name = test_case['name']
                initial_state = test_case['initial_state']
                goal_state = test_case['goal_state']
                
                # Create expand and goal test functions
                expand_func = test_case['expand_func']
                goal_test_func = test_case['goal_test_func']
                
                for depth_limit in depth_limits:
                    print(f"Testing {algorithm_name} on {test_name} at depth {depth_limit}")
                    
                    # Initialize algorithm
                    if algorithm_class.__name__ == 'DepthLimitedSearch':
                        algorithm = algorithm_class(expand_func, goal_test_func, depth_limit)
                    elif algorithm_class.__name__ == 'IterativeDeepeningDLS':
                        algorithm = algorithm_class(expand_func, goal_test_func, depth_limit)
                    else:
                        algorithm = algorithm_class(expand_func, goal_test_func, depth_limit, **algorithm_params)
                    
                    # Measure execution time
                    exec_time, result = self.measure_execution_time(algorithm.search, initial_state)
                    
                    # Measure memory usage
                    memory_usage, _ = self.measure_memory_usage(algorithm.search, initial_state)
                    
                    # Measure CPU usage
                    cpu_usage, _ = self.measure_cpu_usage(algorithm.search, initial_state)
                    
                    # Get algorithm-specific statistics
                    if hasattr(algorithm, 'get_statistics'):
                        stats = algorithm.get_statistics()
                        nodes_explored = stats.get('nodes_explored', 0)
                    elif hasattr(algorithm, 'total_nodes_explored'):
                        nodes_explored = algorithm.total_nodes_explored
                    else:
                        nodes_explored = 0
                    
                    # Create performance results
                    results.append(PerformanceResult(
                        algorithm=algorithm_name,
                        depth_limit=depth_limit,
                        metric=PerformanceMetric.TIME,
                        value=exec_time,
                        timestamp=time.time(),
                        additional_info={'test_case': test_name, 'found': result is not None}
                    ))
                    
                    results.append(PerformanceResult(
                        algorithm=algorithm_name,
                        depth_limit=depth_limit,
                        metric=PerformanceMetric.MEMORY,
                        value=memory_usage,
                        timestamp=time.time(),
                        additional_info={'test_case': test_name}
                    ))
                    
                    results.append(PerformanceResult(
                        algorithm=algorithm_name,
                        depth_limit=depth_limit,
                        metric=PerformanceMetric.NODES_EXPLORED,
                        value=nodes_explored,
                        timestamp=time.time(),
                        additional_info={'test_case': test_name}
                    ))
                    
                    results.append(PerformanceResult(
                        algorithm=algorithm_name,
                        depth_limit=depth_limit,
                        metric=PerformanceMetric.CPU_USAGE,
                        value=cpu_usage,
                        timestamp=time.time(),
                        additional_info={'test_case': test_name}
                    ))
                    
                    if result is not None:
                        results.append(PerformanceResult(
                            algorithm=algorithm_name,
                            depth_limit=depth_limit,
                            metric=PerformanceMetric.SOLUTION_LENGTH,
                            value=len(result),
                            timestamp=time.time(),
                            additional_info={'test_case': test_name}
                        ))
        
        self.results.extend(results)
        return results
    
    def analyze_scalability(self, algorithm_class, test_case_generator, 
                          depth_range: range, branching_factors: List[int]) -> Dict:
        """Analyze algorithm scalability across different parameters"""
        scalability_results = defaultdict(list)
        
        for branching_factor in branching_factors:
            for depth in depth_range:
                # Generate test case with specific parameters
                test_case = test_case_generator(branching_factor, depth)
                
                # Initialize algorithm
                expand_func = test_case['expand_func']
                goal_test_func = test_case['goal_test_func']
                initial_state = test_case['initial_state']
                
                algorithm = algorithm_class(expand_func, goal_test_func, depth)
                
                # Measure performance
                exec_time, result = self.measure_execution_time(algorithm.search, initial_state)
                memory_usage, _ = self.measure_memory_usage(algorithm.search, initial_state)
                
                scalability_results['branching_factor'].append(branching_factor)
                scalability_results['depth'].append(depth)
                scalability_results['time'].append(exec_time)
                scalability_results['memory'].append(memory_usage)
                scalability_results['nodes_explored'].append(algorithm.nodes_explored)
                scalability_results['found'].append(result is not None)
        
        return dict(scalability_results)
    
    def generate_performance_report(self, results: List[PerformanceResult] = None) -> str:
        """Generate comprehensive performance report"""
        if results is None:
            results = self.results
        
        # Convert to DataFrame for analysis
        data = []
        for result in results:
            row = {
                'algorithm': result.algorithm,
                'depth_limit': result.depth_limit,
                'metric': result.metric.value,
                'value': result.value,
                'timestamp': result.timestamp
            }
            if result.additional_info:
                row.update(result.additional_info)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Generate report
        report = []
        report.append("# Performance Analysis Report\n")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary statistics
        report.append("## Summary Statistics\n")
        for metric in PerformanceMetric:
            metric_data = df[df['metric'] == metric.value]
            if not metric_data.empty:
                report.append(f"\n### {metric.value.replace('_', ' ').title()}\n")
                report.append(f"- Mean: {metric_data['value'].mean():.4f}")
                report.append(f"- Std Dev: {metric_data['value'].std():.4f}")
                report.append(f"- Min: {metric_data['value'].min():.4f}")
                report.append(f"- Max: {metric_data['value'].max():.4f}")
        
        # Algorithm comparison
        report.append("\n## Algorithm Comparison\n")
        for algorithm in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == algorithm]
            report.append(f"\n### {algorithm}\n")
            
            for metric in PerformanceMetric:
                metric_data = alg_data[alg_data['metric'] == metric.value]
                if not metric_data.empty:
                    avg_value = metric_data['value'].mean()
                    report.append(f"- Avg {metric.value.replace('_', ' ')}: {avg_value:.4f}")
        
        # Depth limit analysis
        report.append("\n## Depth Limit Analysis\n")
        for depth in sorted(df['depth_limit'].unique()):
            depth_data = df[df['depth_limit'] == depth]
            success_rate = depth_data[depth_data['metric'] == 'found']['value'].mean() if 'found' in depth_data['metric'].values else 0
            avg_time = depth_data[depth_data['metric'] == 'time']['value'].mean() if 'time' in depth_data['metric'].values else 0
            
            report.append(f"\n### Depth {depth}\n")
            report.append(f"- Success Rate: {success_rate:.2%}")
            report.append(f"- Avg Time: {avg_time:.4f}s")
        
        return "\n".join(report)
    
    def plot_performance_comparison(self, results: List[PerformanceResult] = None, 
                                  metric: PerformanceMetric = PerformanceMetric.TIME):
        """Plot performance comparison between algorithms"""
        if results is None:
            results = self.results
        
        # Filter results by metric
        metric_results = [r for r in results if r.metric == metric]
        
        if not metric_results:
            print(f"No data for metric {metric.value}")
            return
        
        # Group by algorithm and depth limit
        data = defaultdict(lambda: defaultdict(list))
        for result in metric_results:
            data[result.algorithm][result.depth_limit].append(result.value)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        for algorithm, depth_data in data.items():
            depths = sorted(depth_data.keys())
            values = [np.mean(depth_data[d]) for d in depths]
            errors = [np.std(depth_data[d]) for d in depths]
            
            plt.errorbar(depths, values, yerr=errors, label=algorithm, marker='o', capsize=5)
        
        plt.xlabel('Depth Limit')
        plt.ylabel(metric.value.replace('_', ' ').title())
        plt.title(f'Performance Comparison: {metric.value.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_filename = f"{self.output_dir}/performance_{metric.value}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plot saved to {plot_filename}")
    
    def export_results(self, results: List[PerformanceResult] = None, 
                      format: str = 'csv'):
        """Export results to file"""
        if results is None:
            results = self.results
        
        # Convert to list of dictionaries
        data = []
        for result in results:
            row = {
                'algorithm': result.algorithm,
                'depth_limit': result.depth_limit,
                'metric': result.metric.value,
                'value': result.value,
                'timestamp': result.timestamp
            }
            if result.additional_info:
                for key, value in result.additional_info.items():
                    row[f'info_{key}'] = value
            data.append(row)
        
        # Export based on format
        timestamp = int(time.time())
        
        if format.lower() == 'csv':
            filename = f"{self.output_dir}/performance_results_{timestamp}.csv"
            with open(filename, 'w', newline='') as csvfile:
                if data:
                    fieldnames = data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)
        
        elif format.lower() == 'json':
            filename = f"{self.output_dir}/performance_results_{timestamp}.json"
            with open(filename, 'w') as jsonfile:
                json.dump(data, jsonfile, indent=2)
        
        print(f"Results exported to {filename}")
    
    def clear_results(self):
        """Clear stored results"""
        self.results = []


class OptimizationProfiler:
    """Profile optimization techniques for DLS"""
    
    def __init__(self):
        self.optimization_results = []
    
    def profile_heuristic_ordering(self, base_algorithm, heuristic_func, 
                                 test_cases: List[Dict], depth_limits: List[int]) -> Dict:
        """Profile heuristic ordering optimization"""
        results = {'without_heuristic': [], 'with_heuristic': []}
        
        for test_case in test_cases:
            for depth_limit in depth_limits:
                # Test without heuristic
                expand_func = test_case['expand_func']
                goal_test_func = test_case['goal_test_func']
                initial_state = test_case['initial_state']
                
                algorithm = base_algorithm(expand_func, goal_test_func, depth_limit)
                start_time = time.time()
                result = algorithm.search(initial_state)
                end_time = time.time()
                
                results['without_heuristic'].append({
                    'depth': depth_limit,
                    'time': end_time - start_time,
                    'nodes_explored': algorithm.nodes_explored,
                    'found': result is not None
                })
                
                # Test with heuristic
                def heuristic_expand(node):
                    children = expand_func(node)
                    # Sort by heuristic
                    children.sort(key=lambda n: heuristic_func(n))
                    return children
                
                algorithm = base_algorithm(heuristic_expand, goal_test_func, depth_limit)
                start_time = time.time()
                result = algorithm.search(initial_state)
                end_time = time.time()
                
                results['with_heuristic'].append({
                    'depth': depth_limit,
                    'time': end_time - start_time,
                    'nodes_explored': algorithm.nodes_explored,
                    'found': result is not None
                })
        
        return results
    
    def profile_parallel_vs_sequential(self, parallel_algorithm, sequential_algorithm,
                                    test_cases: List[Dict], depth_limits: List[int],
                                    num_workers: List[int]) -> Dict:
        """Profile parallel vs sequential performance"""
        results = {'sequential': [], 'parallel': []}
        
        for test_case in test_cases:
            for depth_limit in depth_limits:
                expand_func = test_case['expand_func']
                goal_test_func = test_case['goal_test_func']
                initial_state = test_case['initial_state']
                
                # Sequential
                seq_algorithm = sequential_algorithm(expand_func, goal_test_func, depth_limit)
                start_time = time.time()
                seq_result = seq_algorithm.search(initial_state)
                seq_time = time.time() - start_time
                
                results['sequential'].append({
                    'depth': depth_limit,
                    'time': seq_time,
                    'nodes_explored': seq_algorithm.nodes_explored,
                    'found': seq_result is not None
                })
                
                # Parallel with different worker counts
                for workers in num_workers:
                    par_algorithm = parallel_algorithm(expand_func, goal_test_func, 
                                                      depth_limit, workers)
                    start_time = time.time()
                    par_result = par_algorithm.search(initial_state)
                    par_time = time.time() - start_time
                    
                    results['parallel'].append({
                        'depth': depth_limit,
                        'workers': workers,
                        'time': par_time,
                        'speedup': seq_time / par_time if par_time > 0 else 0,
                        'found': par_result is not None
                    })
        
        return results
    
    def profile_memory_optimization(self, base_algorithm, optimized_algorithm,
                                  test_cases: List[Dict], depth_limits: List[int]) -> Dict:
        """Profile memory optimization techniques"""
        results = {'base': [], 'optimized': []}
        
        for test_case in test_cases:
            for depth_limit in depth_limits:
                expand_func = test_case['expand_func']
                goal_test_func = test_case['goal_test_func']
                initial_state = test_case['initial_state']
                
                # Base algorithm
                base_alg = base_algorithm(expand_func, goal_test_func, depth_limit)
                
                import tracemalloc
                tracemalloc.start()
                base_result = base_alg.search(initial_state)
                base_current, base_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                results['base'].append({
                    'depth': depth_limit,
                    'memory_mb': base_peak / (1024 * 1024),
                    'nodes_explored': base_alg.nodes_explored,
                    'found': base_result is not None
                })
                
                # Optimized algorithm
                opt_alg = optimized_algorithm(expand_func, goal_test_func, depth_limit)
                
                tracemalloc.start()
                opt_result = opt_alg.search(initial_state)
                opt_current, opt_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                results['optimized'].append({
                    'depth': depth_limit,
                    'memory_mb': opt_peak / (1024 * 1024),
                    'nodes_explored': opt_alg.nodes_explored,
                    'found': opt_result is not None,
                    'memory_reduction': (base_peak - opt_peak) / (1024 * 1024)
                })
        
        return results


# Example usage and test case generators
def generate_tree_test_case(branching_factor: int, max_depth: int) -> Dict:
    """Generate a tree-based test case"""
    from dls_implementation import Node
    
    def create_tree_node(node_id: int, depth: int, max_depth: int, branching_factor: int):
        if depth >= max_depth:
            return {'id': node_id, 'children': [], 'depth': depth}
        
        children = []
        for i in range(branching_factor):
            child_id = node_id * 10 + i + 1
            child = create_tree_node(child_id, depth + 1, max_depth, branching_factor)
            children.append(child)
        
        return {'id': node_id, 'children': children, 'depth': depth}
    
    # Create tree
    root = create_tree_node(1, 0, max_depth, branching_factor)
    goal_id = root['id'] * 10**(max_depth - 1)  # Goal at maximum depth
    
    def expand_func(node):
        if isinstance(node.state, dict):
            children = node.state.get('children', [])
            return [Node(child, node, node.depth + 1) for child in children]
        return []
    
    def goal_test_func(node):
        return (isinstance(node.state, dict) and 
               node.state.get('id') == goal_id)
    
    return {
        'name': f'tree_b{branching_factor}_d{max_depth}',
        'initial_state': root,
        'goal_state': goal_id,
        'expand_func': expand_func,
        'goal_test_func': goal_test_func
    }


def generate_maze_test_case(width: int, height: int, obstacle_density: float = 0.2) -> Dict:
    """Generate a maze-based test case"""
    import random
    
    # Create maze
    maze = [[0 for _ in range(width)] for _ in range(height)]
    
    # Add obstacles
    for y in range(height):
        for x in range(width):
            if random.random() < obstacle_density:
                maze[y][x] = 1  # Wall
    
    # Set start and goal
    start_pos = (0, 0)
    goal_pos = (width - 1, height - 1)
    maze[0][0] = 0  # Ensure start is clear
    maze[height - 1][width - 1] = 0  # Ensure goal is clear
    
    from dls_implementation import expand_maze_node, maze_goal_test
    
    return {
        'name': f'maze_{width}x{height}',
        'initial_state': start_pos,
        'goal_state': goal_pos,
        'expand_func': lambda node: expand_maze_node(node, maze),
        'goal_test_func': maze_goal_test(goal_pos)
    }


if __name__ == "__main__":
    # Example performance analysis
    from dls_implementation import DepthLimitedSearch, IterativeDeepeningDLS
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer()
    
    # Define algorithms to test
    algorithms = [
        {'name': 'DLS', 'class': DepthLimitedSearch},
        {'name': 'IDDFS', 'class': IterativeDeepeningDLS}
    ]
    
    # Generate test cases
    test_cases = [
        generate_tree_test_case(3, 4),
        generate_tree_test_case(2, 6),
        generate_maze_test_case(10, 10, 0.1)
    ]
    
    # Define depth limits to test
    depth_limits = [3, 4, 5, 6, 7, 8]
    
    # Run comprehensive benchmark
    print("Running comprehensive benchmark...")
    results = analyzer.comprehensive_benchmark(algorithms, test_cases, depth_limits)
    
    # Generate report
    report = analyzer.generate_performance_report()
    print("\nPerformance Report:")
    print(report)
    
    # Save report
    with open(f"{analyzer.output_dir}/performance_report.txt", 'w') as f:
        f.write(report)
    
    # Plot performance comparison
    analyzer.plot_performance_comparison(results, PerformanceMetric.TIME)
    analyzer.plot_performance_comparison(results, PerformanceMetric.NODES_EXPLORED)
    
    # Export results
    analyzer.export_results(results, 'csv')
    analyzer.export_results(results, 'json')
    
    # Profile optimizations
    print("\nProfiling optimizations...")
    optimizer = OptimizationProfiler()
    
    # Profile heuristic ordering
    def simple_heuristic(node):
        # Simple heuristic: prefer nodes with smaller IDs (assuming smaller IDs are closer to goal)
        if isinstance(node.state, dict):
            return node.state.get('id', float('inf'))
        return float('inf')
    
    heuristic_results = optimizer.profile_heuristic_ordering(
        DepthLimitedSearch, simple_heuristic, test_cases[:1], depth_limits[:3]
    )
    
    print("Heuristic ordering results:")
    print(heuristic_results)
