"""
Advanced Applications and Real-World Implementations of Depth Limited Search
Professional and Industry Level Use Cases
"""

import numpy as np
import networkx as nx
import heapq
from typing import List, Dict, Any, Tuple, Optional, Set
import math
import random
from dataclasses import dataclass
from enum import Enum
import json
import time
from abc import ABC, abstractmethod


class ApplicationType(Enum):
    """Types of applications for DLS"""
    ROBOTICS = "robotics"
    NETWORK_ROUTING = "network_routing"
    GAME_AI = "game_ai"
    SUPPLY_CHAIN = "supply_chain"
    CYBERSECURITY = "cybersecurity"
    BIOINFORMATICS = "bioinformatics"


@dataclass
class ApplicationConfig:
    """Configuration for DLS applications"""
    max_depth: int
    branching_factor_estimate: int
    time_limit: float
    memory_limit: float
    success_threshold: float


class RoboticsPathPlanner:
    """Advanced robotics path planning using DLS"""
    
    def __init__(self, workspace_bounds: Tuple[float, float, float, float],
                 obstacles: List[Dict], robot_radius: float = 0.5):
        self.min_x, self.min_y, self.max_x, self.max_y = workspace_bounds
        self.obstacles = obstacles
        self.robot_radius = robot_radius
        self.resolution = 0.5  # Grid resolution
        
    def discretize_position(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """Convert continuous position to discrete grid coordinates"""
        x, y = position
        grid_x = int((x - self.min_x) / self.resolution)
        grid_y = int((y - self.min_y) / self.resolution)
        return (grid_x, grid_y)
    
    def continuous_position(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to continuous position"""
        grid_x, grid_y = grid_pos
        x = grid_x * self.resolution + self.min_x
        y = grid_y * self.resolution + self.min_y
        return (x, y)
    
    def is_collision_free(self, position: Tuple[float, float]) -> bool:
        """Check if position is collision-free"""
        x, y = position
        
        # Check bounds
        if (x < self.min_x or x > self.max_x or 
            y < self.min_y or y > self.max_y):
            return False
        
        # Check obstacles
        for obstacle in self.obstacles:
            obs_x, obs_y, obs_size = obstacle['center'] + [obstacle['size']]
            distance = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if distance < (obs_size + self.robot_radius):
                return False
        
        return True
    
    def get_valid_neighbors(self, position: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Get valid neighboring positions"""
        x, y = position
        neighbors = []
        
        # 8-directional movement
        movements = [
            (self.resolution, 0),    # Right
            (self.resolution, self.resolution),  # Right-Up
            (0, self.resolution),    # Up
            (-self.resolution, self.resolution), # Left-Up
            (-self.resolution, 0),   # Left
            (-self.resolution, -self.resolution), # Left-Down
            (0, -self.resolution),   # Down
            (self.resolution, -self.resolution)  # Right-Down
        ]
        
        for dx, dy in movements:
            new_pos = (x + dx, y + dy)
            if self.is_collision_free(new_pos):
                neighbors.append(new_pos)
        
        return neighbors
    
    def plan_path_dls(self, start: Tuple[float, float], goal: Tuple[float, float],
                     max_depth: int) -> Optional[List[Tuple[float, float]]]:
        """Plan path using Depth Limited Search"""
        from dls_implementation import DepthLimitedSearch, Node
        
        def expand_func(node):
            position = node.state
            neighbors = self.get_valid_neighbors(position)
            return [Node(neighbor, node, node.depth + 1) for neighbor in neighbors]
        
        def goal_test_func(node):
            # Check if within goal threshold
            distance = math.sqrt((node.state[0] - goal[0])**2 + 
                               (node.state[1] - goal[1])**2)
            return distance < self.resolution
        
        dls = DepthLimitedSearch(expand_func, goal_test_func, max_depth)
        return dls.search(start)
    
    def plan_with_waypoints(self, waypoints: List[Tuple[float, float]], 
                           max_depth: int) -> List[List[Tuple[float, float]]]:
        """Plan path through multiple waypoints"""
        full_path = []
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            goal = waypoints[i + 1]
            
            segment = self.plan_path_dls(start, goal, max_depth)
            if segment is None:
                raise ValueError(f"No path found from {start} to {goal}")
            
            # Avoid duplicate waypoints
            if full_path and segment[0] == full_path[-1]:
                full_path.extend(segment[1:])
            else:
                full_path.extend(segment)
        
        return full_path


class NetworkRouter:
    """Advanced network routing using DLS"""
    
    def __init__(self, network_graph: nx.Graph):
        self.graph = network_graph
        self.bandwidth_cache = {}
        self.latency_cache = {}
    
    def calculate_path_metrics(self, path: List[str]) -> Dict[str, float]:
        """Calculate metrics for a network path"""
        if not path:
            return {'bandwidth': 0, 'latency': 0, 'hop_count': 0}
        
        total_bandwidth = float('inf')
        total_latency = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Get edge data
            edge_data = self.graph[u][v]
            bandwidth = edge_data.get('bandwidth', 1000)  # Mbps
            latency = edge_data.get('latency', 1)  # ms
            
            total_bandwidth = min(total_bandwidth, bandwidth)
            total_latency += latency
        
        return {
            'bandwidth': total_bandwidth,
            'latency': total_latency,
            'hop_count': len(path) - 1
        }
    
    def find_route_dls(self, source: str, destination: str, 
                      max_hops: int, constraints: Dict = None) -> Optional[List[str]]:
        """Find network route using DLS with constraints"""
        from dls_implementation import DepthLimitedSearch, Node
        
        constraints = constraints or {}
        
        def expand_func(node):
            current_node = node.state
            neighbors = []
            
            for neighbor in self.graph.neighbors(current_node):
                # Check constraints
                edge_data = self.graph[current_node][neighbor]
                
                # Bandwidth constraint
                if 'min_bandwidth' in constraints:
                    if edge_data.get('bandwidth', 0) < constraints['min_bandwidth']:
                        continue
                
                # Avoid cycles
                if neighbor in node.get_path():
                    continue
                
                neighbors.append(Node(neighbor, node, node.depth + 1))
            
            return neighbors
        
        def goal_test_func(node):
            return node.state == destination
        
        dls = DepthLimitedSearch(expand_func, goal_test_func, max_hops)
        return dls.search(source)
    
    def find_multi_path_routes(self, source: str, destination: str,
                             max_hops: int, num_paths: int) -> List[List[str]]:
        """Find multiple diverse paths between source and destination"""
        paths = []
        used_edges = set()
        
        for _ in range(num_paths):
            # Create constraints to avoid previously used edges
            def expand_func(node):
                current_node = node.state
                neighbors = []
                
                for neighbor in self.graph.neighbors(current_node):
                    edge = tuple(sorted([current_node, neighbor]))
                    
                    # Avoid used edges
                    if edge in used_edges:
                        continue
                    
                    # Avoid cycles
                    if neighbor in node.get_path():
                        continue
                    
                    neighbors.append(Node(neighbor, node, node.depth + 1))
            
            def goal_test_func(node):
                return node.state == destination
            
            dls = DepthLimitedSearch(expand_func, goal_test_func, max_hops)
            path = dls.search(source)
            
            if path is None:
                break
            
            paths.append(path)
            
            # Mark edges as used
            for i in range(len(path) - 1):
                edge = tuple(sorted([path[i], path[i + 1]]))
                used_edges.add(edge)
        
        return paths
    
    def optimize_network_flow(self, demands: List[Tuple[str, str, float]], 
                            max_hops: int) -> Dict[str, Any]:
        """Optimize network flow for multiple demands"""
        routing_solution = {}
        total_utilization = 0
        
        for source, destination, demand in demands:
            path = self.find_route_dls(source, destination, max_hops)
            
            if path:
                routing_solution[f"{source}-{destination}"] = {
                    'path': path,
                    'demand': demand,
                    'metrics': self.calculate_path_metrics(path)
                }
                
                # Calculate utilization (simplified)
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    capacity = self.graph[u][v].get('bandwidth', 1000)
                    utilization = (demand / capacity) * 100
                    total_utilization += utilization
        
        return {
            'routes': routing_solution,
            'total_utilization': total_utilization / len(demands) if demands else 0,
            'success_rate': len(routing_solution) / len(demands) if demands else 0
        }


class GameAIPlayer:
    """Game AI using DLS for strategic decision making"""
    
    def __init__(self, game_type: str, max_depth: int = 6):
        self.game_type = game_type
        self.max_depth = max_depth
        self.transposition_table = {}
        self.move_history = []
    
    def evaluate_position(self, game_state: Dict) -> float:
        """Evaluate game position (simplified)"""
        if self.game_type == "tic_tac_toe":
            return self._evaluate_tic_tac_toe(game_state)
        elif self.game_type == "connect_four":
            return self._evaluate_connect_four(game_state)
        else:
            return self._evaluate_generic(game_state)
    
    def _evaluate_tic_tac_toe(self, state: Dict) -> float:
        """Evaluate Tic-Tac-Toe position"""
        board = state['board']
        player = state['current_player']
        opponent = 'O' if player == 'X' else 'X'
        
        # Check for wins
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        
        for line in lines:
            player_count = sum(1 for i in line if board[i] == player)
            opponent_count = sum(1 for i in line if board[i] == opponent)
            empty_count = sum(1 for i in line if board[i] == ' ')
            
            if player_count == 3:
                return 1000
            elif opponent_count == 3:
                return -1000
            elif player_count == 2 and empty_count == 1:
                return 100
            elif opponent_count == 2 and empty_count == 1:
                return -100
        
        return 0
    
    def _evaluate_connect_four(self, state: Dict) -> float:
        """Evaluate Connect Four position"""
        board = state['board']
        player = state['current_player']
        opponent = 'R' if player == 'Y' else 'Y'
        
        score = 0
        
        # Check all possible 4-in-a-row combinations
        rows, cols = len(board), len(board[0])
        
        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                window = [board[r][c+i] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)
        
        # Vertical
        for r in range(rows - 3):
            for c in range(cols):
                window = [board[r+i][c] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)
        
        # Diagonal (positive)
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = [board[r+i][c+i] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)
        
        # Diagonal (negative)
        for r in range(3, rows):
            for c in range(cols - 3):
                window = [board[r-i][c+i] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)
        
        return score
    
    def _evaluate_window(self, window: List[str], player: str, opponent: str) -> float:
        """Evaluate a 4-piece window in Connect Four"""
        score = 0
        
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(' ')
        
        if player_count == 4:
            score += 1000
        elif player_count == 3 and empty_count == 1:
            score += 100
        elif player_count == 2 and empty_count == 2:
            score += 10
        
        if opponent_count == 3 and empty_count == 1:
            score -= 80
        
        return score
    
    def _evaluate_generic(self, state: Dict) -> float:
        """Generic position evaluation"""
        # Simplified generic evaluation
        return random.uniform(-1, 1)
    
    def get_best_move_dls(self, game_state: Dict) -> Optional[Any]:
        """Get best move using DLS with minimax"""
        from dls_implementation import DepthLimitedSearch, Node
        
        def expand_func(node):
            state = node.state
            if self._is_terminal(state):
                return []
            
            moves = self._get_legal_moves(state)
            children = []
            
            for move in moves:
                new_state = self._make_move(state, move)
                children.append(Node(new_state, node, node.depth + 1))
            
            return children
        
        def goal_test_func(node):
            return self._is_terminal(node.state)
        
        # Search for best move
        best_move = None
        best_value = float('-inf')
        
        moves = self._get_legal_moves(game_state)
        
        for move in moves:
            new_state = self._make_move(game_state, move)
            
            value = self._minimax_dls(new_state, self.max_depth - 1, False)
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move
    
    def _minimax_dls(self, state: Dict, depth: int, is_maximizing: bool) -> float:
        """Minimax with depth limit"""
        if depth == 0 or self._is_terminal(state):
            return self.evaluate_position(state)
        
        if is_maximizing:
            value = float('-inf')
            for move in self._get_legal_moves(state):
                new_state = self._make_move(state, move)
                value = max(value, self._minimax_dls(new_state, depth - 1, False))
            return value
        else:
            value = float('inf')
            for move in self._get_legal_moves(state):
                new_state = self._make_move(state, move)
                value = min(value, self._minimax_dls(new_state, depth - 1, True))
            return value
    
    def _is_terminal(self, state: Dict) -> bool:
        """Check if game state is terminal"""
        if self.game_type == "tic_tac_toe":
            board = state['board']
            # Check for win or draw
            lines = [
                [0, 1, 2], [3, 4, 5], [6, 7, 8],
                [0, 3, 6], [1, 4, 7], [2, 5, 8],
                [0, 4, 8], [2, 4, 6]
            ]
            
            for line in lines:
                a, b, c = [board[i] for i in line]
                if a != ' ' and a == b == c:
                    return True
            
            return ' ' not in board
        
        elif self.game_type == "connect_four":
            # Simplified terminal check
            return False
        
        return False
    
    def _get_legal_moves(self, state: Dict) -> List[Any]:
        """Get legal moves for current state"""
        if self.game_type == "tic_tac_toe":
            board = state['board']
            return [i for i, cell in enumerate(board) if cell == ' ']
        
        elif self.game_type == "connect_four":
            board = state['board']
            moves = []
            for col in range(len(board[0])):
                if board[0][col] == ' ':
                    moves.append(col)
            return moves
        
        return []
    
    def _make_move(self, state: Dict, move: Any) -> Dict:
        """Make a move and return new state"""
        new_state = state.copy()
        
        if self.game_type == "tic_tac_toe":
            new_board = new_state['board'].copy()
            new_board[move] = new_state['current_player']
            new_state['board'] = new_board
            new_state['current_player'] = 'O' if new_state['current_player'] == 'X' else 'X'
        
        elif self.game_type == "connect_four":
            new_board = [row[:] for row in new_state['board']]
            
            # Find the lowest empty row in the column
            for row in range(len(new_board) - 1, -1, -1):
                if new_board[row][move] == ' ':
                    new_board[row][move] = new_state['current_player']
                    break
            
            new_state['board'] = new_board
            new_state['current_player'] = 'R' if new_state['current_player'] == 'Y' else 'Y'
        
        return new_state


class SupplyChainOptimizer:
    """Supply chain optimization using DLS"""
    
    def __init__(self, network: Dict, inventory: Dict, demands: Dict):
        self.network = network  # Distribution network
        self.inventory = inventory  # Current inventory levels
        self.demands = demands  # Customer demands
        self.costs = {}  # Transportation costs
    
    def optimize_distribution(self, planning_horizon: int, max_depth: int) -> Dict:
        """Optimize distribution plan using DLS"""
        from dls_implementation import DepthLimitedSearch, Node
        
        def expand_func(node):
            state = node.state
            if node.depth >= max_depth:
                return []
            
            # Generate possible distribution actions
            actions = self._generate_distribution_actions(state)
            children = []
            
            for action in actions:
                new_state = self._apply_action(state, action)
                children.append(Node(new_state, node, node.depth + 1))
            
            return children
        
        def goal_test_func(node):
            state = node.state
            # Check if all demands are satisfied
            for customer, demand in self.demands.items():
                if state['satisfied'].get(customer, 0) < demand:
                    return False
            return True
        
        initial_state = {
            'inventory': self.inventory.copy(),
            'satisfied': {customer: 0 for customer in self.demands},
            'cost': 0,
            'actions': []
        }
        
        dls = DepthLimitedSearch(expand_func, goal_test_func, max_depth)
        result = dls.search(initial_state)
        
        if result:
            return {
                'plan': result[-1]['actions'],
                'total_cost': result[-1]['cost'],
                'satisfied_demands': result[-1]['satisfied']
            }
        
        return None
    
    def _generate_distribution_actions(self, state: Dict) -> List[Dict]:
        """Generate possible distribution actions"""
        actions = []
        
        for source, available in state['inventory'].items():
            if available <= 0:
                continue
            
            for destination in self.network.get(source, []):
                for customer in self.demands:
                    if destination == customer:
                        # Ship to customer
                        quantity = min(available, 
                                     self.demands[customer] - state['satisfied'].get(customer, 0))
                        if quantity > 0:
                            actions.append({
                                'type': 'ship_to_customer',
                                'source': source,
                                'destination': destination,
                                'quantity': quantity
                            })
                    else:
                        # Transfer between facilities
                        transfer_quantity = min(available, 10)  # Max transfer of 10 units
                        if transfer_quantity > 0:
                            actions.append({
                                'type': 'transfer',
                                'source': source,
                                'destination': destination,
                                'quantity': transfer_quantity
                            })
        
        return actions
    
    def _apply_action(self, state: Dict, action: Dict) -> Dict:
        """Apply distribution action to state"""
        new_state = {
            'inventory': state['inventory'].copy(),
            'satisfied': state['satisfied'].copy(),
            'cost': state['cost'],
            'actions': state['actions'] + [action]
        }
        
        if action['type'] == 'ship_to_customer':
            # Update inventory
            new_state['inventory'][action['source']] -= action['quantity']
            
            # Update satisfied demands
            customer = action['destination']
            new_state['satisfied'][customer] += action['quantity']
            
            # Calculate cost
            cost = self._calculate_shipping_cost(action['source'], customer, action['quantity'])
            new_state['cost'] += cost
        
        elif action['type'] == 'transfer':
            # Update inventory
            new_state['inventory'][action['source']] -= action['quantity']
            new_state['inventory'][action['destination']] = new_state['inventory'].get(action['destination'], 0) + action['quantity']
            
            # Calculate transfer cost
            cost = self._calculate_transfer_cost(action['source'], action['destination'], action['quantity'])
            new_state['cost'] += cost
        
        return new_state
    
    def _calculate_shipping_cost(self, source: str, destination: str, quantity: float) -> float:
        """Calculate shipping cost"""
        # Simplified cost calculation
        distance = self._get_distance(source, destination)
        return distance * quantity * 0.1  # $0.1 per unit per distance unit
    
    def _calculate_transfer_cost(self, source: str, destination: str, quantity: float) -> float:
        """Calculate transfer cost between facilities"""
        distance = self._get_distance(source, destination)
        return distance * quantity * 0.05  # $0.05 per unit per distance unit
    
    def _get_distance(self, source: str, destination: str) -> float:
        """Get distance between two locations"""
        # Simplified distance calculation
        return abs(hash(source) - hash(destination)) % 100


class CybersecurityAnalyzer:
    """Cybersecurity threat analysis using DLS"""
    
    def __init__(self, network_topology: Dict, vulnerability_database: Dict):
        self.network = network_topology
        self.vulnerabilities = vulnerability_database
        self.attack_graph = None
    
    def build_attack_graph(self) -> nx.DiGraph:
        """Build attack graph for the network"""
        self.attack_graph = nx.DiGraph()
        
        # Add nodes for each host
        for host in self.network['hosts']:
            self.attack_graph.add_node(host, **self.network['hosts'][host])
        
        # Add edges for network connections
        for connection in self.network['connections']:
            source, dest = connection['source'], connection['destination']
            self.attack_graph.add_edge(source, dest, **connection)
        
        return self.attack_graph
    
    def analyze_attack_paths(self, attacker_entry: str, target_asset: str, 
                           max_depth: int) -> List[Dict]:
        """Analyze possible attack paths using DLS"""
        from dls_implementation import DepthLimitedSearch, Node
        
        if not self.attack_graph:
            self.build_attack_graph()
        
        def expand_func(node):
            current_host = node.state
            
            # Get possible next hosts
            neighbors = list(self.attack_graph.successors(current_host))
            children = []
            
            for neighbor in neighbors:
                # Check if neighbor is vulnerable
                if self._is_vulnerable(current_host, neighbor):
                    children.append(Node(neighbor, node, node.depth + 1))
            
            return children
        
        def goal_test_func(node):
            return node.state == target_asset
        
        dls = DepthLimitedSearch(expand_func, goal_test_func, max_depth)
        result = dls.search(attacker_entry)
        
        if result:
            return self._analyze_path_vulnerabilities(result)
        
        return []
    
    def _is_vulnerable(self, source: str, target: str) -> bool:
        """Check if target is vulnerable from source"""
        target_host = self.network['hosts'].get(target, {})
        vulnerabilities = target_host.get('vulnerabilities', [])
        
        # Check if any vulnerability can be exploited from source
        for vuln in vulnerabilities:
            if vuln in self.vulnerabilities:
                vuln_info = self.vulnerabilities[vuln]
                # Check if source has required privileges or access
                if self._can_exploit(source, vuln_info):
                    return True
        
        return False
    
    def _can_exploit(self, source: str, vulnerability: Dict) -> bool:
        """Check if source can exploit given vulnerability"""
        required_access = vulnerability.get('required_access', 'network')
        source_privileges = self.network['hosts'].get(source, {}).get('privileges', [])
        
        return required_access in source_privileges or 'admin' in source_privileges
    
    def _analyze_path_vulnerabilities(self, path: List[str]) -> List[Dict]:
        """Analyze vulnerabilities along attack path"""
        analysis = []
        
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            
            target_host = self.network['hosts'].get(target, {})
            vulns = target_host.get('vulnerabilities', [])
            
            exploitable_vulns = []
            for vuln in vulns:
                if vuln in self.vulnerabilities:
                    vuln_info = self.vulnerabilities[vuln]
                    if self._can_exploit(source, vuln_info):
                        exploitable_vulns.append({
                            'cve': vuln,
                            'severity': vuln_info.get('severity', 'unknown'),
                            'description': vuln_info.get('description', '')
                        })
            
            analysis.append({
                'source': source,
                'target': target,
                'vulnerabilities': exploitable_vulns,
                'risk_score': self._calculate_risk_score(exploitable_vulns)
            })
        
        return analysis
    
    def _calculate_risk_score(self, vulnerabilities: List[Dict]) -> float:
        """Calculate risk score for vulnerabilities"""
        if not vulnerabilities:
            return 0.0
        
        severity_scores = {
            'critical': 10.0,
            'high': 7.5,
            'medium': 5.0,
            'low': 2.5,
            'unknown': 1.0
        }
        
        total_score = sum(severity_scores.get(vuln['severity'], 1.0) 
                         for vuln in vulnerabilities)
        
        return total_score / len(vulnerabilities)


# Example usage and demonstrations
if __name__ == "__main__":
    # Example 1: Robotics Path Planning
    print("=== Robotics Path Planning ===")
    obstacles = [
        {'center': [5, 5], 'size': 1.0},
        {'center': [8, 3], 'size': 0.8},
        {'center': [3, 8], 'size': 1.2}
    ]
    
    planner = RoboticsPathPlanner((0, 0, 10, 10), obstacles)
    start_pos = (1, 1)
    goal_pos = (9, 9)
    
    path = planner.plan_path_dls(start_pos, goal_pos, max_depth=20)
    if path:
        print(f"Path found: {path[:5]}... (length: {len(path)})")
    else:
        print("No path found")
    
    # Example 2: Network Routing
    print("\n=== Network Routing ===")
    # Create sample network
    G = nx.Graph()
    G.add_edge("A", "B", bandwidth=1000, latency=10)
    G.add_edge("B", "C", bandwidth=500, latency=5)
    G.add_edge("C", "D", bandwidth=1000, latency=15)
    G.add_edge("A", "D", bandwidth=200, latency=25)
    G.add_edge("B", "D", bandwidth=800, latency=12)
    
    router = NetworkRouter(G)
    route = router.find_route_dls("A", "D", max_hops=3)
    if route:
        metrics = router.calculate_path_metrics(route)
        print(f"Route: {route}")
        print(f"Metrics: {metrics}")
    
    # Example 3: Game AI
    print("\n=== Game AI ===")
    ai_player = GameAIPlayer("tic_tac_toe", max_depth=4)
    
    game_state = {
        'board': ['X', 'O', 'X', ' ', 'O', ' ', ' ', ' ', ' '],
        'current_player': 'X'
    }
    
    best_move = ai_player.get_best_move_dls(game_state)
    print(f"Best move: {best_move}")
    
    # Example 4: Cybersecurity Analysis
    print("\n=== Cybersecurity Analysis ===")
    network = {
        'hosts': {
            'web_server': {'privileges': ['network'], 'vulnerabilities': ['CVE-2021-1234']},
            'db_server': {'privileges': ['local'], 'vulnerabilities': ['CVE-2021-5678']},
            'workstation': {'privileges': ['user'], 'vulnerabilities': []}
        },
        'connections': [
            {'source': 'web_server', 'destination': 'db_server'},
            {'source': 'workstation', 'destination': 'web_server'}
        ]
    }
    
    vuln_db = {
        'CVE-2021-1234': {'severity': 'high', 'required_access': 'network', 'description': 'SQL Injection'},
        'CVE-2021-5678': {'severity': 'critical', 'required_access': 'local', 'description': 'Privilege Escalation'}
    }
    
    analyzer = CybersecurityAnalyzer(network, vuln_db)
    attack_paths = analyzer.analyze_attack_paths('workstation', 'db_server', max_depth=3)
    
    if attack_paths:
        print("Attack paths found:")
        for i, path_info in enumerate(attack_paths):
            print(f"  Step {i+1}: {path_info['source']} -> {path_info['target']}")
            print(f"    Risk Score: {path_info['risk_score']}")
            for vuln in path_info['vulnerabilities']:
                print(f"    Vulnerability: {vuln['cve']} ({vuln['severity']})")
    else:
        print("No attack paths found")
