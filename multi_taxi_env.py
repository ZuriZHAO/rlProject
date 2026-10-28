"""
multi_taxi_env.py
=================
Ride-hailing Dispatch Optimization —— Multi-Agent Reinforcement Learning Environment (MultiTaxiEnv)

Core Mechanisms:
  A. Congestion Risk Game (Risk-Aware Routing)
  B. Mileage Subsidy Dynamic Reward (Distance-Based Reward)
  C. Hotspot Weighted Spawning (Hotspot Spawning)
  
Current Version:
  - Multi-taxi multi-passenger
  - Each taxi returns independent local observations
  - step receives action_dict, returns multi-agent rewards / dones / infos
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

# ─────────────────────────────────────────────
#  Map Symbol Constants
# ─────────────────────────────────────────────
CELL_ROAD     = 0
CELL_WALL     = 1
CELL_ARTERIAL = 2   # Main roads/congested areas
CELL_HOTSPOT  = 3   # CBD (Central Business District)
# ─────────────────────────────────────────────
#  Action Constants
# ─────────────────────────────────────────────
ACTION_SOUTH   = 0
ACTION_NORTH   = 1
ACTION_EAST    = 2
ACTION_WEST    = 3
ACTION_PICKUP  = 4
ACTION_DROPOFF = 5
N_ACTIONS      = 6

MOVE_DELTA: Dict[int, Tuple[int, int]] = {
    ACTION_SOUTH: ( 1,  0),
    ACTION_NORTH: (-1,  0),
    ACTION_EAST:  ( 0,  1),
    ACTION_WEST:  ( 0, -1),
}

IN_TAXI_CODE   = -1
DELIVERED_CODE = -2


class MultiTaxiEnv:
    """
    Multi-taxi + multiple passengers dispatch environment.

    Map: 7x7 Grid World
    Task: Multiple taxis collaborate to complete multiple passenger pickups and deliveries

    Parameters
    ----
    n_taxis        : Number of taxis, >= 1
    n_passengers   : Number of passengers, >= 1
    congestion_prob: Arterial road congestion probability, default 0.3
    reward_alpha   : Mileage subsidy coefficient α, default 2.0
    base_reward    : Base reward for single passenger delivery, default 20.0
    max_steps      : Maximum steps per episode, default 200
    seed           : Random seed
    """

    # Map Layout (7×7)
    #   Col:  0  1  2  3  4  5  6
    # Row 0 [ 3  2  2  2  2  2  3 ]  Top arterial + CBD
    # Row 1 [ 0  0  1  0  0  1  0 ]
    # Row 2 [ 0  0  1  0  0  1  0 ]
    # Row 3 [ 0  0  0  3  0  0  0 ]  Center CBD
    # Row 4 [ 0  0  1  0  0  1  0 ]
    # Row 5 [ 0  0  1  0  0  1  0 ]
    # Row 6 [ 3  2  2  2  2  2  3 ]  Bottom arterial + CBD
    MAP_GRID: List[List[int]] = [
        [3, 2, 2, 2, 2, 2, 3],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 3, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [3, 2, 2, 2, 2, 2, 3],
    ]

    HOTSPOT_WEIGHT_FACTOR: float = 5.0

    def __init__(
        self,
        n_taxis: int = 1,
        n_passengers: int = 1,
        congestion_prob: float = 0.3,
        reward_alpha: float = 2.0,
        base_reward: float = 20.0,
        max_steps: int = 200,
        overlap_penalty: float = 4.0, # Add overlapping penalties
        seed: Optional[int] = None,
    ) -> None:
        # Currently supports: multi-taxi multi-passenger
        if n_taxis < 1:
            raise ValueError("n_taxis must >= 1")
        if n_passengers < 1:
            raise ValueError("n_passengers must >= 1")
        
        self.n_taxis         = n_taxis
        self.n_passengers    = n_passengers
        self.congestion_prob = congestion_prob
        self.reward_alpha    = reward_alpha
        self.base_reward     = base_reward
        self.max_steps       = max_steps
        self.overlap_penalty = overlap_penalty
        self.rng             = np.random.default_rng(seed)

        self.grid = np.array(self.MAP_GRID, dtype=np.int8)
        self.n_rows, self.n_cols = self.grid.shape  # 7, 7

        # Walkable cells list
        self._walkable: List[Tuple[int, int]] = [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if self.grid[r, c] != CELL_WALL
        ]
        # O(1) reverse lookup
        self._walkable_idx: Dict[Tuple[int, int], int] = {
            cell: i for i, cell in enumerate(self._walkable)
        }

        # Precompute shortest path distances between all walkable cells
        self._dist_table = self._build_distance_table()
        
        # Arterial cells set (Mechanism A)
        self._arterial_cells: set = {
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if self.grid[r, c] == CELL_ARTERIAL
        }

        # Hotspots (Mechanism C)
        self._hotspot_cells: List[Tuple[int, int]] = [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if self.grid[r, c] == CELL_HOTSPOT
        ]
        _hotspot_set = set(self._hotspot_cells)
        self._normal_cells: List[Tuple[int, int]] = [
            cell for cell in self._walkable
            if cell not in _hotspot_set
        ]
        self._spawn_pool, self._spawn_weights = self._build_spawn_pool()

        if self.n_passengers > len(self._spawn_pool):
            raise ValueError(
                f"n_passengers={self.n_passengers} exceeds available pickup spawn positions {len(self._spawn_pool)}"
            )
        
        self.agent_ids: List[str] = [f"taxi_{i+1}" for i in range(self.n_taxis)]
        self.action_space_n = N_ACTIONS

        # Episode state: multi-taxi + multi-passenger
        self.taxi_positions: Dict[str, Tuple[int, int]] = {
            aid: (0, 0) for aid in self.agent_ids
        }

        # Each passenger has a set of states
        self.passenger_locs: List[int] = [0 for _ in range(self.n_passengers)]      # walkable idx / IN_TAXI / DELIVERED
        self.passenger_dests: List[int] = [0 for _ in range(self.n_passengers)]     # walkable idx
        self.passenger_dists: List[int] = [0 for _ in range(self.n_passengers)]     # pickup time BFS distance
        self.passenger_owners: List[Optional[str]] = [None for _ in range(self.n_passengers)]
        
        # Which passenger_id is currently carried by each taxi; None if none
        self.taxi_carrying: Dict[str, Optional[int]] = {
            aid: None for aid in self.agent_ids
        }

        self.delivered_count: int = 0
        self.delivered: bool = False     # True only when all passengers are delivered
        self.step_count: int = 0

        print(
            f"[MultiTaxiEnv] Initialization complete | Taxis:{self.n_taxis} | Passengers:{self.n_passengers} | "
            f"Map:{self.n_rows}×{self.n_cols} | Congestion prob:{congestion_prob:.0%} | "
            f"α={reward_alpha} | base_reward={base_reward} | overlap_penalty={overlap_penalty}"
        )
        self._print_map()

    # Random passenger spawning
    def _build_spawn_pool(self) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        pool = self._hotspot_cells + self._normal_cells
        raw  = np.array(
            [self.HOTSPOT_WEIGHT_FACTOR] * len(self._hotspot_cells)
            + [1.0] * len(self._normal_cells),
            dtype=np.float64,
        )
        return pool, raw / raw.sum()

    def _print_map(self) -> None:
        sym = {CELL_ROAD:"·", CELL_WALL:"█", CELL_ARTERIAL:"═", CELL_HOTSPOT:"★"}
        print("\n  Map Layout (·=normal road ═=arterial ★=CBD ██=wall)")
        print("  Col:", " ".join(str(c) for c in range(self.n_cols)))
        for r in range(self.n_rows):
            print(f"Row{r}:", " ".join(sym[int(self.grid[r, c])] for c in range(self.n_cols)))
        print()
    
    # In multi-taxi multi-passenger scenarios, multiple lookups are needed, such as nearest waiting passengers, PBRS potential functions, pickup mileage subsidies, etc.
    # Perform BFS from each walkable cell to get shortest path lengths to all other walkable cells, forming a 2D table
    def _build_distance_table(self) -> np.ndarray:
        """
        Precompute shortest path distances between walkable cells
        dist[i, j] = shortest path length from _walkable[i] to _walkable[j]
        """
        n = len(self._walkable)
        dist = np.full((n, n), fill_value=999, dtype=np.int16)

        for i, start in enumerate(self._walkable):
            dist[i, i] = 0
            visited = {start}
            queue = deque([(start, 0)])

            while queue:
                (r, c), d = queue.popleft()
                cur_idx = self._walkable_idx[(r, c)]
                dist[i, cur_idx] = d

                for dr, dc in MOVE_DELTA.values():
                    nr, nc = r + dr, c + dc
                    nxt = (nr, nc)
                    if (0 <= nr < self.n_rows and 0 <= nc < self.n_cols
                            and self.grid[nr, nc] != CELL_WALL
                            and nxt not in visited):
                        visited.add(nxt)
                        queue.append((nxt, d + 1))

        return dist

    # O(1) lookup from precomputed table
    def _get_shortest_path_idx(self, start_idx: int, goal_idx: int) -> int:
        """O(1) lookup from precomputed table"""
        return int(self._dist_table[start_idx, goal_idx])

    # Use precomputed distance table to return shortest path length (calls previous function)
    def _get_shortest_path_length(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> int:
        """Use precomputed distance table to return shortest path length"""
        start_idx = self._walkable_idx[start]
        goal_idx = self._walkable_idx[goal]
        return self._get_shortest_path_idx(start_idx, goal_idx)

    # List of passenger ids still waiting for pickup
    def _get_waiting_passenger_ids(self) -> List[int]:
        """List of passenger ids still waiting for pickup"""
        return [
            pid
            for pid in range(self.n_passengers)
            if self.passenger_locs[pid] >= 0 and self.passenger_owners[pid] is None
        ]

    # Find idle taxis
    def _get_idle_taxi_ids(self) -> List[str]:
        return [aid for aid in self.agent_ids if self.taxi_carrying[aid] is None]

    # Lightweight global assignment: only for providing joint hints to agents, does not replace the strategy itself.
    # Rule: Greedy shortest path matching between current idle taxis and waiting passengers,
    # allowing one idle taxi to focus on one independent passenger, reducing multiple taxis crowding for the same order.
    def _assign_idle_taxis_to_waiting_passengers(self) -> Dict[str, Optional[int]]:
        """
        Lightweight global assignment: only for providing joint hints to agents, does not replace the strategy itself.
        Rule: Greedy shortest path matching between current idle taxis and waiting passengers,
        allowing one idle taxi to focus on one independent passenger, reducing multiple taxis crowding for the same order.
        """
        assignments: Dict[str, Optional[int]] = {aid: None for aid in self.agent_ids}
        idle_taxis = self._get_idle_taxi_ids()
        waiting = self._get_waiting_passenger_ids()

        if not idle_taxis or not waiting:
            return assignments

        remaining_taxis = set(idle_taxis)
        remaining_pids = set(waiting)

        while remaining_taxis and remaining_pids:
            best_pair: Optional[Tuple[int, int, str, int]] = None
            for aid in remaining_taxis:
                start_idx = self._walkable_idx[self.taxi_positions[aid]]
                for pid in remaining_pids:
                    dist = self._get_shortest_path_idx(start_idx, self.passenger_locs[pid])
                    candidate = (dist, pid, aid, pid)
                    if best_pair is None or candidate < best_pair:
                        best_pair = candidate

            assert best_pair is not None
            _, _, aid, pid = best_pair
            assignments[aid] = pid
            remaining_taxis.remove(aid)
            remaining_pids.remove(pid)

        return assignments
    
    def _get_assigned_waiting_passenger(self, agent_id: str) -> Optional[int]:
        return self._assign_idle_taxis_to_waiting_passengers().get(agent_id)

    # Assign the currently nearest waiting passenger to idle taxi (based on precomputed distance table)
    def _get_nearest_waiting_passenger(self, agent_id: str) -> Optional[int]:
        """Assign the currently nearest waiting passenger to idle taxi (based on precomputed distance table)"""
        waiting = self._get_waiting_passenger_ids()
        if not waiting:
            return None

        start_idx = self._walkable_idx[self.taxi_positions[agent_id]]

        return min(
            waiting,
            key=lambda pid: (
                self._get_shortest_path_idx(start_idx, self.passenger_locs[pid]),
                pid,
            ),
        )

    # Lightweight coordination observation (only keep 2 binary features):
    #    1) need_diversion: The nearest passenger is not the one currently assigned globally, indicating to avoid competition
    #    2) demand_high    : Whether the current number of pending passengers exceeds the number of available idle taxis
    def _get_coordination_bits(self, agent_id: str, target_pid: Optional[int]) -> Tuple[int, int]:
        """
        Lightweight coordination observation:
        1) need_diversion: The nearest passenger is not the one currently assigned globally, indicating to avoid competition
        2) demand_high   : Whether the current number of pending passengers exceeds the number of available idle taxis
        """
        waiting_ids = self._get_waiting_passenger_ids()
        idle_taxis = self._get_idle_taxi_ids()
        demand_high = int(len(waiting_ids) > len(idle_taxis))

        nearest_pid = self._get_nearest_waiting_passenger(agent_id)
        need_diversion = int(nearest_pid != target_pid)
        return need_diversion, demand_high         
    
    # ──────────────────────────────────────────
    #  reset
    # ──────────────────────────────────────────
    def reset(self) -> Dict[str, Tuple]:
        """
        Reset the episode.

        Returns
        ----
        {
            "taxi_1": (
                taxi_row, taxi_col,
                goal_row, goal_col,
                has_passenger,
                need_diversion,
                demand_high,
            ),
            "taxi_2": (...),
            ...
        }
        """
        
        self.step_count = 0
        self.delivered = False
        self.delivered_count = 0

        self.passenger_locs = [0 for _ in range(self.n_passengers)]
        self.passenger_dests = [0 for _ in range(self.n_passengers)]
        self.passenger_dists = [0 for _ in range(self.n_passengers)]
        self.passenger_owners = [None for _ in range(self.n_passengers)]
        self.taxi_carrying = {aid: None for aid in self.agent_ids}


        n_w = len(self._walkable)

        # Multi-taxis randomly assigned initial positions (not repeated)
        start_indices = self.rng.choice(n_w, size=self.n_taxis, replace=False)
        self.taxi_positions = {
            aid: self._walkable[int(idx)]
            for aid, idx in zip(self.agent_ids, np.atleast_1d(start_indices))
        }
        
        # Multi-passengers: Try not to duplicate boarding points
        used_pickups = set()
        for pid in range(self.n_passengers):
            while True:
                pool_idx = int(self.rng.choice(len(self._spawn_pool), p=self._spawn_weights))
                p_cell = self._spawn_pool[pool_idx]
                p_idx = self._walkable_idx[p_cell]
                if p_idx not in used_pickups:
                    self.passenger_locs[pid] = p_idx
                    used_pickups.add(p_idx)
                    break

            while True:
                d = int(self.rng.integers(0, n_w))
                if d != self.passenger_locs[pid]:
                    self.passenger_dests[pid] = d
                    break

        return {aid: self._get_obs(aid) for aid in self.agent_ids}


    # taxi_row, taxi_col: The taxi's own current position
    # goal_row, goal_col:
    # If the taxi has a passenger, it's the destination of that passenger
    # If the taxi is empty, it's the position of the nearest waiting passenger to pick up (consistent with joint assignment)
    # has_passenger: Whether carrying a passenger
    # need_diversion: Whether the current target is competed by other idle taxis
    # demand_high: Whether passenger demand exceeds idle taxi supply.
    def _get_obs(self, agent_id: str) -> Tuple:
        """
        Lightweight local observation + coordination binary features:
        (
            taxi_row, taxi_col,
            goal_row, goal_col,
            has_passenger,
            need_diversion,
            demand_high
        )
        """
        tr, tc = self.taxi_positions[agent_id]
        carrying_pid = self.taxi_carrying[agent_id]
        hp = int(carrying_pid is not None)

        if self.delivered:
            return (tr, tc, tr, tc, 0, 0, 0)

        target_pid: Optional[int] = None

        if carrying_pid is not None:
            target_pid = carrying_pid
            gr, gc = self._walkable[self.passenger_dests[carrying_pid]]
        else:
            target_pid = self._get_assigned_waiting_passenger(agent_id)
            if target_pid is None:
                gr, gc = tr, tc
            else:
                gr, gc = self._walkable[self.passenger_locs[target_pid]]
        
        need_diversion, demand_high = self._get_coordination_bits(agent_id, target_pid)

        return (tr, tc, gr, gc, hp, need_diversion, demand_high)

    # Single taxi step reward
    
    # Movement
    # Normal road normal move: -1
    # Wall/out of bounds: -1
    # Arterial road if congestion triggered: -10, and taxi does not move
    # Arterial road if no congestion: taxi moves, reward -2
    
    # Pickup
    # If taxi already has passenger, pickup fails, reward -2
    # If no waiting passenger at current position, also fails, reward -2
    # Reward given: 5.0 + reward_alpha * passenger_dists[pid]
    
    # Dropoff
    # If taxi has no passenger, fails, -2
    # If taxi has passenger but position not at passenger destination, serious failure, -10
    # Dropoff success, give base_reward
    def _apply_action(self, agent_id: str, action: int) -> Tuple[float, Dict[str, Any]]:
        # sourcery skip: low-code-quality, use-next
        r, c = self.taxi_positions[agent_id]
        reward = 0.0
        info: Dict[str, Any] = {}

        # ── Movement (0~3) ──────────────────────────
        if action in MOVE_DELTA:
            dr, dc = MOVE_DELTA[action]
            nr, nc = r + dr, c + dc

            if (0 <= nr < self.n_rows and 0 <= nc < self.n_cols
                    and self.grid[nr, nc] != CELL_WALL):
                if (nr, nc) in self._arterial_cells:
                    if bool(self.rng.binomial(1, self.congestion_prob)):
                        reward = -10.0
                        info["event"] = "arterial_blocked"
                    else:
                        self.taxi_positions[agent_id] = (nr, nc)
                        reward = -2.0
                        info["event"] = "arterial_clear"
                else:
                    self.taxi_positions[agent_id] = (nr, nc)
                    reward = -1.0
                    info["event"] = "normal_move"
            else:
                reward = -3.0
                info["event"] = "wall_hit"

        # ── Pickup (4) ────────────────────────────
        elif action == ACTION_PICKUP:
            if self.taxi_carrying[agent_id] is not None:
                reward = -2.0
                info["event"] = "pickup_failed_already_has"
            else:
                taxi_idx = self._walkable_idx[self.taxi_positions[agent_id]]

                pickup_pid = None
                for pid in range(self.n_passengers):
                    if self.passenger_locs[pid] == taxi_idx and self.passenger_owners[pid] is None:
                        pickup_pid = pid
                        break

                if pickup_pid is None:
                    reward = -2.0
                    info["event"] = "pickup_failed_wrong_pos"
                else:
                    self.passenger_owners[pickup_pid] = agent_id
                    self.taxi_carrying[agent_id] = pickup_pid
                    self.passenger_locs[pickup_pid] = IN_TAXI_CODE

                    pickup_idx = self._walkable_idx[self.taxi_positions[agent_id]]
                    dest_idx = self.passenger_dests[pickup_pid]
                    self.passenger_dists[pickup_pid] = self._get_shortest_path_idx(
                        pickup_idx, dest_idx
                    )

                    reward = 5.0 + self.reward_alpha * self.passenger_dists[pickup_pid]
                    info["event"] = "pickup_success"
                    info["passenger_id"] = pickup_pid
                    info["bfs_dist"] = self.passenger_dists[pickup_pid]

        # ── Dropoff (5) ────────────────────────────
        elif action == ACTION_DROPOFF:
            carrying_pid = self.taxi_carrying[agent_id]

            if carrying_pid is None:
                reward = -2.0
                info["event"] = "dropoff_failed_no_passenger"
            elif self._walkable_idx[self.taxi_positions[agent_id]] == self.passenger_dests[carrying_pid]:
                reward = self.base_reward

                self.passenger_owners[carrying_pid] = None
                self.passenger_locs[carrying_pid] = DELIVERED_CODE
                self.taxi_carrying[agent_id] = None

                self.delivered_count += 1
                self.delivered = (self.delivered_count == self.n_passengers)

                info["event"] = "dropoff_success"
                info["passenger_id"] = carrying_pid
                info["delivery_reward"] = reward
                info["bfs_dist"] = self.passenger_dists[carrying_pid]
            else:
                reward = -10.0
                info["event"] = "dropoff_wrong_location"
                
        else:
            reward = -2.0
            np.info["event"] = "invalid_action"

        return reward, info

    # ──────────────────────────────────────────
    #  step receives actions from each taxi, returns rewards for each taxi
    # ──────────────────────────────────────────
    # When competing for passengers in the same step, execute in agent_ids order; earlier taxis succeed first
    # If steps run out but passengers not delivered, deduct 5 points extra from each taxi
    def step(
        self, action_dict: Dict[str, int]
    ) -> Tuple[Dict[str, Tuple], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        # sourcery skip: low-code-quality
        """
        Execute one step.

        Parameters
        ----
        action_dict: {"taxi_1": action, "taxi_2": action, ...}

        Returns
        ----
        next_obs, rewards, dones, infos
        """
        self.step_count += 1
        rewards: Dict[str, float] = {}
        infos: Dict[str, Any] = {}

        # ── Passenger Competition ──────────────────────────
        # Execute actions in agent_ids order; if competing for passenger in same step, earlier taxis succeed first
        for agent_id in self.agent_ids:
            if self.delivered:
                rewards[agent_id] = 0.0
                infos[agent_id] = {"event": "episode_already_done"}
                continue

            action = action_dict.get(agent_id, ACTION_SOUTH)
            reward, info = self._apply_action(agent_id, action)
            rewards[agent_id] = reward
            infos[agent_id] = info

        # Multi-taxi overlap penalty: encourage zonal coordination, reduce multiple taxis crowding in same cell
        cell_to_agents: Dict[Tuple[int, int], List[str]] = {}
        for aid, pos in self.taxi_positions.items():
            cell_to_agents.setdefault(pos, []).append(aid)

        for pos, agents in cell_to_agents.items():
            if len(agents) >= 2:
                for aid in agents:
                    rewards[aid] -= self.overlap_penalty
                    infos[aid]["overlap_at"] = pos
                    infos[aid]["overlap_count"] = len(agents)
                    base_event = infos[aid].get("event", "")
                    infos[aid]["event"] = f"{base_event}+overlap" if base_event else "overlap"

        done_task = self.delivered
        timeout   = self.step_count >= self.max_steps
        done_all  = done_task or timeout

        if timeout and not done_task:
            for agent_id in self.agent_ids:
                rewards[agent_id] -= 5.0
                infos[agent_id]["event"] = infos[agent_id].get("event", "") + "+timeout"

        next_obs = {aid: self._get_obs(aid) for aid in self.agent_ids}
        dones    = {aid: done_all for aid in self.agent_ids}
        dones["__all__"] = done_all

        return next_obs, rewards, dones, infos


    # ──────────────────────────────────────────
    #  Pure Visualization
    # ──────────────────────────────────────────
    def render(self) -> None:
        # sourcery skip: assign-if-exp, swap-if-else-branches
        sym = {CELL_ROAD:"·", CELL_WALL:"█", CELL_ARTERIAL:"═", CELL_HOTSPOT:"★"}
        display = [[sym[int(self.grid[r, c])] for c in range(self.n_cols)]
                for r in range(self.n_rows)]

        # Mark passengers not yet picked up
        for pid in range(self.n_passengers):
            if self.passenger_locs[pid] >= 0:
                pr, pc = self._walkable[self.passenger_locs[pid]]
                display[pr][pc] = f"P{pid+1}"

        # Mark destinations of incomplete passengers
        for pid in range(self.n_passengers):
            if self.passenger_locs[pid] != DELIVERED_CODE:
                dr, dc = self._walkable[self.passenger_dests[pid]]
                if display[dr][dc] in {"·", "═", "★"}:
                    display[dr][dc] = f"D{pid+1}"

        # Mark taxis
        for i, agent_id in enumerate(self.agent_ids, start=1):
            tr, tc = self.taxi_positions[agent_id]
            if self.taxi_carrying[agent_id] is not None:
                display[tr][tc] = f"C{i}"
            else:
                display[tr][tc] = f"T{i}"

        phase = f"[Delivered {self.delivered_count}/{self.n_passengers}]"
        print(f"\n── Step {self.step_count} ──  {phase}")
        print("  Col:", " ".join(f"{c:>2}" for c in range(self.n_cols)))
        for r in range(self.n_rows):
            print(f"Row{r}:", " ".join(f"{cell:>2}" for cell in display[r]))
        print()
