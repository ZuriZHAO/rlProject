"""
web_visualizer.py
=================
Run a lightweight Python web server to visualize the multi-agent Taxi environment
in a browser with Canvas rendering.

Usage:
    python web_visualizer.py --n_taxis 2 --n_passengers 3 --port 8765

Open the browser at http://localhost:8765
"""

import argparse
import json
import threading
import time
import os
import pickle
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from multi_taxi_env import MultiTaxiEnv, ACTION_SOUTH, ACTION_NORTH, ACTION_EAST, ACTION_WEST, ACTION_PICKUP, ACTION_DROPOFF, N_ACTIONS, IN_TAXI_CODE, DELIVERED_CODE

STATIC_DIR = Path(__file__).resolve().parent / "web_static"

ACTION_MAP = {
    "south": ACTION_SOUTH,
    "north": ACTION_NORTH,
    "east": ACTION_EAST,
    "west": ACTION_WEST,
    "pickup": ACTION_PICKUP,
    "dropoff": ACTION_DROPOFF,
}

MOVE_DELTA = {
    ACTION_SOUTH: (1, 0),
    ACTION_NORTH: (-1, 0),
    ACTION_EAST: (0, 1),
    ACTION_WEST: (0, -1),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-agent Taxi Web Visualizer")
    parser.add_argument("--n_taxis", type=int, default=2, help="Number of taxis")
    parser.add_argument("--n_passengers", type=int, default=3, help="Number of passengers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for environment")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (use 0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int, default=8765, help="Web server port")
    parser.add_argument("--step_delay", type=float, default=0.4, help="Auto play step delay in seconds")
    parser.add_argument("--overlap_penalty", type=float, default=4.0)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument(
        "--qtable",
        type=str,
        default=None,
        help="Path to trained Q-table (qtables.pkl). Defaults to local qtables.pkl when present.",
    )
    return parser.parse_args()


def bfs_next_action(env, start, goal):
    rows, cols = env.n_rows, env.n_cols
    grid = env.grid
    visited = {start}
    queue = [(start, [])]
    while queue:
        (r, c), path = queue.pop(0)
        if (r, c) == goal:
            if not path:
                return ACTION_PICKUP if env.taxi_positions else ACTION_SOUTH
            return path[0]
        for action, (dr, dc) in [(ACTION_SOUTH, (1, 0)), (ACTION_NORTH, (-1, 0)), (ACTION_EAST, (0, 1)), (ACTION_WEST, (0, -1))]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue
            if (nr, nc) in visited:
                continue
            visited.add((nr, nc))
            queue.append(((nr, nc), path + [action]))
    return ACTION_SOUTH


def choose_actions(env):
    actions = {}
    for aid in env.agent_ids:
        pos = env.taxi_positions[aid]
        carrying = env.taxi_carrying[aid]

        if carrying is not None:
            dest_cell = env._walkable[env.passenger_dests[carrying]]
            actions[aid] = ACTION_DROPOFF if _dropoff_legal(env, aid) else bfs_next_action(env, pos, dest_cell)
            continue

        target_pid = env._get_assigned_waiting_passenger(aid)
        if target_pid is None:
            actions[aid] = ACTION_SOUTH
            continue

        target_cell = env._walkable[env.passenger_locs[target_pid]]
        actions[aid] = ACTION_PICKUP if pos == target_cell else bfs_next_action(env, pos, target_cell)
    return actions

def greedy_qtable_action(q_table, state):
    """Match test_policy.py: choose the first action among max-Q ties."""
    q_values = q_table.get(state)
    if q_values is None:
        return None
    max_q = np.max(q_values)
    best_actions = np.where(q_values == max_q)[0]
    return int(best_actions[0])


def _pickup_legal(env: MultiTaxiEnv, aid: str) -> bool:  # sourcery skip: use-any, use-next
    if env.taxi_carrying[aid] is not None:
        return False
    taxi_idx = env._walkable_idx[env.taxi_positions[aid]]
    for pid in range(env.n_passengers):
        if env.passenger_locs[pid] == taxi_idx and env.passenger_owners[pid] is None:
            return True
    return False


def _dropoff_legal(env: MultiTaxiEnv, aid: str) -> bool:
    carrying_pid = env.taxi_carrying[aid]
    if carrying_pid is None:
        return False
    taxi_idx = env._walkable_idx[env.taxi_positions[aid]]
    return taxi_idx == env.passenger_dests[carrying_pid]


def greedy_qtable_action_safe(env: MultiTaxiEnv, aid: str, q_table, state):
    # If we are at a real pickup/dropoff opportunity, do it.
    if _pickup_legal(env, aid):
        return ACTION_PICKUP
    if _dropoff_legal(env, aid):
        return ACTION_DROPOFF

    q_values = q_table.get(state)
    if q_values is None:
        return None
    q_values = np.asarray(q_values, dtype=np.float64).copy()

    # Mask obviously illegal actions.
    if not _pickup_legal(env, aid):
        q_values[ACTION_PICKUP] = -np.inf
    if not _dropoff_legal(env, aid):
        q_values[ACTION_DROPOFF] = -np.inf

    max_q = np.max(q_values)
    best_actions = np.where(q_values == max_q)[0]
    return int(best_actions[0])


def choose_actions_with_qtable(env, q_tables):  # sourcery skip: low-code-quality, use-named-expression
    """Choose actions using trained Q-table (greedy, no exploration)"""
    actions = {}
    for aid in env.agent_ids:
        pos = env.taxi_positions[aid]
        carrying = env.taxi_carrying[aid]

        obs = env._get_obs(aid)
        state_key = tuple(obs)

        # Carrying passenger: bias toward making progress to destination.
        if carrying is not None:
            dest_cell = env._walkable[env.passenger_dests[carrying]]
            if _dropoff_legal(env, aid):
                actions[aid] = ACTION_DROPOFF
                continue

            q_table = q_tables.get(aid, {}) if q_tables else {}
            q_values = q_table.get(state_key)
            if q_values is None:
                actions[aid] = bfs_next_action(env, pos, dest_cell)
                continue

            q_values = np.asarray(q_values, dtype=np.float64)
            current_dist = env._get_shortest_path_length(pos, dest_cell)

            picked = None
            for cand in np.argsort(-q_values):
                cand = int(cand)
                if cand in (ACTION_PICKUP, ACTION_DROPOFF):
                    continue
                if cand not in MOVE_DELTA:
                    continue
                dr, dc = MOVE_DELTA[cand]
                nr, nc = pos[0] + dr, pos[1] + dc
                if not (0 <= nr < env.n_rows and 0 <= nc < env.n_cols):
                    continue
                if env.grid[nr, nc] == 1:
                    continue
                next_pos = (nr, nc)
                next_dist = env._get_shortest_path_length(next_pos, dest_cell)
                if next_dist < current_dist:
                    picked = cand
                    break

            actions[aid] = picked if picked is not None else bfs_next_action(env, pos, dest_cell)
            continue

        # Not carrying: normal greedy + BFS fallback.
        action = greedy_qtable_action_safe(env, aid, q_tables.get(aid, {}), state_key) if q_tables else None
        if action is not None:
            actions[aid] = action
            continue

        target_pid = env._get_assigned_waiting_passenger(aid)
        if target_pid is None:
            actions[aid] = ACTION_SOUTH
            continue

        target_cell = env._walkable[env.passenger_locs[target_pid]]
        actions[aid] = ACTION_PICKUP if pos == target_cell else bfs_next_action(env, pos, target_cell)
  
    return actions


def env_state(env, playing=False, policy_name="none", last_actions=None, last_infos=None):
    # sourcery skip: inline-immediately-returned-variable
    taxi_positions = [{"id": aid, "pos": env.taxi_positions[aid], "carrying": env.taxi_carrying[aid] is not None} for aid in env.agent_ids]
    passengers = []
    for pid in range(env.n_passengers):
        # Determine status based on passenger_locs value
        # DELIVERED_CODE (-2): delivered
        # IN_TAXI_CODE (-1): in taxi
        # >= 0: waiting (walkable index)
        if env.passenger_locs[pid] == DELIVERED_CODE:
            status = "delivered"
            cell = env._walkable[env.passenger_dests[pid]]
        elif env.passenger_locs[pid] == IN_TAXI_CODE:
            status = "in_taxi"
            owner = env.passenger_owners[pid]
            cell = env.taxi_positions[owner] if owner is not None else None
        else:  # env.passenger_locs[pid] >= 0
            status = "waiting"
            cell = env._walkable[env.passenger_locs[pid]]
        
        passengers.append({"id": pid, "status": status, "cell": cell, "dest": env._walkable[env.passenger_dests[pid]]})
    
    last_events = {}
    if isinstance(last_infos, dict):
        for aid, info in last_infos.items():
            if isinstance(info, dict):
                last_events[aid] = info.get("event")

    result = {
        "grid": env.grid.tolist(),
        "n_rows": env.n_rows,
        "n_cols": env.n_cols,
        "taxi_positions": taxi_positions,
        "passengers": passengers,
        "step_count": env.step_count,
        "delivered_count": env.delivered_count,
        "n_taxis": env.n_taxis,
        "n_passengers": env.n_passengers,
        "done": env.delivered or env.step_count >= env.max_steps,
        "playing": playing,
        "policy": policy_name,
        "last_actions": last_actions or {},
        "last_events": last_events,
    }
    return result


class State:
    def __init__(self, args):
        self.args = args
        self.q_tables = None
        self.policy_name = "bfs"

        # Prefer command-line path, then try qtables.pkl in the same directory
        qtable_path = args.qtable
        if qtable_path is None:
            default_qtable = Path(__file__).resolve().parent / "qtables.pkl"
            if default_qtable.exists():
                qtable_path = str(default_qtable)

        if qtable_path and os.path.exists(qtable_path):
            with open(qtable_path, "rb") as f:
                q_tables_dict = pickle.load(f)

            self.q_tables = {}
            for aid, q_dict in q_tables_dict.items():
                # Training script saves as regular dict, convert to tuple-key dict for visualization
                self.q_tables[aid] = {tuple(state): np.asarray(values) for state, values in q_dict.items()}
            self.policy_name = "qtable-greedy"
            print(f"Loaded Q-table: {qtable_path}")
            print(f"Q-table sizes: {[len(q) for q in self.q_tables.values()]}")
        else:
            print("No Q-table detected, using BFS fallback policy.")

        self.env = MultiTaxiEnv(
            n_taxis=args.n_taxis,
            n_passengers=args.n_passengers,
            congestion_prob=0.3,
            reward_alpha=2.0,
            base_reward=20.0,
            max_steps=200,
            overlap_penalty=args.overlap_penalty,
            seed=args.seed,
        )
        self.env.reset()
        self.playing = False  # Default to not auto-play
        self.delay = args.step_delay
        self.lock = threading.Lock()
        self.last_actions = {}
        self.last_infos = {}
        self.step_result = env_state(self.env, self.playing, self.policy_name, self.last_actions, self.last_infos)
        print("Visualizer initialized paused. Click Play to run the policy.")

    def reset(self):
        with self.lock:
            self.env = MultiTaxiEnv(
                n_taxis=self.args.n_taxis,
                n_passengers=self.args.n_passengers,
                congestion_prob=0.3,
                reward_alpha=2.0,
                base_reward=20.0,
                max_steps=200,
                overlap_penalty=self.args.overlap_penalty,
                seed=self.args.seed,
            )
            self.env.reset()
            self.playing = False  # Pause after reset
            self.last_actions = {}
            self.last_infos = {}
            self.step_result = env_state(self.env, self.playing, self.policy_name, self.last_actions, self.last_infos)
            print("State reset")

    def step(self):
        with self.lock:
            if self.step_result["done"]:
                return self.step_result
            if self.q_tables:
                actions = choose_actions_with_qtable(self.env, self.q_tables)
            else:
                actions = choose_actions(self.env)
            next_obs, rewards, dones, infos = self.env.step(actions)
            self.last_actions = dict(actions)
            self.last_infos = dict(infos)
            if dones.get("__all__"):
                self.playing = False
            self.step_result = env_state(self.env, self.playing, self.policy_name, self.last_actions, self.last_infos)
            return self.step_result

    def get_state(self):
        with self.lock:
            return self.step_result

    def set_play(self, value: bool):  # sourcery skip: remove-unnecessary-cast
        with self.lock:
            self.playing = bool(value) and not self.step_result["done"]
            self.step_result = env_state(self.env, self.playing, self.policy_name, self.last_actions, self.last_infos)
            print(f"Set play: {value}")

    def tick(self):
        while True:
            try:
                with self.lock:
                    playing = self.playing
                if playing:
                    # step() handles its own locking, cannot call while holding lock or deadlock occurs.
                    self.step()
                time.sleep(self.delay)
            except Exception as e:
                print(f"Error in tick thread: {e}")
                time.sleep(1)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "MultiTaxiWeb/0.1"
    state_obj: State = None

    def _send_json(self, obj, status=200):  # sourcery skip: class-extract-method
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_file(self, path, content_type="text/html"):
        try:
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404, "File not found")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file(STATIC_DIR / "index.html")
            return
        if parsed.path == "/state":
            print("HTTP GET /state")
            self._send_json(self.state_obj.get_state())
            return
        if parsed.path == "/control":
            params = parse_qs(parsed.query)
            cmd = params.get("cmd", [None])[0]
            print(f"HTTP GET /control?cmd={cmd}")
            if cmd == "play":
                self.state_obj.set_play(True)
            elif cmd == "pause":
                self.state_obj.set_play(False)
            elif cmd == "step":
                self.state_obj.step()
            elif cmd == "reset":
                self.state_obj.reset()
            state = self.state_obj.get_state()
            self._send_json({"status": "ok", "state": state})
            return
        if parsed.path.startswith("/static/"):
            file_path = STATIC_DIR / parsed.path[len("/static/"):]
            content_type = "application/javascript" if file_path.suffix == ".js" else "text/css" if file_path.suffix == ".css" else "text/plain"
            self._serve_file(file_path, content_type=content_type)
            return
        self.send_error(404, "Not found")


def run_server(args):
    state = State(args)
    RequestHandler.state_obj = state
    thread = threading.Thread(target=state.tick, daemon=True)
    thread.start()

    address = (args.host, args.port)
    try:
        httpd = HTTPServer(address, RequestHandler)
    except OSError as e:
        host, port = address
        print(f"Unable to bind to port {port} (host='{host or '0.0.0.0'}'). Common reasons: port occupied or permission restrictions.")
        print(f"OSError: {e}")
        raise

    print(f"Web visualizer running at http://{args.host}:{args.port}")
    httpd.serve_forever()


def main():
    args = parse_args()
    run_server(args)


if __name__ == "__main__":
    main()
