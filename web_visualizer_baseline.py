"""
web_visualizer_baseline.py
==========================
Baseline web visualizer for the same MultiTaxiEnv settings and the same UI.

Policy:
  - epsilon-greedy with a weak hand-coded greedy rule
  - with probability epsilon: choose a random action
  - otherwise: move to reduce Manhattan distance to current goal (from env obs)
  - pickup/dropoff are forced when they are legally available

Usage:
    python web_visualizer_baseline.py --n_taxis 2 --n_passengers 3 --port 8766 --epsilon 0.5
Open:
    http://127.0.0.1:8766
"""

import argparse
import json
import threading
import time
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from multi_taxi_env import (
    MultiTaxiEnv,
    ACTION_SOUTH,
    ACTION_NORTH,
    ACTION_EAST,
    ACTION_WEST,
    ACTION_PICKUP,
    ACTION_DROPOFF,
    N_ACTIONS,
    IN_TAXI_CODE,
    DELIVERED_CODE,
)

STATIC_DIR = Path(__file__).resolve().parent / "web_static"

MOVE_DELTA = {
    ACTION_SOUTH: (1, 0),
    ACTION_NORTH: (-1, 0),
    ACTION_EAST: (0, 1),
    ACTION_WEST: (0, -1),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-agent Taxi Web Visualizer (Baseline)")
    parser.add_argument("--n_taxis", type=int, default=2, help="Number of taxis")
    parser.add_argument("--n_passengers", type=int, default=3, help="Number of passengers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for environment")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (use 0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int, default=8766, help="Web server port")
    parser.add_argument("--step_delay", type=float, default=0.4, help="Auto play step delay in seconds")
    parser.add_argument("--epsilon", type=float, default=0.5, help="Exploration rate for epsilon-greedy baseline")
    return parser.parse_args()


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


def _legal_moves(env: MultiTaxiEnv, pos):
    r, c = pos
    moves = []
    for action, (dr, dc) in MOVE_DELTA.items():
        nr, nc = r + dr, c + dc
        if not (0 <= nr < env.n_rows and 0 <= nc < env.n_cols):
            continue
        if env.grid[nr, nc] == 1:
            continue
        moves.append(action)
    return moves


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def baseline_action(env: MultiTaxiEnv, aid: str, rng: np.random.Generator, epsilon: float) -> int:
    # sourcery skip: remove-unnecessary-cast
    # Force pickup/dropoff when legally available (keeps baseline sane and comparable).
    if _pickup_legal(env, aid):
        return ACTION_PICKUP
    if _dropoff_legal(env, aid):
        return ACTION_DROPOFF

    # Explore: uniform random over all actions (including pickup/dropoff, which will just fail sometimes).
    if rng.random() < float(epsilon):
        return int(rng.integers(0, N_ACTIONS))

    # Exploit (weak greedy): move to reduce Manhattan distance to current goal from observation.
    tr, tc, gr, gc, *_ = env._get_obs(aid)
    pos = (tr, tc)
    goal = (gr, gc)

    legal_moves = _legal_moves(env, pos)
    if not legal_moves:
        return ACTION_SOUTH

    best = []
    best_d = _manhattan(pos, goal)
    for action in legal_moves:
        dr, dc = MOVE_DELTA[action]
        next_pos = (pos[0] + dr, pos[1] + dc)
        d = _manhattan(next_pos, goal)
        if d < best_d:
            best_d = d
            best = [action]
        elif d == best_d:
            # Keep ties to stay weak and a bit noisy.
            best.append(action)

    candidates = best if best else legal_moves
    return int(rng.choice(candidates))


def choose_actions_baseline(env: MultiTaxiEnv, rng: np.random.Generator, epsilon: float):
    return {aid: baseline_action(env, aid, rng, epsilon) for aid in env.agent_ids}


def env_state(env, playing=False, policy_name="none"):
    taxi_positions = [
        {"id": aid, "pos": env.taxi_positions[aid], "carrying": env.taxi_carrying[aid] is not None}
        for aid in env.agent_ids
    ]
    passengers = []
    for pid in range(env.n_passengers):
        if env.passenger_locs[pid] == DELIVERED_CODE:
            status = "delivered"
            cell = env._walkable[env.passenger_dests[pid]]
        elif env.passenger_locs[pid] == IN_TAXI_CODE:
            status = "in_taxi"
            owner = env.passenger_owners[pid]
            cell = env.taxi_positions[owner] if owner is not None else None
        else:
            status = "waiting"
            cell = env._walkable[env.passenger_locs[pid]]
        passengers.append({"id": pid, "status": status, "cell": cell, "dest": env._walkable[env.passenger_dests[pid]]})

    return {
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
    }


class State:
    def __init__(self, args):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.epsilon = float(args.epsilon)
        self.policy_name = f"baseline-eps{self.epsilon:g}"

        # Keep env settings identical to train.py / existing visualizer defaults.
        self.env = MultiTaxiEnv(
            n_taxis=args.n_taxis,
            n_passengers=args.n_passengers,
            congestion_prob=0.3,
            reward_alpha=2.0,
            base_reward=20.0,
            max_steps=200,
            seed=args.seed,
        )
        self.env.reset()

        self.playing = False
        self.delay = args.step_delay
        self.lock = threading.Lock()
        self.step_result = env_state(self.env, self.playing, self.policy_name)
        print("Baseline visualizer initialized paused. Click Play to run the baseline policy.")

    def reset(self):
        with self.lock:
            self.env = MultiTaxiEnv(
                n_taxis=self.args.n_taxis,
                n_passengers=self.args.n_passengers,
                congestion_prob=0.3,
                reward_alpha=2.0,
                base_reward=20.0,
                max_steps=200,
                seed=self.args.seed,
            )
            self.env.reset()
            self.playing = False
            self.step_result = env_state(self.env, self.playing, self.policy_name)
            print("State reset")

    def step(self):
        with self.lock:
            if self.step_result["done"]:
                return self.step_result
            actions = choose_actions_baseline(self.env, self.rng, self.epsilon)
            _, _, dones, _ = self.env.step(actions)
            if dones.get("__all__"):
                self.playing = False
            self.step_result = env_state(self.env, self.playing, self.policy_name)
            return self.step_result

    def get_state(self):
        with self.lock:
            return self.step_result

    def set_play(self, value: bool):  # sourcery skip: remove-unnecessary-cast
        with self.lock:
            self.playing = bool(value) and not self.step_result["done"]
            self.step_result = env_state(self.env, self.playing, self.policy_name)
            print(f"Set play: {value}")

    def tick(self):
        while True:
            try:
                with self.lock:
                    playing = self.playing
                if playing:
                    self.step()
                time.sleep(self.delay)
            except Exception as e:
                print(f"Error in tick thread: {e}")
                time.sleep(1)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "MultiTaxiWebBaseline/0.1"
    state_obj: State = None

    def _send_json(self, obj, status=200):
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
            self._send_json(self.state_obj.get_state())
            return
        if parsed.path == "/control":
            params = parse_qs(parsed.query)
            cmd = params.get("cmd", [None])[0]
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
            content_type = (
                "application/javascript"
                if file_path.suffix == ".js"
                else "text/css"
                if file_path.suffix == ".css"
                else "text/plain"
            )
            self._serve_file(file_path, content_type=content_type)
            return
        self.send_error(404, "Not found")


def run_server(args):
    state = State(args)
    RequestHandler.state_obj = state
    thread = threading.Thread(target=state.tick, daemon=True)
    thread.start()

    address = (args.host, args.port)
    httpd = HTTPServer(address, RequestHandler)
    print(f"Baseline visualizer running at http://{args.host}:{args.port}")
    httpd.serve_forever()


def main():
    args = parse_args()
    run_server(args)


if __name__ == "__main__":
    main()

