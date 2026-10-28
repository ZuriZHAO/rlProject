"""
train.py
========
Multi-vehicles & multiple passengers —— n-Step SARSA + PBRS training Script

State space: The tuple returned by the environment is directly used as the key for the Q-table.
    The current observation is:
    (
        taxi_row, taxi_col,
        goal_row, goal_col,
        has_passenger,
        need_diversion,
        demand_high
    )

Instructions:
    - Goal: When the taxi is empty, it indicates the current nearest passenger position to be picked up; When carrying passengers, indicate the destination
    - need_diversion: Is there any other empty taxi competitor that is equally close or closer to the current target
    - demand_high: Is the current number of pending passengers greater than the number of available empty taxis
    - Q-table still uses defaultdict sparse storage without modifying the training main logic
run:
    python train.py                   # Default 20000 rounds
    python train.py --episodes 6000
    python train.py --n_taxis 2 --n_passengers 3
    python train.py --episodes 200 --render
"""

import argparse
import numpy as np
import pickle
import os
from collections import defaultdict, deque
from typing import Dict, List, Tuple

from multi_taxi_env import (
    MultiTaxiEnv,
    N_ACTIONS,
    MOVE_DELTA,
    ACTION_PICKUP,
    ACTION_DROPOFF,
)

# ══════════════════════════════════════════════
#  Hyperparameters
# ══════════════════════════════════════════════
# The following are "default values", which will be overridden by command-line arguments
DEFAULT_EPISODES = 20000
DEFAULT_N_STEPS  = 4      # n-Step SARSA's n
GAMMA            = 0.97#0.95   # Discount factor
ALPHA            = 0.25#0.2    # Learning rate
EPS_START        = 1.0    # Initial exploration rate
EPS_END          = 0.10#0.05   # Final exploration rate
EPS_DECAY        = 0.99985#0.9993 # Epsilon decay per episode, approximately 5000 steps to reach 0.05
PBRS_ETA         = 0.45# 0.7    # PBRS potential weight; must satisfy η < 1/((1-γ)·d_max) ≈ 1.0, otherwise standing still can yield positive reward
DEFAULT_OVERLAP_PENALTY = 4.0  # Vehicle overlap penalty (same level as base_reward, encourages dispersion)
DEFAULT_MAX_STEPS = 200
DEFAULT_N_TAXIS  = 2      # Number of taxis
DEFAULT_N_PASSENGERS = 3  # Number of passengers

# ══════════════════════════════════════════════
#  Q-table
# ══════════════════════════════════════════════
def make_qtable() -> defaultdict:
    """state(tuple) → Q-value array [6], initially all zeros"""
    return defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float64))

def legal_actions(env: MultiTaxiEnv, agent_id: str) -> List[int]:
    # sourcery skip: hoist-similar-statement-from-if, hoist-statement-from-if, move-assign-in-block
    actions: List[int] = []
    r, c = env.taxi_positions[agent_id]

    for action, (dr, dc) in MOVE_DELTA.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < env.n_rows and 0 <= nc < env.n_cols and env.grid[nr, nc] != 1:
            actions.append(action)

    if env.taxi_carrying[agent_id] is None:
        taxi_idx = env._walkable_idx[env.taxi_positions[agent_id]]
        if any(
            env.passenger_locs[pid] == taxi_idx and env.passenger_owners[pid] is None
            for pid in range(env.n_passengers)
        ):
            actions.append(ACTION_PICKUP)
    else:
        carrying_pid = env.taxi_carrying[agent_id]
        taxi_idx = env._walkable_idx[env.taxi_positions[agent_id]]
        if taxi_idx == env.passenger_dests[carrying_pid]:
            actions.append(ACTION_DROPOFF)

    return actions or list(range(N_ACTIONS))

# ══════════════════════════════════════════════
# ε - growth action selection
# ══════════════════════════════════════════════
def epsilon_greedy(
    q_table: defaultdict,
    state: tuple,
    epsilon: float,
    rng: np.random.Generator,
    legal: List[int],
) -> int:
    if rng.random() < epsilon:
        return int(rng.choice(legal))

    q_values = np.asarray(q_table[state], dtype=np.float64).copy()
    illegal = [a for a in range(N_ACTIONS) if a not in legal]
    q_values[illegal] = -np.inf

    max_q = np.max(q_values)
    if not np.isfinite(max_q):
        return int(rng.choice(legal))

    best_actions = np.where(q_values == max_q)[0]
    return int(rng.choice(best_actions))

# ══════════════════════════════════════════════
# PBRS potential energy function Φ (s)
#
# Φ (s)=- η× BFS shortest path distance (taxi, target)
#
# Goal:
# Empty taxi → passenger waiting position
# Carrying passengers → Destination
# Delivered → 0 (Task completed, potential energy returns to zero)
#
# Shaped reward F=γ ·Φ (s') - Φ (s)
# Approaching the target → Φ (s')>Φ (s) → F>0 → Positive guidance
# ══════════════════════════════════════════════
def compute_potential(env: MultiTaxiEnv, agent_id: str, eta: float) -> float:
    if env.delivered or eta == 0.0:
        return 0.0

    taxi_idx = env._walkable_idx[env.taxi_positions[agent_id]]
    carrying_pid = env.taxi_carrying[agent_id]

    # Carrying passenger: target is the destination of the carried passenger
    if carrying_pid is not None:
        dest_idx = env.passenger_dests[carrying_pid]
        dist = env._get_shortest_path_idx(taxi_idx, dest_idx)
        return -eta * dist  
    
    # Empty taxi: target is the nearest waiting passenger
    target_pid = env._get_assigned_waiting_passenger(agent_id)
    if target_pid is None or env.passenger_locs[target_pid] < 0:
        return 0.0

    pickup_idx = env.passenger_locs[target_pid]
    dest_idx = env.passenger_dests[target_pid]

    dist_to_pickup = env._get_shortest_path_idx(taxi_idx, pickup_idx)
    dist_to_dropoff = env._get_shortest_path_idx(pickup_idx, dest_idx)
    return -eta * (dist_to_pickup + dist_to_dropoff)

# ══════════════════════════════════════════════
#  n-Step SARSA Update
#
#  Paper formula (Sutton & Barto Ch.7):
#    G = Σ_{i=0}^{n-1} γ^i · r̃_{t+i+1}  +  γ^n · Q(s_{t+n}, a_{t+n})
#    Q(s_t, a_t) ← Q(s_t, a_t) + α · [G - Q(s_t, a_t)]
#
#  r̃ = r + γ·Φ(s') - Φ(s)  (PBRS shaped reward)
# ══════════════════════════════════════════════
def nstep_update(
    q_table:     defaultdict,
    buffer:      deque,       # [(state, action, shaped_r), ...]
    next_state:  tuple,
    next_action: int,
    gamma:       float,
    alpha:       float,
    is_done:     bool,
) -> None:
    if not buffer:
        return
    G = sum((gamma ** i) * r for i, (_, _, r) in enumerate(buffer))
    if not is_done:
        G += (gamma ** len(buffer)) * q_table[next_state][next_action]
    s, a, _ = buffer.popleft()
    q_table[s][a] += alpha * (G - q_table[s][a])


# ══════════════════════════════════════════════
#  Single Episode
# ══════════════════════════════════════════════
def run_episode(
    env:     MultiTaxiEnv,
    q_tables: Dict[str, defaultdict],
    epsilon: float,
    n:       int,
    gamma:   float,
    alpha:   float,
    eta:     float,
    rng:     np.random.Generator,
    render:  bool = False,
) -> Tuple[float, int, bool]:  # sourcery skip: low-code-quality
    """
    Multi-taxi multi-passenger: Each taxi learns independently (Independent Learners)

    Returns
    ----
    (Cumulative raw reward, steps used, whether successfully delivered)
    """
    obs     = env.reset()
    states = {aid: obs[aid] for aid in env.agent_ids}
    actions = {
        aid: epsilon_greedy(
            q_tables[aid],
            states[aid],
            epsilon,
            rng,
            legal_actions(env, aid),
        )
        for aid in env.agent_ids
    }
    buffers = {aid: deque() for aid in env.agent_ids}
    potentials = {
        aid: compute_potential(env, aid, eta)
        for aid in env.agent_ids
    }
    ep_return = 0.0

    while True:
        if render:
            env.render()

        next_obs, rewards, dones, infos = env.step(actions)
        is_done    = dones["__all__"]
        
        next_states = {}
        next_actions = {}
        
        for aid in env.agent_ids:
            raw_r = rewards[aid]
            ep_return += raw_r

            phi_next = compute_potential(env, aid, eta)
            shaped_r = raw_r + gamma * phi_next - potentials[aid]
            potentials[aid] = phi_next

            next_state = next_obs[aid]
            if not is_done:
                next_action = epsilon_greedy(
                    q_tables[aid],
                    next_state,
                    epsilon,
                    rng,
                    legal_actions(env, aid),
                )
            else:
                next_action = 0
                
            buffers[aid].append((states[aid], actions[aid], shaped_r))

            if len(buffers[aid]) >= n or is_done:
                nstep_update(
                    q_tables[aid],
                    buffers[aid],
                    next_state,
                    next_action,
                    gamma,
                    alpha,
                    is_done,
                )

            next_states[aid] = next_state
            next_actions[aid] = next_action
            
        states = next_states
        actions = next_actions

        if is_done:
            for aid in env.agent_ids:
                while buffers[aid]:
                    nstep_update(
                        q_tables[aid],
                        buffers[aid],
                        states[aid],
                        actions[aid],
                        gamma,
                        alpha,
                        True,
                    )
            break

    return ep_return, env.step_count, env.delivered


# ══════════════════════════════════════════════
#  Main Training Loop
# ══════════════════════════════════════════════
def train(args) -> None:
    env = MultiTaxiEnv(
        n_taxis=args.n_taxis,
        n_passengers=args.n_passengers,
        congestion_prob=0.3,
        reward_alpha=2.0,
        base_reward=20.0,
        max_steps= args.max_steps,
        overlap_penalty=args.overlap_penalty,
        seed=getattr(args, "seed", 0),
    )
    rng     = np.random.default_rng(getattr(args, "seed", 42))
    q_tables = {aid: make_qtable() for aid in env.agent_ids}
    epsilon = EPS_START

    history_r         = []
    history_steps     = []
    history_delivered = []

    print("\n" + "═" * 58)
    print(f"  Multi-taxi multi-passenger  |  n-Step SARSA + PBRS  |  Taxis={args.n_taxis} |  Passengers={args.n_passengers}")
    print(f"  Episodes={args.episodes}  n={args.n_steps}  γ={GAMMA}  α={ALPHA}")
    print(f"  η={PBRS_ETA}  ε: {EPS_START}→{EPS_END}  Decay={EPS_DECAY}")
    print("═" * 58)

    LOG = max(1, args.episodes // 20)   # Print every 5%

    for ep in range(1, args.episodes + 1):
        ep_r, steps, delivered = run_episode(
            env, q_tables, epsilon,
            args.n_steps, GAMMA, ALPHA, PBRS_ETA, rng,
            render=(args.render and ep % LOG == 0),
        )
        history_r.append(ep_r)
        history_steps.append(steps)
        history_delivered.append(delivered)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if ep % LOG == 0:
            w         = min(LOG, ep)
            avg_r     = np.mean(history_r[-w:])
            avg_steps = np.mean(history_steps[-w:])
            success   = np.mean(history_delivered[-w:])
            total_q   = sum(len(q_tables[aid]) for aid in env.agent_ids)

            print(
                f"  Ep {ep:>5}/{args.episodes}"
                f"  ε={epsilon:.4f}"
                f"  Steps={avg_steps:>6.1f}"
                f"  Success={success:>5.1%}"
                f"  Avg Reward={avg_r:>8.2f}"
                f"  Q-table size={total_q}"
            )
    print("\nTraining completed.")

    if getattr(args, "save_qtable", True):
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtables.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({aid: dict(q_tables[aid]) for aid in env.agent_ids}, f)
        print(f"Q-table saved: {save_path}")

    # ── Greedy Evaluation 200 Episodes ──────────────────────
    # Independent env/rng, results unaffected by training random sequence
    print("\n── Greedy Evaluation (ε=0.01, 50 episodes) ──")
    eval_env = MultiTaxiEnv(
        n_taxis=args.n_taxis, 
        n_passengers=args.n_passengers,
        congestion_prob=0.3,
        reward_alpha=2.0,
        base_reward=20.0,
        max_steps=args.max_steps,
        overlap_penalty=args.overlap_penalty,
        seed=1,
    )
    eval_rng = np.random.default_rng(123)
    eval_r, eval_steps, eval_success = [], [], []
    for _ in range(50):
        r, s, done = run_episode(
            eval_env, q_tables, 0.01,
            args.n_steps, GAMMA, 0.0, 0.0, eval_rng
        )
        eval_r.append(r)
        eval_steps.append(s)
        eval_success.append(done)

    success  = np.mean(eval_success)
    avg_r    = np.mean(eval_r)
    avg_step = np.mean(eval_steps)
    print(f"  Success rate:   {success:.1%}")
    print(f"  Average Steps: {avg_step:.1f}")
    print(f"  Average Reward: {avg_r:.2f}")
    print(f"  Best Reward: {max(eval_r):.2f}")
    print(f"  Worst Reward: {min(eval_r):.2f}")

    return {
        "history": {
            "reward": history_r,
            "steps": history_steps,
            "delivered": history_delivered,
        },
        "eval": {
            "success": success,
            "avg_reward": avg_r,
            "avg_steps": avg_step,
            "max_reward": max(eval_r),
            "min_reward": min(eval_r),
        },
    }


# ══════════════════════════════════════════════
#  Main Entry
# ══════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int,  default=DEFAULT_EPISODES)
    parser.add_argument("--n_steps",  type=int,  default=DEFAULT_N_STEPS)
    parser.add_argument("--n_taxis",  type=int, default=DEFAULT_N_TAXIS)
    parser.add_argument("--n_passengers", type=int, default=DEFAULT_N_PASSENGERS)
    parser.add_argument("--render",   action="store_true")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--overlap_penalty", type=float, default=DEFAULT_OVERLAP_PENALTY)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
 