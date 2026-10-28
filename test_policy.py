"""
test_policy.py
==============
Load and test trained qtables.pkl policy

Features:
  - Load saved Q-table
  - Run greedy policy (ε=0.0), print step-by-step state, action, reward, etc.
  - Finally summarize and display episode results

Run:
    python test_policy.py                    # Default 1 episode
    python test_policy.py --episodes 5 --seed 42
    python test_policy.py --n_taxis 2 --n_passengers 3 --episodes 3
"""

import argparse
import numpy as np
import pickle
import os
from collections import defaultdict
from typing import Dict

from multi_taxi_env import (
    MultiTaxiEnv, N_ACTIONS, ACTION_SOUTH, ACTION_NORTH, 
    ACTION_EAST, ACTION_WEST, ACTION_PICKUP, ACTION_DROPOFF, IN_TAXI_CODE
)

# ──────────────────────────────────────────────
#  Action Name Mapping
# ──────────────────────────────────────────────
ACTION_NAMES = {
    ACTION_SOUTH:   "↓ SOUTH",
    ACTION_NORTH:   "↑ NORTH",
    ACTION_EAST:    "→ EAST",
    ACTION_WEST:    "← WEST",
    ACTION_PICKUP:  "🚗 PICKUP",
    ACTION_DROPOFF: "🏁 DROPOFF",
}


def load_qtables(filepath: str) -> Dict[str, defaultdict]:
    # sourcery skip: dict-assign-update-to-union
    """Load Q-table from pickle file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "rb") as f:
        qtables_dict = pickle.load(f)
    
    # Convert back to defaultdict format
    qtables = {}
    for aid, q_dict in qtables_dict.items():
        q_table = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float64))
        q_table.update(q_dict)
        qtables[aid] = q_table
    
    return qtables


def greedy_action(q_table: defaultdict, state: tuple) -> int:
    """Greedy policy: select action with maximum Q value (kept for compatibility, real evaluation uses greedy_action_safe)"""
    q_values = q_table[state]
    max_q = np.max(q_values)
    best_actions = np.where(q_values == max_q)[0]
    return int(best_actions[0])  # Take first optimal action


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


# def greedy_action_safe(env: MultiTaxiEnv, aid: str, q_table: defaultdict, state: tuple) -> int:
#     """
#     Greedy + common sense constraints:
#     1) Standing on passenger cell and can pickup -> force PICKUP
#     2) Standing on destination and can dropoff -> force DROPOFF
#     3) Otherwise mask illegal PICKUP/DROPOFF, then greedy
#     """
#     if _pickup_legal(env, aid):
#         return ACTION_PICKUP
#     if _dropoff_legal(env, aid):
#         return ACTION_DROPOFF

#     q_values = np.asarray(q_table[state], dtype=np.float64).copy()
#     if not _pickup_legal(env, aid):
#         q_values[ACTION_PICKUP] = -np.inf
#     if not _dropoff_legal(env, aid):
#         q_values[ACTION_DROPOFF] = -np.inf

#     max_q = np.max(q_values)
#     best_actions = np.where(q_values == max_q)[0]
#     return int(best_actions[0])


def test_episode(
    env: MultiTaxiEnv,
    q_tables: Dict[str, defaultdict],
    episode_num: int = 1,
    verbose: bool = True,
) -> tuple:  # sourcery skip: dict-comprehension, extract-duplicate-method, low-code-quality, move-assign, remove-redundant-fstring
    """
    Run one episode using greedy policy (ε=0.0)
    
    Returns: (cumulative raw reward, steps used, whether all passengers successfully delivered)
    """
    obs = env.reset()
    states = {aid: obs[aid] for aid in env.agent_ids}
    ep_return = 0.0
    step = 0

    if verbose:
        print(f"\n{'='*80}")
        print(f"📍 Episode {episode_num} starts")
        print(f"{'='*80}")
        print(f"Environment config: {len(env.agent_ids)} taxis, {env.n_passengers} passengers")
        print(f"Initial observation states:")
        for aid in env.agent_ids:
            print(f"  {aid}: {states[aid]}")
        print()

    while True:
        step += 1

        # Select greedy action for each agent
        actions = {}
        for aid in env.agent_ids:
            # actions[aid] = greedy_action_safe(env, aid, q_tables[aid], states[aid])
            actions[aid] = greedy_action(q_tables[aid], states[aid])
        
        if verbose:
            print(f"┌─ Step {step:3d} ──────────────────────────────────────────────")
            print(f"│ Current state:")
            for aid in env.agent_ids:
                state = states[aid]
                taxi_r, taxi_c, goal_r, goal_c, has_pass, need_div, demand_high = state
                q_vals = q_tables[aid][state]
                chosen_action = actions[aid]
                print(
                    f"│   {aid}:"
                    f" Position=({taxi_r},{taxi_c})"
                    f" Goal=({goal_r},{goal_c})"
                    f" Carrying={'✓' if has_pass else '✗'}"
                    f" Diversion={'Y' if need_div else 'N'}"
                    f" High demand={'Y' if demand_high else 'N'}"
                )
                print(
                    f"│         Q values(SNEWPD)={[f'{q:.2f}' for q in q_vals]}"
                )
                print(
                    f"│         → Chosen: {ACTION_NAMES[chosen_action]}"
                )

        # perform action
        next_obs, rewards, dones, infos = env.step(actions)
        is_done = dones["__all__"]

        if verbose:
            print(f"│ After execution:")
            total_raw_reward = 0
            for aid in env.agent_ids:
                raw_r = rewards[aid]
                total_raw_reward += raw_r
                info_str = infos.get(aid, {})
                if isinstance(info_str, dict):
                    info_str = ", ".join(f"{k}={v}" for k, v in info_str.items())
                print(
                    f"│   {aid}:"
                    f" New position=({next_obs[aid][0]},{next_obs[aid][1]})"
                    f" Reward={raw_r:>7.2f}"
                    f" (info: {info_str})"
                )
            print(f"│ Total reward this step: {total_raw_reward:>7.2f}")
            if is_done:
                print(f"│ ✓ Episode ended")
            print(f"└{'─'*76}")
            print()

        ep_return += sum(rewards.values())
        states = next_obs

        if is_done:
            break

        if step >= env.max_steps:
            if verbose:
                print(f"⚠️  Reached maximum steps limit: {env.max_steps}")
            break

    # ── Summary Results ──
    delivered_count = env.delivered_count  # Use delivered_count instead of boolean
    total_passengers = env.n_passengers
    delivery_rate = (delivered_count / total_passengers) if total_passengers > 0 else 0.0
    is_fully_delivered = (delivered_count == total_passengers)

    if verbose:
        print(f"\n{'='*80}")
        print(f"📊 Episode {episode_num} results summary")
        print(f"{'='*80}")
        print(f"  Total steps:        {step}")
        print(f"  Cumulative reward:      {ep_return:.2f}")
        print(f"  Delivered passengers:      {delivered_count}/{total_passengers} ({delivery_rate*100:.1f}%)")
        print(f"  All delivered:      {'✅ Yes' if is_fully_delivered else '❌ No'}")
        print(f"  Environment step count:  {env.step_count}")
        print(f"{'='*80}\n")

    return ep_return, step, delivered_count


def main():  # sourcery skip: remove-redundant-fstring
    parser = argparse.ArgumentParser(
        description="Test trained Q-table policy"
    )
    parser.add_argument(
        "--qtable-path",
        type=str,
        default=None,
        help="Q-table file path (default: current directory/qtables.pkl)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)"
    )
    parser.add_argument(
        "--n-taxis",
        type=int,
        default=2,
        help="Number of taxis (default: 2)"
    )
    parser.add_argument(
        "--n-passengers",
        type=int,
        default=3,
        help="Number of passengers (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print details, only show summary"
    )

    args = parser.parse_args()

    # ── Determine Q-table path ──
    if args.qtable_path is None:
        args.qtable_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "qtables.pkl"
        )

    # ── Load Q-table ──
    print(f"📂 Loading Q-table: {args.qtable_path}")
    try:
        q_tables = load_qtables(args.qtable_path)
        print(f"✓ Successfully loaded Q-table, containing {len(q_tables)} agents:")
        for aid, q_table in q_tables.items():
            print(f"    {aid}: {len(q_table)} states")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return

    # ── Create Environment ──
    env = MultiTaxiEnv(
        n_taxis=args.n_taxis,
        n_passengers=args.n_passengers,
        congestion_prob=0.3,
        reward_alpha=2.0,
        base_reward=20.0,
        max_steps=200,
        seed=args.seed,
    )

    # ── Check if agents in Q-table match environment agents ──
    q_table_aids = set(q_tables.keys())
    env_aids = set(env.agent_ids)
    if q_table_aids != env_aids:
        print(f"⚠️  Warning: Agents in Q-table {q_table_aids} do not match environment agents {env_aids}")
        print(f"   Continuing test...")

    # ── Run Test ──
    print(f"\nStarting test of {args.episodes} episodes (greedy policy, ε=0.0)\n")

    total_return = 0.0
    total_steps = 0
    total_delivered = 0  # Total delivered passengers

    for ep in range(1, args.episodes + 1):
        ep_r, steps, delivered_count = test_episode(
            env,
            q_tables,
            episode_num=ep,
            verbose=(not args.quiet),
        )
        total_return += ep_r
        total_steps += steps
        total_delivered += delivered_count

    # ── Final Statistics ──
    if args.episodes > 1:
        avg_delivery_rate = (total_delivered / (args.episodes * env.n_passengers)) if env.n_passengers > 0 else 0.0
        print(f"\n{'='*80}")
        print(f"📈 All episodes statistics")
        print(f"{'='*80}")
        print(f"  Episodes run:    {args.episodes}")
        print(f"  Total reward:        {total_return:.2f}")
        print(f"  Average reward/episode: {total_return / args.episodes:.2f}")
        print(f"  Average steps/episode: {total_steps / args.episodes:.1f}")
        print(f"  Delivered passengers:    {total_delivered}/{args.episodes * env.n_passengers}")
        print(f"  Average delivery rate:    {avg_delivery_rate*100:.1f}%")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
