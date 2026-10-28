"""
test_env.py
===========
MultiTaxiEnv Quick Validation Script
Run: python test_env.py
"""

from multi_taxi_env import MultiTaxiEnv

# ── Initialize Environment ──────────────────────────────
env = MultiTaxiEnv(n_taxis=2, n_passengers=3, seed=42)

# ── Reset ───────────────────────────────────
states = env.reset()
print("=" * 40)
print("Initial states:", states)
env.render()

# ── Run a few steps to see output ───────────────────────────
actions = [
    {"taxi_1": 0, "taxi_2": 1},   # taxi_1 south
    {"taxi_1": 2, "taxi_2": 3},   # east
    {"taxi_1": 4, "taxi_2": 5},   # try pickup
    {"taxi_1": 0, "taxi_2": 1},   # move south
    {"taxi_1": 5, "taxi_2": 4},   # try dropoff
]

for i, action_dict in enumerate(actions):
    next_states, rewards, dones, infos = env.step(action_dict)
    print(f"Step {i+1} | Actions: {action_dict}")
    print(f"  Rewards:   {rewards}")
    print(f"  Dones:   {dones}")
    print(f"  Events:   { {k: v.get('event') for k, v in infos.items()} }")
    env.render()
    if dones["__all__"]:
        print(">>> Episode ended <<<")
        break
