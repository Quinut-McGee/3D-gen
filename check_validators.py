#!/usr/bin/env python3
"""Check validators on subnet 17"""
import bittensor as bt

# Connect to subnet 17
subtensor = bt.subtensor(network="finney")
metagraph = subtensor.metagraph(netuid=17)

print(f"Metagraph: {metagraph}")
print(f"Total neurons: {metagraph.n}")
print(f"\nValidator Analysis:")
print("=" * 100)

# Min stake requirement from miner config
MIN_STAKE = 1000

validators_found = 0
serving_validators = 0
high_stake_validators = 0
available_validators = 0

for uid in range(metagraph.n):
    stake = metagraph.S[uid]
    is_serving = metagraph.axons[uid].is_serving
    hotkey = metagraph.hotkeys[uid]

    # Check if this is a validator (has stake and serving)
    if stake > 0:
        validators_found += 1

        if is_serving:
            serving_validators += 1

        if stake >= MIN_STAKE:
            high_stake_validators += 1

        if is_serving and stake >= MIN_STAKE:
            available_validators += 1
            print(f"✅ UID {uid:3d} | Stake: {stake:10.2f} TAO | Serving: {is_serving} | Hotkey: {hotkey[:20]}...")
        elif stake >= MIN_STAKE:
            print(f"⚠️  UID {uid:3d} | Stake: {stake:10.2f} TAO | Serving: {is_serving} | Hotkey: {hotkey[:20]}... (NOT SERVING)")

print("\n" + "=" * 100)
print(f"\nSummary:")
print(f"  Total neurons: {metagraph.n}")
print(f"  Validators (stake > 0): {validators_found}")
print(f"  Serving validators: {serving_validators}")
print(f"  High-stake validators (>= {MIN_STAKE} TAO): {high_stake_validators}")
print(f"  Available validators (serving + high stake): {available_validators}")
print(f"\n🔍 Subnet owner hotkey: 5E7eSeRr2aHzCV7SkY4a2Pi5NXHrU4anZz3phEQgn4HCen2B")

# Check if subnet owner is in metagraph
owner_hotkey = "5E7eSeRr2aHzCV7SkY4a2Pi5NXHrU4anZz3phEQgn4HCen2B"
if owner_hotkey in metagraph.hotkeys:
    owner_uid = metagraph.hotkeys.index(owner_hotkey)
    print(f"✅ Subnet owner found at UID {owner_uid}")
    print(f"   Stake: {metagraph.S[owner_uid]:.2f} TAO")
    print(f"   Serving: {metagraph.axons[owner_uid].is_serving}")
else:
    print(f"❌ Subnet owner NOT FOUND in metagraph")
