import time
import weakref
import random

import bittensor as bt
from common import owner


# COMPETITIVE MINING: Blacklist WC (Weight Copy) validators
# These validators don't provide rewards, waste compute
# Add UIDs discovered from network monitoring
BLACKLISTED_VALIDATORS = [
    180,  # UID 180 is mentioned as WC in Discord FAQ - confirmed non-rewarding
    199,  # Re-blacklisted - gave Score=0.0 on first submission (Gen #11)
]

# PHASE 7 SPEED OPTIMIZATION: Validator Reliability Weighting (Nov 11, 2025)
# Based on Tier 1 data analysis (17 generations, 02:49-03:30)
# Strategy: Bias selection toward reliable validators to reduce failure rate 18% → 12-15%
# This is NOT a blacklist - unreliable validators still get tasks, just less frequently
VALIDATOR_RELIABILITY_SCORES = {
    27:  1.00,  # 0% failure rate (4/4 success) - RELIABLE ✅
    142: 1.00,  # 0% failure rate (3/3 success) - RELIABLE ✅
    49:  0.75,  # 25% failure rate (3/4 success) - UNPREDICTABLE ⚠️
    81:  0.67,  # 33% failure rate (2/3 success) - UNPREDICTABLE ⚠️
    128: 0.67,  # 33% failure rate (2/3 success) - UNPREDICTABLE ⚠️
    # Default for unknown validators: 0.5 (neutral, no bias)
}


class ValidatorSelector:
    """Encapsulates validator selection with blacklisting."""

    def __init__(self, metagraph: bt.metagraph, min_stake: int) -> None:
        self._metagraph_ref = weakref.ref(metagraph)
        self._min_stake = min_stake
        self._cooldowns: dict[int | None, int] = {}
        self._next_uid = 0
        self._blacklist = set(BLACKLISTED_VALIDATORS)

        # Log blacklisted validators
        if self._blacklist:
            bt.logging.info(f"Blacklisted validators: {sorted(self._blacklist)}")

        # Temporary measure.
        # For test period organic traffic will go only through the subnet owner's validator.
        # Subnet owner's validator will be asked more often for tasks to provide enough throughput.
        # Once the testing is done and more validators provide public API, this code will be removed.
        self._ask_owner_in = 5  # turns
        self._owner_hotkey = owner.HOTKEY
        if self._owner_hotkey not in metagraph.hotkeys:
            self._owner_uid = None
        else:
            self._owner_uid = metagraph.hotkeys.index(self._owner_hotkey)

    def get_next_validator_to_query(self) -> int | None:
        current_time = int(time.time())
        metagraph: bt.metagraph = self._metagraph_ref()

        if self._query_subnet_owner(current_time):
            bt.logging.debug("Querying task from the subnet owner")
            return self._owner_uid

        # PHASE 7 SPEED OPTIMIZATION: Weighted Random Selection
        # Collect all eligible validators
        eligible_validators = []
        for uid in range(metagraph.n):
            # Skip blacklisted
            if uid in self._blacklist:
                continue

            # Check eligibility
            is_serving = metagraph.axons[uid].is_serving
            stake = metagraph.S[uid]
            cooldown = self._cooldowns.get(uid, 0)

            if is_serving and stake >= self._min_stake and cooldown < current_time:
                eligible_validators.append(uid)

        # If no eligible validators, return None
        if not eligible_validators:
            bt.logging.info(f"No available validators to pull the task. Min stake required: {self._min_stake}")
            return None

        # Apply reliability weighting
        weights = [VALIDATOR_RELIABILITY_SCORES.get(uid, 0.5) for uid in eligible_validators]

        # Weighted random selection (biases toward reliable validators)
        selected_uid = random.choices(eligible_validators, weights=weights)[0]

        reliability = VALIDATOR_RELIABILITY_SCORES.get(selected_uid, 0.5)
        bt.logging.debug(f"Selected validator [{selected_uid}] (reliability: {reliability:.2f}, stake: {metagraph.S[selected_uid]:.1f})")

        return selected_uid

    def set_cooldown(self, validator_uid: int, cooldown_until: int) -> None:
        self._cooldowns[validator_uid] = cooldown_until

    def is_blacklisted(self, validator_uid: int) -> bool:
        """Check if validator is blacklisted"""
        return validator_uid in self._blacklist

    def add_to_blacklist(self, validator_uid: int) -> None:
        """Add validator to blacklist"""
        self._blacklist.add(validator_uid)
        bt.logging.warning(f"Added validator [{validator_uid}] to blacklist")

    def remove_from_blacklist(self, validator_uid: int) -> None:
        """Remove validator from blacklist"""
        if validator_uid in self._blacklist:
            self._blacklist.remove(validator_uid)
            bt.logging.info(f"Removed validator [{validator_uid}] from blacklist")

    def _query_subnet_owner(self, current_time: int) -> bool:
        if self._cooldowns.get(self._owner_uid, 0) > current_time:
            return False

        if self._ask_owner_in > 1:
            self._ask_owner_in -= 1
            return False

        self._ask_owner_in = 5
        return True
