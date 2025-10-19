import time
import weakref

import bittensor as bt
from common import owner


class ValidatorSelector:
    """Encapsulates validator selection."""

    def __init__(self, metagraph: bt.metagraph, min_stake: int) -> None:
        self._metagraph_ref = weakref.ref(metagraph)
        self._min_stake = min_stake
        self._cooldowns: dict[int | None, int] = {}
        self._next_uid = 0

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

        start_uid = self._next_uid
        checked_validators = 0
        while True:
            is_serving = metagraph.axons[self._next_uid].is_serving
            stake = metagraph.S[self._next_uid]
            cooldown = self._cooldowns.get(self._next_uid, 0)

            if (
                is_serving
                and stake >= self._min_stake
                and cooldown < current_time
            ):
                bt.logging.debug(f"Querying task from [{self._next_uid}]. Stake: {stake}")
                return self._next_uid

            # Log why validator was skipped (only log first full cycle)
            if checked_validators < metagraph.n:
                if not is_serving:
                    bt.logging.debug(f"Validator [{self._next_uid}] not serving")
                elif stake < self._min_stake:
                    bt.logging.debug(f"Validator [{self._next_uid}] stake {stake} < min {self._min_stake}")
                elif cooldown >= current_time:
                    bt.logging.debug(f"Validator [{self._next_uid}] on cooldown until {cooldown}")

            self._next_uid = 0 if self._next_uid + 1 == metagraph.n else self._next_uid + 1
            checked_validators += 1

            if start_uid == self._next_uid:
                bt.logging.info(f"No available validators to pull the task. Checked {checked_validators} validators. Min stake required: {self._min_stake}")
                return None

    def set_cooldown(self, validator_uid: int, cooldown_until: int) -> None:
        self._cooldowns[validator_uid] = cooldown_until

    def _query_subnet_owner(self, current_time: int) -> bool:
        if self._cooldowns.get(self._owner_uid, 0) > current_time:
            return False

        if self._ask_owner_in > 1:
            self._ask_owner_in -= 1
            return False

        self._ask_owner_in = 5
        return True
