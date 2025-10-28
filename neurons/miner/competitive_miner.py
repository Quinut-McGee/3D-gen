"""
Competitive Miner implementation using async task management.
"""

import asyncio
import copy
import bittensor as bt
from common import create_neuron_dir

from miner.metagraph_sync import MetagraphSynchronizer
from miner.validator_selector import ValidatorSelector
from miner.async_task_manager import AsyncTaskManager
from miner.competitive_workers import process_task_competitive


class CompetitiveMiner:
    """
    Competitive miner with async multi-validator support.

    Key improvements over base miner:
    - Queries ALL validators simultaneously
    - Processes 4+ tasks in parallel
    - Non-blocking submission
    - CLIP validation
    - Validator blacklisting
    """

    def __init__(self, config: bt.config) -> None:
        self.config = copy.deepcopy(config)
        create_neuron_dir(self.config)

        bt.logging.set_config(config=self.config.logging)
        bt.logging.info(f"Starting COMPETITIVE miner with config: {config}")

        # Initialize Bittensor components
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        self._self_check_for_registration()

        self.metagraph = bt.metagraph(
            netuid=self.config.netuid,
            network=self.subtensor.network,
            sync=False,
            lite=True
        )

        # Metagraph sync
        self.metagraph_sync = MetagraphSynchronizer(
            self.metagraph,
            self.subtensor,
            self.config.neuron.sync_interval,
            self.config.neuron.log_info_interval
        )
        self.metagraph_sync.sync()

        bt.logging.info(f"Metagraph: {self.metagraph}")

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running COMPETITIVE miner on subnet {self.config.netuid} "
            f"with UID {self.uid} using {self.subtensor.chain_endpoint}"
        )

        # Validator selector (with blacklisting)
        self.validator_selector = ValidatorSelector(
            self.metagraph,
            self.config.neuron.min_stake_to_set_weights
        )

        # Async task manager
        concurrent_tasks = len(self.config.generation.endpoints)
        bt.logging.info(f"Initializing async task manager with {concurrent_tasks} workers")

        self.task_manager = AsyncTaskManager(
            max_concurrent_tasks=concurrent_tasks,
            max_queue_size=2,  # Conservative queue size for faster delivery (max wait ~40s vs 160s with queue=8)
            pull_interval=5.0  # Reduced from 10.0 to 5.0 for faster task intake (saves ~2.5s avg)
        )

    def _self_check_for_registration(self) -> None:
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            raise RuntimeError(
                f"Wallet {self.wallet} not registered on netuid {self.config.netuid}. "
                "Please register using `btcli subnets register`"
            )

    async def run(self) -> None:
        """
        Run the competitive miner.

        This starts:
        - Async validator polling (queries all validators in parallel)
        - Multiple task workers (process tasks concurrently)
        - Metagraph sync (background)
        """
        bt.logging.info("=" * 60)
        bt.logging.info("🚀 COMPETITIVE MINER STARTING")
        bt.logging.info("=" * 60)
        bt.logging.info(f"UID: {self.uid}")
        bt.logging.info(f"Workers: {self.task_manager.max_concurrent_tasks}")
        bt.logging.info(f"Endpoints: {self.config.generation.endpoints}")
        bt.logging.info(f"Validator polling: every {self.task_manager.pull_interval}s")
        bt.logging.info("=" * 60)

        # Start async task manager
        await asyncio.gather(
            # Main task processing
            self.task_manager.start(
                validator_selector=self.validator_selector,
                metagraph=self.metagraph,
                wallet=self.wallet,
                generation_endpoints=self.config.generation.endpoints,
                validator_class=None,  # CLIP validator (loaded in generation service)
                processor_func=process_task_competitive
            ),

            # Metagraph sync in background
            self._metagraph_sync_loop()
        )

    async def _metagraph_sync_loop(self):
        """Background task to sync metagraph"""
        while True:
            await asyncio.sleep(5)
            self.metagraph_sync.log_info(self.uid)
            self.metagraph_sync.sync()
