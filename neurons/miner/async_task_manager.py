"""
Async multi-validator task manager for competitive mining.

Key improvement from base miner:
- Base: Query validators one at a time (sequential)
- Competitive: Query ALL validators simultaneously (parallel)

This dramatically increases throughput and ensures no missed opportunities.
"""

import asyncio
import time
from typing import List, Optional, Dict, Set
from collections import deque
import bittensor as bt
from common.protocol import ProtocolTask, PullTask, SubmitResults

from loguru import logger


class AsyncTaskManager:
    """
    Manages asynchronous task pulling and processing across multiple validators.

    Features:
    - Parallel validator polling (query all simultaneously)
    - Task queue with priority
    - Concurrent generation workers
    - Non-blocking submission (fire-and-forget)
    - Automatic retry logic
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 4,
        max_queue_size: int = 20,
        pull_interval: float = 10.0
    ):
        """
        Args:
            max_concurrent_tasks: Max parallel generations
            max_queue_size: Max tasks in queue
            pull_interval: Seconds between validator pulls
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_queue_size = max_queue_size
        self.pull_interval = pull_interval

        # Task queue (FIFO)
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)

        # Active task tracking
        self.active_tasks: Dict[str, asyncio.Task] = {}

        # Cooldown tracking per validator
        self.validator_cooldowns: Dict[int, float] = {}

        # Statistics
        self.stats = {
            "tasks_pulled": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_validated": 0,
            "tasks_rejected": 0,
        }

        logger.info(
            f"AsyncTaskManager initialized: "
            f"max_concurrent={max_concurrent_tasks}, "
            f"queue_size={max_queue_size}"
        )

    async def start(
        self,
        validator_selector,
        metagraph: bt.metagraph,
        wallet: bt.wallet,
        generation_endpoints: List[str],
        validator_class,
        processor_func
    ):
        """
        Start the async task manager.

        Args:
            validator_selector: ValidatorSelector instance
            metagraph: Bittensor metagraph
            wallet: Bittensor wallet
            generation_endpoints: List of generation service URLs
            validator_class: CLIPValidator class instance
            processor_func: Async function to process tasks
        """
        # Start background tasks
        tasks = [
            # Pull tasks from validators
            asyncio.create_task(
                self._validator_poller(
                    validator_selector,
                    metagraph,
                    wallet
                )
            ),

            # Process task queue with multiple workers
            *[
                asyncio.create_task(
                    self._task_worker(
                        worker_id=i,
                        endpoint=generation_endpoints[i % len(generation_endpoints)],
                        validator_class=validator_class,
                        processor_func=processor_func,
                        metagraph=metagraph,
                        wallet=wallet
                    )
                )
                for i in range(self.max_concurrent_tasks)
            ],

            # Stats logger
            asyncio.create_task(self._stats_logger())
        ]

        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _validator_poller(
        self,
        validator_selector,
        metagraph: bt.metagraph,
        wallet: bt.wallet
    ):
        """
        Continuously poll ALL validators in parallel.

        This is the key competitive advantage:
        - Base miner: Polls one validator, waits for response
        - Competitive: Polls ALL validators simultaneously
        """
        while True:
            try:
                # Get all available validators
                validator_uids = self._get_available_validators(
                    validator_selector,
                    metagraph
                )

                if not validator_uids:
                    logger.debug("No validators available for polling")
                    await asyncio.sleep(self.pull_interval)
                    continue

                # Pull from ALL validators in parallel
                logger.debug(f"Polling {len(validator_uids)} validators...")

                pull_tasks = [
                    self._pull_from_validator(uid, metagraph, wallet)
                    for uid in validator_uids
                ]

                results = await asyncio.gather(*pull_tasks, return_exceptions=True)

                # Process results
                tasks_added = 0
                for uid, result in zip(validator_uids, results):
                    if isinstance(result, Exception):
                        logger.debug(f"Validator {uid} pull failed: {result}")
                        continue

                    if result and result.task:
                        try:
                            # Add to queue (non-blocking)
                            self.task_queue.put_nowait((uid, result.task, result.cooldown_until))
                            tasks_added += 1
                            self.stats["tasks_pulled"] += 1
                        except asyncio.QueueFull:
                            logger.warning("Task queue full, dropping task")

                    # Update cooldown
                    if result and result.cooldown_until:
                        self.validator_cooldowns[uid] = result.cooldown_until

                if tasks_added > 0:
                    logger.info(f"Added {tasks_added} tasks to queue")

            except Exception as e:
                logger.error(f"Validator polling error: {e}")

            await asyncio.sleep(self.pull_interval)

    async def _pull_from_validator(
        self,
        validator_uid: int,
        metagraph: bt.metagraph,
        wallet: bt.wallet
    ) -> Optional[PullTask]:
        """
        Pull task from single validator (async, non-blocking).
        """
        try:
            async with bt.dendrite(wallet=wallet) as dendrite:
                response = await dendrite.call(
                    target_axon=metagraph.axons[validator_uid],
                    synapse=PullTask(),
                    deserialize=False,
                    timeout=12.0
                )

                return response

        except Exception as e:
            logger.debug(f"Failed to pull from validator {validator_uid}: {e}")
            return None

    async def _task_worker(
        self,
        worker_id: int,
        endpoint: str,
        validator_class,
        processor_func,
        metagraph: bt.metagraph,
        wallet: bt.wallet
    ):
        """
        Worker that processes tasks from the queue.

        Multiple workers run in parallel, each with its own generation endpoint.
        """
        logger.info(f"Worker {worker_id} started with endpoint {endpoint}")

        while True:
            try:
                # Get task from queue (blocking)
                validator_uid, task, cooldown_until = await self.task_queue.get()

                logger.info(
                    f"Worker {worker_id} processing task: '{task.prompt}' "
                    f"from validator {validator_uid}"
                )

                # Process task
                await processor_func(
                    validator_uid=validator_uid,
                    task=task,
                    cooldown_until=cooldown_until,
                    endpoint=endpoint,
                    validator_class=validator_class,
                    metagraph=metagraph,
                    wallet=wallet,
                    stats=self.stats
                )

                self.stats["tasks_completed"] += 1

            except Exception as e:
                logger.error(f"Worker {worker_id} task processing error: {e}")
                self.stats["tasks_failed"] += 1

            finally:
                self.task_queue.task_done()

    def _get_available_validators(
        self,
        validator_selector,
        metagraph: bt.metagraph
    ) -> List[int]:
        """
        Get all validators that are:
        - Serving
        - Meet stake requirements
        - Not on cooldown
        - Not blacklisted
        """
        current_time = int(time.time())
        available = []

        for uid in range(metagraph.n):
            # Check basic criteria
            if not metagraph.axons[uid].is_serving:
                continue

            if metagraph.S[uid] < validator_selector._min_stake:
                continue

            # Check cooldown
            if self.validator_cooldowns.get(uid, 0) > current_time:
                continue

            # Check blacklist (will add this next)
            if hasattr(validator_selector, 'is_blacklisted'):
                if validator_selector.is_blacklisted(uid):
                    continue

            available.append(uid)

        return available

    async def _stats_logger(self):
        """Log statistics periodically"""
        while True:
            await asyncio.sleep(60)  # Every minute

            logger.info(
                f"AsyncTaskManager stats: "
                f"Pulled={self.stats['tasks_pulled']}, "
                f"Completed={self.stats['tasks_completed']}, "
                f"Failed={self.stats['tasks_failed']}, "
                f"Validated={self.stats['tasks_validated']}, "
                f"Rejected={self.stats['tasks_rejected']}, "
                f"Queue={self.task_queue.qsize()}/{self.max_queue_size}"
            )

    def get_stats(self) -> Dict:
        """Get current statistics"""
        return {
            **self.stats,
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks)
        }


# Example usage:
#
# task_manager = AsyncTaskManager(max_concurrent_tasks=4)
#
# await task_manager.start(
#     validator_selector=self.validator_selector,
#     metagraph=self.metagraph,
#     wallet=self.wallet,
#     generation_endpoints=self.config.generation.endpoints,
#     validator_class=clip_validator,
#     processor_func=process_task_async
# )
