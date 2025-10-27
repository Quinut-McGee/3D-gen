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

        # Per-validator polling cooldown (prevents re-polling too quickly)
        # Maps: {validator_uid: last_poll_timestamp}
        self.last_validator_poll: Dict[int, float] = {}

        # Per-validator active task tracking (prevents duplicates)
        # Maps: {validator_uid: {task_id1, task_id2, ...}}
        self.active_validator_tasks: Dict[int, Set[str]] = {}

        # Global task deduplication (prevents cross-validator duplicates)
        # Maps: {task_id: submission_timestamp}
        self.recently_submitted_tasks: Dict[str, float] = {}

        # Task cleanup tracking (keeps tasks in active tracking after completion)
        # Maps: {task_id: cleanup_timestamp}
        self.task_cleanup_times: Dict[str, float] = {}

        # Statistics
        self.stats = {
            "tasks_pulled": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_validated": 0,
            "tasks_rejected": 0,
            "duplicates_prevented": 0,
            "validators_skipped_cooldown": 0,  # Track polling cooldown effectiveness
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
            asyncio.create_task(self._stats_logger()),

            # Cleanup old task tracking entries
            asyncio.create_task(self._cleanup_old_tasks())
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
                current_time = time.time()

                for uid, result in zip(validator_uids, results):
                    if isinstance(result, Exception):
                        logger.debug(f"Validator {uid} pull failed: {result}")
                        continue

                    # Only update polling timestamp if validator responded (even with no task)
                    # This prevents re-polling validators that are online but have no tasks
                    if result is not None:
                        self.last_validator_poll[uid] = current_time

                    if result and result.task:
                        task_id = result.task.id

                        # GLOBAL deduplication check (prevents same task from ANY validator)
                        if task_id in self.recently_submitted_tasks:
                            time_since_submission = current_time - self.recently_submitted_tasks[task_id]
                            if time_since_submission < 300.0:  # 5 minutes
                                logger.warning(
                                    f"🚫 GLOBAL DUPLICATE DETECTED: Task {task_id} was submitted {time_since_submission:.1f}s ago. "
                                    f"Skipping to prevent re-submission."
                                )
                                self.stats["duplicates_prevented"] += 1
                                continue

                        # Check if this task from this validator is already being processed
                        if uid in self.active_validator_tasks and task_id in self.active_validator_tasks[uid]:
                            logger.warning(
                                f"🚫 DUPLICATE DETECTED: Task {task_id} from validator {uid} "
                                f"already in progress (active: {len(self.active_validator_tasks[uid])} tasks). "
                                f"This should be rare due to polling cooldown."
                            )
                            self.stats["duplicates_prevented"] += 1
                            continue

                        try:
                            # Add to per-validator tracking
                            if uid not in self.active_validator_tasks:
                                self.active_validator_tasks[uid] = set()
                            self.active_validator_tasks[uid].add(task_id)

                            # Add to queue (non-blocking)
                            self.task_queue.put_nowait((uid, result.task, result.cooldown_until))
                            tasks_added += 1
                            self.stats["tasks_pulled"] += 1

                            logger.debug(
                                f"Added task {task_id} from validator {uid} to queue. "
                                f"Active tasks from this validator: {len(self.active_validator_tasks[uid])}"
                            )
                        except asyncio.QueueFull:
                            logger.warning("Task queue full, dropping task")
                            # Remove from tracking since we didn't queue it
                            self.active_validator_tasks[uid].discard(task_id)

                    # Update cooldown
                    if result and result.cooldown_until:
                        self.validator_cooldowns[uid] = result.cooldown_until

                if tasks_added > 0:
                    logger.info(
                        f"✅ Added {tasks_added} tasks to queue from {len(validator_uids)} validators polled. "
                        f"Queue size: {self.task_queue.qsize()}/{self.max_queue_size}"
                    )

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
            validator_uid = None
            task_id = None
            got_task = False

            try:
                # Get task from queue (blocking)
                validator_uid, task, cooldown_until = await self.task_queue.get()
                task_id = task.id
                got_task = True

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

                # Mark task as recently submitted (global deduplication)
                if task_id is not None:
                    self.recently_submitted_tasks[task_id] = time.time()
                    logger.debug(f"Task {task_id} marked as recently submitted for global deduplication")

            except asyncio.CancelledError:
                # Worker is being shut down
                logger.debug(f"Worker {worker_id} cancelled during shutdown")
                break

            except Exception as e:
                logger.error(f"Worker {worker_id} task processing error: {e}")
                self.stats["tasks_failed"] += 1

            finally:
                # *** CRITICAL FIX: Delayed task cleanup ***
                # Don't remove immediately! Keep task in active tracking for 60s after completion.
                # This prevents validator from re-assigning same task before it processes our submission.
                if validator_uid is not None and task_id is not None:
                    # Schedule cleanup for 60 seconds in the future
                    cleanup_time = time.time() + 60.0
                    self.task_cleanup_times[task_id] = cleanup_time
                    logger.debug(
                        f"Worker {worker_id} scheduled cleanup of task {task_id} from validator {validator_uid} in 60s. "
                        f"Active tasks from this validator: {len(self.active_validator_tasks.get(validator_uid, set()))}"
                    )

                # Only call task_done if we actually got a task from the queue
                if got_task:
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
        - Not on cooldown (task cooldown from validator)
        - Not recently polled (polling cooldown - prevents duplicate pulls)
        - Not blacklisted
        """
        current_time = time.time()
        available = []

        for uid in range(metagraph.n):
            # Check basic criteria
            if not metagraph.axons[uid].is_serving:
                continue

            if metagraph.S[uid] < validator_selector._min_stake:
                continue

            # Check task assignment cooldown (from validator response)
            if self.validator_cooldowns.get(uid, 0) > current_time:
                continue

            # *** CRITICAL FIX: Per-validator polling cooldown ***
            # Prevents re-polling same validator before workers finish generation (~25s)
            # This fixes the primary bug: same validator giving same task to all 3 workers
            # 35s = generation time (20-25s) + submission time (5s) + validator processing (5-10s)
            if uid in self.last_validator_poll:
                time_since_last_poll = current_time - self.last_validator_poll[uid]
                if time_since_last_poll < 35.0:  # 35 second cooldown (increased from 20s)
                    self.stats["validators_skipped_cooldown"] += 1
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
                f"📊 AsyncTaskManager stats: "
                f"Pulled={self.stats['tasks_pulled']}, "
                f"Completed={self.stats['tasks_completed']}, "
                f"Failed={self.stats['tasks_failed']}, "
                f"Validated={self.stats['tasks_validated']}, "
                f"Rejected={self.stats['tasks_rejected']}, "
                f"DupesPrevented={self.stats['duplicates_prevented']}, "
                f"PollCooldowns={self.stats['validators_skipped_cooldown']}, "
                f"Queue={self.task_queue.qsize()}/{self.max_queue_size}"
            )

    async def _cleanup_old_tasks(self):
        """
        Periodically clean up old task tracking entries.

        Removes:
        - Tasks from active_validator_tasks that are past their cleanup time
        - Old entries from recently_submitted_tasks (older than 5 minutes)
        """
        while True:
            await asyncio.sleep(30)  # Run every 30 seconds

            try:
                current_time = time.time()
                cleaned_tasks = 0
                cleaned_submissions = 0

                # Clean up tasks that have passed their scheduled cleanup time
                tasks_to_cleanup = [
                    (task_id, cleanup_time)
                    for task_id, cleanup_time in self.task_cleanup_times.items()
                    if cleanup_time <= current_time
                ]

                for task_id, _ in tasks_to_cleanup:
                    # Remove from per-validator tracking
                    for validator_uid in list(self.active_validator_tasks.keys()):
                        if task_id in self.active_validator_tasks[validator_uid]:
                            self.active_validator_tasks[validator_uid].discard(task_id)
                            cleaned_tasks += 1

                    # Remove from cleanup schedule
                    del self.task_cleanup_times[task_id]

                # Clean up old submissions (older than 5 minutes)
                old_submissions = [
                    task_id
                    for task_id, submit_time in self.recently_submitted_tasks.items()
                    if current_time - submit_time > 300.0  # 5 minutes
                ]

                for task_id in old_submissions:
                    del self.recently_submitted_tasks[task_id]
                    cleaned_submissions += 1

                if cleaned_tasks > 0 or cleaned_submissions > 0:
                    logger.debug(
                        f"🧹 Cleanup: Removed {cleaned_tasks} tasks from active tracking, "
                        f"{cleaned_submissions} old submissions from deduplication cache"
                    )

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

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
