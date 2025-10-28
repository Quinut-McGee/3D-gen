"""
Competitive worker implementation with CLIP validation.

This replaces the base workers.py with competitive features:
- CLIP pre-validation
- Async submission
- Better error handling
"""

import asyncio
import base64
import time
import aiohttp
import bittensor as bt
import pyspz
from aiohttp import ClientTimeout
from aiohttp.helpers import sentinel
from common.miner_license_consent_declaration import MINER_LICENSE_CONSENT_DECLARATION
from common.protocol import ProtocolTask, SubmitResults


async def process_task_competitive(
    validator_uid: int,
    task: ProtocolTask,
    cooldown_until: float,
    endpoint: str,
    validator_class,  # CLIP validator instance
    metagraph: bt.metagraph,
    wallet: bt.wallet,
    stats: dict
):
    """
    Process a single task with competitive features.

    This is called by AsyncTaskManager workers.
    """
    bt.logging.debug(f"Processing task '{task.prompt}' from validator {validator_uid}")

    # Generate
    results = await _generate(endpoint, task.prompt)

    if not results or len(results) < 1000:
        bt.logging.warning(
            f"Generation failed or too small ({len(results) if results else 0} bytes), "
            "submitting empty to avoid penalty"
        )
        results = b""
        stats["tasks_rejected"] += 1
    else:
        # CLIP validation (optional but recommended)
        if validator_class and results:
            # For now, skip PLY rendering validation (too slow)
            # Just check file size as proxy for quality
            # TODO: Could add quick render + CLIP check here
            bt.logging.debug(f"Generation size OK: {len(results)} bytes")
            stats["tasks_validated"] += 1

    # Submit (async, non-blocking to allow worker to continue)
    # Fire-and-forget allows worker to start next generation immediately
    asyncio.create_task(
        _submit_results_async(
            wallet=wallet,
            metagraph=metagraph,
            validator_uid=validator_uid,
            task=task,
            results=results,
            prompt=task.prompt  # For better logging
        )
    )

    bt.logging.info(f"Task queued for submission to validator {validator_uid}")


async def _generate(generate_url: str, prompt: str, timeout: float = 60.0) -> bytes | None:
    """
    Generate 3D model from prompt.

    Timeout is generous (60s) to allow for full pipeline.
    """
    bt.logging.debug(f"Generating: '{prompt}'")

    client_timeout = ClientTimeout(total=timeout)

    try:
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(generate_url, data={"prompt": prompt}) as response:
                if response.status == 200:
                    results = await response.read()

                    # Check validation headers
                    validation_failed = response.headers.get("X-Validation-Failed") == "true"
                    clip_score = response.headers.get("X-CLIP-Score", "N/A")

                    if validation_failed:
                        bt.logging.warning(
                            f"Generation service validation failed (CLIP={clip_score})"
                        )
                        return b""  # Empty result

                    bt.logging.info(
                        f"Generation completed: {len(results)} bytes, CLIP={clip_score}"
                    )
                    return results

                else:
                    bt.logging.error(f"Generation failed with status {response.status}")
                    return None

    except asyncio.TimeoutError:
        bt.logging.error(f"Generation timeout after {timeout}s")
        return None
    except aiohttp.ClientError as e:
        bt.logging.error(f"Generation client error: {e}")
        return None
    except Exception as e:
        bt.logging.error(f"Generation unexpected error: {e}")
        return None


async def _submit_results_async(
    wallet: bt.wallet,
    metagraph: bt.metagraph,
    validator_uid: int,
    task: ProtocolTask,
    results: bytes,
    prompt: str = ""
):
    """
    Submit results asynchronously (fire-and-forget).

    This doesn't block the worker from processing next task.
    """
    submission_start = time.time()
    bt.logging.debug(f"Starting submission of '{prompt[:50]}...' to validator {validator_uid}")
    try:
        async with bt.dendrite(wallet=wallet) as dendrite:
            submit_time = time.time_ns()

            # Sign message
            t1 = time.time()
            message = (
                f"{MINER_LICENSE_CONSENT_DECLARATION}"
                f"{submit_time}{task.prompt}{metagraph.hotkeys[validator_uid]}{wallet.hotkey.ss58_address}"
            )
            signature = base64.b64encode(dendrite.keypair.sign(message)).decode(encoding="utf-8")
            signing_time = time.time() - t1

            # Compress results
            t2 = time.time()
            if results:
                compressed_results = base64.b64encode(pyspz.compress(results, workers=-1)).decode(encoding="utf-8")
            else:
                compressed_results = ""
            compression_time = time.time() - t2

            # Submit
            t3 = time.time()
            synapse = SubmitResults(
                task_id=task.id,
                results=compressed_results,
                submit_time=submit_time,
                signature=signature
            )

            response = await dendrite.call(
                target_axon=metagraph.axons[validator_uid],
                synapse=synapse,
                deserialize=False,
                timeout=60.0  # Reduced from 300s - validators should respond quickly
            )
            network_time = time.time() - t3

            total_time = time.time() - submission_start

            # Log feedback with timing breakdown
            bt.logging.info(
                f"Submission to [{validator_uid}] timing: "
                f"sign={signing_time:.2f}s, compress={compression_time:.2f}s, "
                f"network={network_time:.2f}s, total={total_time:.2f}s"
            )

            if response.feedback:
                feedback = response.feedback
                score = "failed" if feedback.validation_failed else feedback.task_fidelity_score

                # Extract observation count if available
                obs_count = getattr(feedback, 'generations_within_the_window', 'N/A')

                bt.logging.info(
                    f"Feedback from [{validator_uid}]: Score={score}, "
                    f"AvgScore={feedback.average_fidelity_score:.3f}, "
                    f"ELO={feedback.current_duel_rating}, "
                    f"Reward={feedback.current_miner_reward:.4f}, "
                    f"Observations={obs_count}"
                )
            else:
                bt.logging.warning(f"No feedback from validator {validator_uid}")

    except asyncio.TimeoutError:
        total_time = time.time() - submission_start
        bt.logging.error(f"Submission to validator {validator_uid} TIMED OUT after {total_time:.1f}s")
    except Exception as e:
        total_time = time.time() - submission_start
        bt.logging.error(f"Submission failed after {total_time:.1f}s: {e}")
