"""
404-GEN COMPETITIVE MINER - Main Entry Point

This is the production-ready async miner that uses:
- Async multi-validator polling
- CLIP validation
- Parallel task processing
- Validator blacklisting
"""

import asyncio

from miner.config import read_config
from miner.competitive_miner import CompetitiveMiner


async def main() -> None:
    """Start the competitive miner"""
    config = read_config()

    miner = CompetitiveMiner(config)
    await miner.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Miner stopped by user")
    except Exception as e:
        print(f"\n❌ Miner crashed: {e}")
        raise
