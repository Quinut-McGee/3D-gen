#!/bin/bash

###############################################################################
# 404-GEN COMPETITIVE MINER - STOP SCRIPT
###############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping competitive miner services...${NC}"
echo ""

# Stop generation service
if [ -f .generation_service.pid ]; then
    GENERATION_PID=$(cat .generation_service.pid)
    if ps -p $GENERATION_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}Stopping generation service (PID: ${GENERATION_PID})...${NC}"
        kill $GENERATION_PID
        sleep 2

        # Force kill if still running
        if ps -p $GENERATION_PID > /dev/null 2>&1; then
            echo -e "${YELLOW}Force killing generation service...${NC}"
            kill -9 $GENERATION_PID
        fi

        echo -e "${GREEN}✅ Generation service stopped${NC}"
    else
        echo -e "${YELLOW}Generation service not running${NC}"
    fi
    rm -f .generation_service.pid
fi

# Stop miner
if [ -f .miner.pid ]; then
    MINER_PID=$(cat .miner.pid)
    if ps -p $MINER_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}Stopping miner (PID: ${MINER_PID})...${NC}"
        kill $MINER_PID
        sleep 2

        # Force kill if still running
        if ps -p $MINER_PID > /dev/null 2>&1; then
            echo -e "${YELLOW}Force killing miner...${NC}"
            kill -9 $MINER_PID
        fi

        echo -e "${GREEN}✅ Miner stopped${NC}"
    else
        echo -e "${YELLOW}Miner not running${NC}"
    fi
    rm -f .miner.pid
fi

# Also kill any lingering processes
echo -e "${YELLOW}Checking for lingering processes...${NC}"

LINGERING_GENERATION=$(pgrep -f "serve_competitive.py" || true)
if [ -n "$LINGERING_GENERATION" ]; then
    echo -e "${YELLOW}Killing lingering generation services: ${LINGERING_GENERATION}${NC}"
    kill -9 $LINGERING_GENERATION || true
fi

LINGERING_MINER=$(pgrep -f "serve_miner_competitive.py" || true)
if [ -n "$LINGERING_MINER" ]; then
    echo -e "${YELLOW}Killing lingering miners: ${LINGERING_MINER}${NC}"
    kill -9 $LINGERING_MINER || true
fi

echo ""
echo -e "${GREEN}✅ All competitive miner services stopped${NC}"
