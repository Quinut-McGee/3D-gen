#!/bin/bash

###############################################################################
# 404-GEN COMPETITIVE MINER - ONE-CLICK DEPLOYMENT SCRIPT
###############################################################################
#
# This script deploys the full competitive mining system:
# - FLUX.1-schnell text-to-image
# - BRIA RMBG 2.0 background removal
# - DreamGaussian 3D generation
# - CLIP validation
# - Async multi-validator polling
#
# Usage:
#   ./deploy_competitive.sh [--port 10006] [--skip-deps]
#
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
PORT=10006
FLUX_STEPS=4
VALIDATION_THRESHOLD=0.6
SKIP_DEPS=false
CONFIG="configs/text_mv_fast.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --flux-steps)
            FLUX_STEPS="$2"
            shift 2
            ;;
        --validation-threshold)
            VALIDATION_THRESHOLD="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--port PORT] [--skip-deps] [--flux-steps STEPS] [--validation-threshold THRESHOLD] [--config CONFIG]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}404-GEN COMPETITIVE MINER DEPLOYMENT${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Port: ${PORT}"
echo -e "  FLUX steps: ${FLUX_STEPS}"
echo -e "  Validation threshold: ${VALIDATION_THRESHOLD}"
echo -e "  DreamGaussian config: ${CONFIG}"
echo -e "  Skip dependencies: ${SKIP_DEPS}"
echo ""

###############################################################################
# Step 1: Check Dependencies
###############################################################################

if [ "$SKIP_DEPS" = false ]; then
    echo -e "${YELLOW}[1/6] Checking dependencies...${NC}"

    # Check if pip is available
    if ! command -v pip &> /dev/null; then
        echo -e "${RED}Error: pip not found. Please install Python and pip first.${NC}"
        exit 1
    fi

    # Check for required packages
    REQUIRED_PACKAGES=(
        "transformers"
        "diffusers"
        "accelerate"
        "xformers"
        "clip-by-openai"
        "Pillow"
        "torch"
        "fastapi"
        "uvicorn"
        "aiohttp"
        "loguru"
    )

    MISSING_PACKAGES=()

    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! python -c "import ${package//-/_}" &> /dev/null; then
            MISSING_PACKAGES+=("$package")
        fi
    done

    if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
        echo -e "${YELLOW}Installing missing packages: ${MISSING_PACKAGES[*]}${NC}"
        pip install "${MISSING_PACKAGES[@]}" --upgrade
        echo -e "${GREEN}✅ Dependencies installed${NC}"
    else
        echo -e "${GREEN}✅ All dependencies already installed${NC}"
    fi
else
    echo -e "${YELLOW}[1/6] Skipping dependency check (--skip-deps)${NC}"
fi

echo ""

###############################################################################
# Step 2: Check GPU
###############################################################################

echo -e "${YELLOW}[2/6] Checking GPU availability...${NC}"

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    VRAM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}')")
    echo -e "${GREEN}✅ GPU detected: ${GPU_NAME} (${VRAM} GB VRAM)${NC}"
else
    echo -e "${RED}⚠️  No GPU detected - will run on CPU (VERY SLOW)${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

###############################################################################
# Step 3: Stop Old Services
###############################################################################

echo -e "${YELLOW}[3/6] Stopping old services...${NC}"

# Find and kill old generation services
OLD_GENERATION_PIDS=$(pgrep -f "serve.py\|serve_competitive.py" || true)
if [ -n "$OLD_GENERATION_PIDS" ]; then
    echo -e "${YELLOW}Stopping old generation services: ${OLD_GENERATION_PIDS}${NC}"
    kill $OLD_GENERATION_PIDS || true
    sleep 2
    echo -e "${GREEN}✅ Old generation services stopped${NC}"
else
    echo -e "${GREEN}✅ No old generation services running${NC}"
fi

# Find and kill old miners
OLD_MINER_PIDS=$(pgrep -f "serve_miner.py\|serve_miner_competitive.py" || true)
if [ -n "$OLD_MINER_PIDS" ]; then
    echo -e "${YELLOW}Stopping old miners: ${OLD_MINER_PIDS}${NC}"
    kill $OLD_MINER_PIDS || true
    sleep 2
    echo -e "${GREEN}✅ Old miners stopped${NC}"
else
    echo -e "${GREEN}✅ No old miners running${NC}"
fi

echo ""

###############################################################################
# Step 4: Start Competitive Generation Service
###############################################################################

echo -e "${YELLOW}[4/6] Starting competitive generation service...${NC}"

# Check if serve_competitive.py exists
if [ ! -f "generation/serve_competitive.py" ]; then
    echo -e "${RED}Error: generation/serve_competitive.py not found${NC}"
    exit 1
fi

# Start generation service in background
nohup python generation/serve_competitive.py \
    --port $PORT \
    --config $CONFIG \
    --flux-steps $FLUX_STEPS \
    --validation-threshold $VALIDATION_THRESHOLD \
    --enable-validation \
    > logs/generation_service.log 2>&1 &

GENERATION_PID=$!
echo $GENERATION_PID > .generation_service.pid

echo -e "${GREEN}✅ Generation service started (PID: ${GENERATION_PID})${NC}"
echo -e "   Logs: logs/generation_service.log"

# Wait for service to be ready
echo -e "${YELLOW}Waiting for generation service to initialize...${NC}"
MAX_WAIT=180  # 3 minutes max (model loading takes time)
WAITED=0

while ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo -e "${RED}Error: Generation service failed to start within ${MAX_WAIT}s${NC}"
        echo -e "${RED}Check logs/generation_service.log for details${NC}"
        exit 1
    fi

    echo -n "."
    sleep 5
    WAITED=$((WAITED + 5))
done

echo ""
echo -e "${GREEN}✅ Generation service ready at http://localhost:${PORT}${NC}"

# Show health check
HEALTH=$(curl -s http://localhost:$PORT/health)
echo -e "${BLUE}Health check:${NC}"
echo "$HEALTH" | python -m json.tool

echo ""

###############################################################################
# Step 5: Start Competitive Miner
###############################################################################

echo -e "${YELLOW}[5/6] Starting competitive miner...${NC}"

# Check if serve_miner_competitive.py exists
if [ ! -f "neurons/serve_miner_competitive.py" ]; then
    echo -e "${RED}Error: neurons/serve_miner_competitive.py not found${NC}"
    exit 1
fi

# Start miner in background
nohup python neurons/serve_miner_competitive.py \
    > logs/miner.log 2>&1 &

MINER_PID=$!
echo $MINER_PID > .miner.pid

echo -e "${GREEN}✅ Miner started (PID: ${MINER_PID})${NC}"
echo -e "   Logs: logs/miner.log"

# Wait a bit for miner to initialize
sleep 5

echo ""

###############################################################################
# Step 6: Verify Everything is Running
###############################################################################

echo -e "${YELLOW}[6/6] Verifying deployment...${NC}"

# Check generation service
if ps -p $GENERATION_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Generation service running (PID: ${GENERATION_PID})${NC}"
else
    echo -e "${RED}❌ Generation service crashed - check logs/generation_service.log${NC}"
    exit 1
fi

# Check miner
if ps -p $MINER_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Miner running (PID: ${MINER_PID})${NC}"
else
    echo -e "${RED}❌ Miner crashed - check logs/miner.log${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🚀 DEPLOYMENT COMPLETE!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Services:${NC}"
echo -e "  Generation Service: http://localhost:${PORT}"
echo -e "  Generation PID: ${GENERATION_PID}"
echo -e "  Miner PID: ${MINER_PID}"
echo ""
echo -e "${GREEN}Logs:${NC}"
echo -e "  Generation: tail -f logs/generation_service.log"
echo -e "  Miner: tail -f logs/miner.log"
echo ""
echo -e "${GREEN}Management:${NC}"
echo -e "  Stop all: ./stop_competitive.sh"
echo -e "  Restart: ./deploy_competitive.sh"
echo -e "  Test: python test_competitive.py"
echo ""
echo -e "${YELLOW}Monitor your miner for ~15 minutes to ensure stability.${NC}"
echo -e "${YELLOW}Check that tasks are being pulled and submitted successfully.${NC}"
echo ""
echo -e "${GREEN}Expected performance:${NC}"
echo -e "  Generation time: 15-25 seconds"
echo -e "  CLIP scores: >0.7"
echo -e "  Throughput: 120+ tasks per 4h"
echo ""
