#!/bin/bash
# Quick deployment script for fast config
# Run this on your virtual server with RTX 4090

echo "================================================"
echo "404-GEN Fast Config Deployment"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Step 1: Checking current setup...${NC}"
pm2 list

echo -e "\n${YELLOW}Step 2: Stopping current generation service...${NC}"
pm2 stop generation

echo -e "\n${YELLOW}Step 3: Deploying FAST config (target: <30s)...${NC}"
cd generation
pm2 delete generation
pm2 start serve.py --name generation -- --config configs/text_mv_fast.yaml

echo -e "\n${GREEN}Waiting 10 seconds for service to initialize...${NC}"
sleep 10

echo -e "\n${YELLOW}Step 4: Testing generation speed...${NC}"
echo "Testing with prompt: 'a red sports car'"

START_TIME=$(date +%s)
curl -X POST http://localhost:10006/generate/ \
  -F "prompt=a red sports car" \
  --output /tmp/speed_test.ply \
  2>&1 | grep -E "(200 OK|error)"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo -e "\n${GREEN}Generation completed in ${ELAPSED} seconds${NC}"

if [ -f /tmp/speed_test.ply ]; then
    FILE_SIZE=$(stat -f%z /tmp/speed_test.ply 2>/dev/null || stat -c%s /tmp/speed_test.ply 2>/dev/null)
    FILE_SIZE_KB=$((FILE_SIZE / 1024))
    echo -e "${GREEN}File size: ${FILE_SIZE_KB} KB${NC}"

    if [ $ELAPSED -lt 30 ]; then
        echo -e "\n${GREEN}✅ SUCCESS! Generation time <30s - ready for production!${NC}"
    elif [ $ELAPSED -lt 60 ]; then
        echo -e "\n${YELLOW}⚠️  Time is acceptable but could be better.${NC}"
        echo "Consider using ultra_fast config for <15s"
    else
        echo -e "\n${RED}❌ Still too slow. Check GPU utilization and config.${NC}"
    fi

    if [ $FILE_SIZE -gt 10000 ]; then
        echo -e "${GREEN}✅ File size acceptable (>10KB)${NC}"
    else
        echo -e "${YELLOW}⚠️  File size seems small. May indicate quality issues.${NC}"
    fi
else
    echo -e "${RED}❌ Generation failed - no output file created${NC}"
fi

echo -e "\n${YELLOW}Step 5: Checking GPU utilization...${NC}"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv

echo -e "\n${YELLOW}Step 6: Viewing recent generation logs...${NC}"
pm2 logs generation --lines 20 --nostream

echo -e "\n================================================"
echo -e "${GREEN}Deployment complete!${NC}"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Monitor with: pm2 logs generation"
echo "2. Watch miner logs: pm2 logs miner"
echo "3. Check for 'Generation took: X min' - should be <0.5 min"
echo ""
echo "If generation is still too slow:"
echo "  pm2 restart generation --update-env -- --config configs/text_mv_ultra_fast.yaml"
echo ""
echo "To test thoroughly:"
echo "  python test_generation_speed.py"
echo ""
