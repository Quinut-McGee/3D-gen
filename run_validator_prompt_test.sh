#!/bin/bash
# Validator Prompt Validation Test (20 samples)
# Uses REAL validator prompts to validate production readiness

set -e

echo "=== VALIDATOR PROMPT VALIDATION TEST ==="
echo "Sampling 20 real prompts from validator pool (100K prompts)"
echo ""

# Output files
LOG_FILE="/tmp/validator_prompt_test.log"
RESULT_FILE="/tmp/validator_prompt_test_results.json"

# Clear previous logs
> "$LOG_FILE"

# Sample 20 diverse prompts (stratified by word count for diversity)
PROMPTS_FILE="/home/kobe/404-gen/v1/3D-gen/resources/text_prompts.txt"
TEMP_PROMPTS="/tmp/validator_test_prompts.txt"

# Get diverse prompts by word count
{
    # 5 simple (1-6 words)
    awk '{if (NF >= 1 && NF <= 6) print}' "$PROMPTS_FILE" | shuf -n 5
    # 5 moderate (7-12 words)
    awk '{if (NF >= 7 && NF <= 12) print}' "$PROMPTS_FILE" | shuf -n 5
    # 5 complex (13-18 words)
    awk '{if (NF >= 13 && NF <= 18) print}' "$PROMPTS_FILE" | shuf -n 5
    # 5 very complex (19+ words)
    awk '{if (NF >= 19) print}' "$PROMPTS_FILE" | shuf -n 5
} > "$TEMP_PROMPTS"

echo "📝 Selected 20 diverse prompts by complexity"
echo ""

# Test each prompt
SAMPLE_NUM=0
while IFS= read -r PROMPT; do
    SAMPLE_NUM=$((SAMPLE_NUM + 1))
    WORD_COUNT=$(echo "$PROMPT" | wc -w)

    echo "[$SAMPLE_NUM/20] Testing: '$PROMPT' ($WORD_COUNT words)" | tee -a "$LOG_FILE"

    # Call generation service (returns binary PLY with metadata in headers)
    TEMP_PLY="/tmp/validator_test_${SAMPLE_NUM}.ply"
    HTTP_CODE=$(curl -s -w "%{http_code}" -X POST http://localhost:10010/generate/ \
        -F "prompt=$PROMPT" \
        -D /tmp/validator_headers_${SAMPLE_NUM}.txt \
        -o "$TEMP_PLY" \
        2>&1)

    # Parse headers and check status
    if [ "$HTTP_CODE" == "200" ]; then
        # Extract metadata from headers
        NUM_GAUSSIANS=$(grep -i "X-Gaussian-Count:" /tmp/validator_headers_${SAMPLE_NUM}.txt | awk -F': ' '{print $2}' | tr -d '\r')
        FILE_SIZE=$(grep -i "X-File-Size-MB:" /tmp/validator_headers_${SAMPLE_NUM}.txt | awk -F': ' '{print $2}' | tr -d '\r')
        GEN_TIME=$(grep -i "X-Generation-Time:" /tmp/validator_headers_${SAMPLE_NUM}.txt | awk -F': ' '{print $2}' | tr -d '\r')

        echo "  ✅ Success: ${NUM_GAUSSIANS} gaussians, ${FILE_SIZE}MB, ${GEN_TIME}s" | tee -a "$LOG_FILE"

        # Log full headers for analysis
        cat /tmp/validator_headers_${SAMPLE_NUM}.txt >> "$LOG_FILE"
        rm -f "$TEMP_PLY" /tmp/validator_headers_${SAMPLE_NUM}.txt
    else
        # Parse error from header
        ERROR=$(grep -i "X-Generation-Error:" /tmp/validator_headers_${SAMPLE_NUM}.txt | awk -F': ' '{print $2}' | tr -d '\r' || echo "HTTP $HTTP_CODE error")
        echo "  ❌ Failed: $ERROR" | tee -a "$LOG_FILE"
        rm -f "$TEMP_PLY" /tmp/validator_headers_${SAMPLE_NUM}.txt
    fi

    echo "" | tee -a "$LOG_FILE"

done < "$TEMP_PROMPTS"

echo "=== TEST COMPLETE ==="
echo "Results saved to: $LOG_FILE"
echo ""
echo "Running analysis..."

# Analyze results
python3 <<'ANALYSIS_SCRIPT'
import re
import json
from pathlib import Path

log_file = Path("/tmp/validator_prompt_test.log")
log_content = log_file.read_text()

# Parse generations
pattern = r'\[(\d+)/20\] Testing: \'(.+?)\' \((\d+) words\).*?(?:✅ Success: ([\d,]+) gaussians, ([\d.]+)MB, ([\d.]+)s|❌ Failed: (.+?)$)'
matches = re.findall(pattern, log_content, re.DOTALL | re.MULTILINE)

results = {
    "total_samples": 20,
    "successful": 0,
    "failed": 0,
    "success_rate": 0.0,
    "avg_gaussians": 0,
    "avg_file_size_mb": 0.0,
    "avg_gen_time": 0.0,
    "high_density_rate": 0.0,
    "low_density_rate": 0.0,
    "by_complexity": {},
    "samples": []
}

gaussians_list = []
file_sizes = []
gen_times = []
high_density_count = 0
low_density_count = 0

for match in matches:
    sample_num, prompt, word_count, gaussians, file_size, gen_time, error = match
    word_count = int(word_count)

    if gaussians:  # Success
        gaussians = int(gaussians.replace(',', ''))  # Remove commas from number
        file_size = float(file_size)
        gen_time = float(gen_time)

        results["successful"] += 1
        gaussians_list.append(gaussians)
        file_sizes.append(file_size)
        gen_times.append(gen_time)

        if gaussians > 400000:
            high_density_count += 1
        elif gaussians < 150000:
            low_density_count += 1

        results["samples"].append({
            "prompt": prompt,
            "word_count": word_count,
            "gaussians": gaussians,
            "file_size_mb": file_size,
            "gen_time": gen_time,
            "success": True
        })
    else:  # Failed
        results["failed"] += 1
        results["samples"].append({
            "prompt": prompt,
            "word_count": word_count,
            "error": error.strip(),
            "success": False
        })

# Calculate averages
if results["successful"] > 0:
    results["success_rate"] = results["successful"] / results["total_samples"]
    results["avg_gaussians"] = int(sum(gaussians_list) / len(gaussians_list))
    results["avg_file_size_mb"] = round(sum(file_sizes) / len(file_sizes), 2)
    results["avg_gen_time"] = round(sum(gen_times) / len(gen_times), 2)
    results["high_density_rate"] = high_density_count / results["successful"]
    results["low_density_rate"] = low_density_count / results["successful"]

# Save JSON
output_file = Path("/tmp/validator_prompt_test_results.json")
output_file.write_text(json.dumps(results, indent=2))

# Print summary
print("\n" + "="*70)
print("VALIDATOR PROMPT TEST SUMMARY")
print("="*70)
print(f"Total Samples: {results['total_samples']}")
print(f"Successful: {results['successful']}")
print(f"Failed: {results['failed']}")
print(f"Success Rate: {results['success_rate']:.1%}")
print()
print(f"Average Gaussians: {results['avg_gaussians']:,}")
print(f"Average File Size: {results['avg_file_size_mb']:.1f} MB")
print(f"Average Gen Time: {results['avg_gen_time']:.1f}s")
print()
print(f"High Density (>400K): {results['high_density_rate']:.1%}")
print(f"Low Density (<150K): {results['low_density_rate']:.1%}")
print("="*70)

# Compare with comprehensive test
print("\n📊 COMPARISON WITH COMPREHENSIVE TEST:")
print("  Comprehensive Test (120 samples):")
print("    - Success: 100% (0% corruption)")
print("    - Avg Gaussians: 466,323")
print("    - Avg Gen Time: 12.77s")
print("    - High Density: 54.3%")
print()
print(f"  Validator Prompt Test (20 samples):")
print(f"    - Success: {results['success_rate']:.1%}")
print(f"    - Avg Gaussians: {results['avg_gaussians']:,}")
print(f"    - Avg Gen Time: {results['avg_gen_time']:.1f}s")
print(f"    - High Density: {results['high_density_rate']:.1%}")
print()

# Determine readiness
if results['success_rate'] >= 0.95 and results['avg_gaussians'] >= 400000:
    print("✅ MAINNET READY: Validator prompts match comprehensive test quality!")
elif results['success_rate'] >= 0.90:
    print("⚠️  MOSTLY READY: Minor issues detected, review failures")
else:
    print("❌ NOT READY: Significant issues with validator prompts")

print("\n📄 Full results: /tmp/validator_prompt_test_results.json")
ANALYSIS_SCRIPT

echo ""
echo "✅ Validator prompt test complete!"
