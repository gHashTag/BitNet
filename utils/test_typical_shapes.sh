#!/bin/bash
# Test typical matrix shapes for BitNet-2B model
# Based on BitNet-b1.58-2B-4T architecture

echo "=========================================="
echo "BitNet-2B Typical Shapes Performance Test"
echo "=========================================="
echo ""

ITERATIONS=1000
BENCHMARK="../build/test_gemm_kernel"

# Create stats directory if not exists
mkdir -p ../stats

# Generate output CSV filename
CSV_FILE="../stats/gemm_kernel_test_noparal.csv"

# Write CSV header
echo "test_name,n,nr,nc,time_ms,gflops,throughput_tokens_per_sec" > "$CSV_FILE"
echo "Results will be saved to: $CSV_FILE"
echo ""

# Function to extract metrics and append to CSV
extract_and_save() {
    local test_name="$1"
    local output="$2"
    
    # Extract values using grep and awk
    local n=$(echo "$output" | grep "Embedding dimension" | awk '{print $5}')
    local nr=$(echo "$output" | grep "Matrix Y rows" | awk '{print $6}')
    local nc=$(echo "$output" | grep "Matrix X columns" | awk '{print $6}')
    local avg_time=$(echo "$output" | grep "Average time" | awk '{print $4}')
    local min_time=$(echo "$output" | grep "Min time" | awk '{print $4}')
    local max_time=$(echo "$output" | grep "Max time" | awk '{print $4}')
    local gflops=$(echo "$output" | grep "GFLOPS" | awk '{print $3}')
    local throughput=$(echo "$output" | grep "Throughput" | awk '{print $3}')
    
    # Calculate standard deviation estimate from range (assuming ~95% of data within min-max)
    # For normal distribution, range ≈ 4*std, so std ≈ range/4
    local std_time=$(echo "scale=4; ($max_time - $min_time) / 4" | bc)
    
    # Format as mean±std
    local time_formatted="${avg_time}±${std_time}"
    
    # For GFLOPS and throughput, we don't have std info, so just use the value
    # If you want to estimate std for these as well, you would need more data
    
    # Append to CSV
    echo "${test_name},${n},${nr},${nc},${time_formatted},${gflops},${throughput}" >> "$CSV_FILE"
}

echo "Test 1: Single Token Generation (Attention QKV projection)"
echo "  Scenario: Generating 1 token at a time"
echo "  Shape: n=2048, r=1, c=2048"
OUTPUT=$($BENCHMARK -n 2048 -r 1 -c 2048 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "single_token_gen" "$OUTPUT"
echo ""

echo "Test 2: Small Batch Prompt Processing (Attention QKV projection)"
echo "  Scenario: Processing prompt with 128 tokens, batch size 1"
echo "  Shape: n=2048, r=128, c=2048"
OUTPUT=$($BENCHMARK -n 2048 -r 128 -c 2048 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "small_batch_prompt" "$OUTPUT"
echo ""

echo "Test 3: Medium Batch Prompt Processing (Attention QKV projection)"
echo "  Scenario: Processing prompt with 256 tokens or batch of 256"
echo "  Shape: n=2048, r=256, c=2048"
OUTPUT=$($BENCHMARK -n 2048 -r 256 -c 2048 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "medium_batch_prompt" "$OUTPUT"
echo ""

echo "Test 4: Large Batch Processing (Attention QKV projection)"
echo "  Scenario: Processing 512 tokens or batch of 512"
echo "  Shape: n=2048, r=512, c=2048"
OUTPUT=$($BENCHMARK -n 2048 -r 512 -c 2048 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "large_batch_prompt" "$OUTPUT"
echo ""

echo "Test 5: FFN Up-projection (Small batch)"
echo "  Scenario: Feed-forward network expansion, 128 tokens"
echo "  Shape: n=2048, r=128, c=8192"
OUTPUT=$($BENCHMARK -n 2048 -r 128 -c 8192 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "ffn_up_projection" "$OUTPUT"
echo ""

echo "Test 6: FFN Down-projection (Small batch)"
echo "  Scenario: Feed-forward network reduction, 128 tokens"
echo "  Shape: n=8192, r=128, c=2048"
OUTPUT=$($BENCHMARK -n 8192 -r 128 -c 2048 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "ffn_down_projection" "$OUTPUT"
echo ""

echo "Test 7: Long Context Processing"
echo "  Scenario: Processing very long context (2048 tokens)"
echo "  Shape: n=2048, r=2048, c=2048"
OUTPUT=$($BENCHMARK -n 2048 -r 2048 -c 2048 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "long_context" "$OUTPUT"
echo ""

echo "Test 8: Batched Token Generation"
echo "  Scenario: Generating tokens for 32 sequences simultaneously"
echo "  Shape: n=2048, r=32, c=2048"
OUTPUT=$($BENCHMARK -n 2048 -r 32 -c 2048 -i $ITERATIONS 2>&1)
echo "$OUTPUT"
extract_and_save "batched_token_gen" "$OUTPUT"
echo ""

echo "=========================================="
echo "All tests completed!"
echo "Results saved to: $CSV_FILE"
echo "=========================================="
