#!/bin/bash

# Script: Test different GEMM parallel strategy performance
# Strategies: weight-parallel and no-parallel
# Thread counts: 1,2,4,8,12,16

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GEMM_CONFIG="$PROJECT_ROOT/include/gemm-config.h"
GEMM_CONFIG_BACKUP="$PROJECT_ROOT/include/gemm-config.h.bak"
BUILD_DIR="$PROJECT_ROOT/build"
STATS_DIR="$PROJECT_ROOT/stats"
CSV_FILE="$STATS_DIR/test_parallel_strategy_benchmark.csv"
MODEL_PATH="$PROJECT_ROOT/models/BitNet-b1.58-2B-4T/ggml-model-original.gguf"
BENCHMARK_CMD="./build/bin/llama-bench"
THREADS_LIST="1 2 4 8 12 16"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if [ ! -f "$GEMM_CONFIG" ]; then
        log_error "gemm-config.h not found: $GEMM_CONFIG"
        exit 1
    fi
    
    if [ ! -f "$MODEL_PATH" ]; then
        log_error "Model file not found: $MODEL_PATH"
        exit 1
    fi
    
    if [ ! -d "$BUILD_DIR" ]; then
        log_error "Build directory not found: $BUILD_DIR"
        exit 1
    fi
    
    if [ ! -f "$BUILD_DIR/bin/llama-bench" ]; then
        log_warn "llama-bench executable not found, building..."
        build_project
    fi
    
    if [ ! -d "$STATS_DIR" ]; then
        log_info "Creating stats directory..."
        mkdir -p "$STATS_DIR"
    fi
    
    log_info "Prerequisites check completed"
}

# Backup original config file
backup_config() {
    log_info "Backing up gemm-config.h..."
    cp "$GEMM_CONFIG" "$GEMM_CONFIG_BACKUP"
    log_info "Backup completed: $GEMM_CONFIG_BACKUP"
}

# Restore original config file
restore_config() {
    if [ -f "$GEMM_CONFIG_BACKUP" ]; then
        log_info "Restoring original gemm-config.h..."
        cp "$GEMM_CONFIG_BACKUP" "$GEMM_CONFIG"
        rm "$GEMM_CONFIG_BACKUP"
        log_info "Restore completed"
    else
        log_warn "Backup file not found, skipping restore"
    fi
}

# Set activation-parallel configuration (keep original ACT_PARALLEL)
set_activation_parallel() {
    log_info "Configuration: activation-parallel (keeping #define ACT_PARALLEL)"
    log_info "Configuration completed"
}

# Set weight-parallel configuration (remove ACT_PARALLEL)
set_weight_parallel() {
    log_info "Configuration: weight-parallel (removing #define ACT_PARALLEL)"
    
    # Remove ACT_PARALLEL definition
    sed -i '/#define ACT_PARALLEL/d' "$GEMM_CONFIG"
    
    # Verify modification
    if grep -q "^#define ACT_PARALLEL" "$GEMM_CONFIG"; then
        log_error "Failed to remove ACT_PARALLEL"
        exit 1
    fi
    log_info "Configuration completed"
}

# Set no-parallel configuration (remove ACT_PARALLEL + modify SIZE to 1)
set_no_parallel() {
    log_info "Configuration: no-parallel (removing #define ACT_PARALLEL + modifying SIZE to 1)"
    
    # Remove ACT_PARALLEL definition
    sed -i '/#define ACT_PARALLEL/d' "$GEMM_CONFIG"
    
    # Modify all ROW_BLOCK_SIZE and COL_BLOCK_SIZE to 1
    sed -i 's/#define ROW_BLOCK_SIZE [0-9]\+/#define ROW_BLOCK_SIZE 1/g' "$GEMM_CONFIG"
    sed -i 's/#define COL_BLOCK_SIZE [0-9]\+/#define COL_BLOCK_SIZE 1/g' "$GEMM_CONFIG"
    
    log_info "Configuration completed"
}

# Build project
build_project() {
    log_info "Building project..."
    cd "$PROJECT_ROOT"
    
    if [ ! -f "$BUILD_DIR/Makefile" ]; then
        log_info "First build, running cmake..."
        cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
    fi
    
    cd "$BUILD_DIR"
    make -j$(nproc) llama-bench > /dev/null 2>&1
    
    if [ ! -f "./bin/llama-bench" ]; then
        log_error "Build failed"
        exit 1
    fi
    
    log_info "Build completed"
    cd "$PROJECT_ROOT"
}

# Run benchmark test
run_benchmark() {
    local strategy=$1
    local threads=$2
    
    cd "$PROJECT_ROOT"
    
    # Run llama-bench
    local output=$($BENCHMARK_CMD -m "$MODEL_PATH" -p 128 -n 0 -t "$threads" -ngl 0 2>&1)
    
    # Extract line containing "pp128"
    local line=$(echo "$output" | grep "pp128" | tail -1)
    
    if [ -z "$line" ]; then
        return 1
    fi
    
    echo "$line"
}

# Extract throughput value from benchmark output
extract_throughput() {
    local line=$1
    
    # Remove any leading/trailing whitespace and log messages
    # The line format is: | model | size | params | backend | threads | test | throughput |
    # We need to extract the last field which contains the throughput in format "XXX.XX ± YY.YY"
    local throughput=$(echo "$line" | awk -F'|' '{print $NF}' | xargs | sed 's/\[.*\]//' | xargs)
    
    echo "$throughput"
}

# Initialize CSV file
init_csv() {
    log_info "Initializing CSV file: $CSV_FILE"
    
    cat > "$CSV_FILE" << 'EOF'
Strategy,Threads,Throughput
EOF
    
    log_info "CSV file created"
}

# Add result to CSV
add_to_csv() {
    local strategy=$1
    local threads=$2
    local throughput=$3
    
    echo "$strategy,$threads,$throughput" >> "$CSV_FILE"
}

# Main function
main() {
    log_info "Starting GEMM parallel strategy benchmark tests"
    log_info "================================================"
    
    # Check prerequisites
    check_prerequisites
    
    # Backup original configuration
    backup_config
    
    # Initialize CSV file
    init_csv
    
    # Define strategies to test
    local strategies=("activation-parallel" "weight-parallel" "no-parallel")
    
    for strategy in "${strategies[@]}"; do
        log_info "================================================"
        log_info "Testing strategy: $strategy"
        log_info "================================================"
        
        # Restore to original configuration
        restore_config
        backup_config
        
        # Apply configuration based on strategy
        case $strategy in
            activation-parallel)
                set_activation_parallel
                ;;
            weight-parallel)
                set_weight_parallel
                ;;
            no-parallel)
                set_no_parallel
                ;;
        esac
        
        # Rebuild project to apply new configuration
        log_info "Rebuilding project to apply new configuration..."
        build_project
        
        # Run test for each thread count
        for threads in $THREADS_LIST; do
            log_info ""
            log_info "Strategy: $strategy, Threads: $threads"
            
            # Run test (capture only output, not log messages)
            local result=$(run_benchmark "$strategy" "$threads")
            local test_status=$?
            
            if [ $test_status -eq 0 ]; then
                # Extract throughput value from the result line
                local throughput=$(extract_throughput "$result")
                log_info "Throughput: $throughput"
                
                # Add to CSV
                add_to_csv "$strategy" "$threads" "$throughput"
            else
                log_warn "Test failed for strategy $strategy, threads $threads"
            fi
            
            sleep 2  # Give system time to cool down
        done
    done
    
    # Restore original configuration
    restore_config
    
    log_info "================================================"
    log_info "Test completed!"
    log_info "Results saved to: $CSV_FILE"
    log_info "================================================"
    
    # Display CSV content
    log_info "CSV file content:"
    cat "$CSV_FILE"
}

# Run main function
main "$@"
