/**
 * Standalone benchmark for ggml_gemm_i2_i8_s kernel
 * 
 * This program tests the performance of the ggml_gemm_i2_i8_s kernel
 * with configurable matrix sizes and iteration counts.
 * 
 * Usage: ./test_gemm_kernel [options]
 *   -n <size>   : embedding dimension (must be divisible by 4, default: 2048)
 *   -r <rows>   : number of rows in matrix Y (default: 32)
 *   -c <cols>   : number of columns in matrix X (default: 128)
 *   -i <iters>  : number of iterations (default: 1000)
 *   -w <warmup> : number of warmup iterations (default: 10)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

// Include necessary headers
#include "../include/gemm-config.h"

// Function declarations (from ggml-quants.h)
extern "C" void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

// GEMM kernel definition
void ggml_gemm_i2_i8_s(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
#if defined(ACT_PARALLEL)
    const int64_t row_block = ROW_BLOCK_SIZE;
    const int64_t col_block = COL_BLOCK_SIZE;

    for (int64_t c0 = 0; c0 < nc; c0 += col_block) {
        int64_t cur_c = (c0 + col_block <= nc) ? col_block : (nc - c0);
        for (int64_t r0 = 0; r0 < nr; r0 += row_block) {
            int64_t cur_r = (r0 + row_block <= nr) ? row_block : (nr - r0);
            const void * vy_r = (const uint8_t *)vy + r0 * n;
            for (int64_t c = 0; c < cur_c; ++c) {
                const int64_t col = c0 + c;
                float * s_col = s + col;
                const void * vx_col = (const uint8_t *)vx + col * n / 4;
                ggml_vec_dot_i2_i8_s(n, s_col + r0 * bs, bs, vx_col, n, vy_r, n, cur_r);
            }
        }
    }
#else
    const int64_t row_block = ROW_BLOCK_SIZE;
    const int64_t col_block = COL_BLOCK_SIZE;

    for (int64_t r0 = 0; r0 < nr; r0 += row_block) {
        int64_t cur_r = (r0 + row_block <= nr) ? row_block : (nr - r0);
        for (int64_t c0 = 0; c0 < nc; c0 += col_block) {
            int64_t cur_c = (c0 + col_block <= nc) ? col_block : (nc - c0);
            const void * vx_c = (const uint8_t *)vx + c0 * n / 4;
            for (int64_t r = 0; r < cur_r; ++r) {
                const int64_t row = r0 + r;
                float * s_row = s + row * bs;
                const void * vy_row = (const uint8_t *)vy + row * n;
                ggml_vec_dot_i2_i8_s(n, s_row + c0, bs, vx_c, n, vy_row, n, cur_c);
            }
        }
    }
#endif
}

// Helper function to get current time in nanoseconds
double get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

// Initialize matrix with random i2 values (2-bit quantized)
void init_matrix_i2(uint8_t* data, int n, int cols) {
    // i2 format: 4 values per byte (2 bits each)
    int total_bytes = n * cols / 4;
    for (int i = 0; i < total_bytes; i++) {
        data[i] = rand() & 0xFF;
    }
}

// Initialize matrix with random i8 values
void init_matrix_i8(int8_t* data, int n, int rows) {
    int total_elements = n * rows;
    for (int i = 0; i < total_elements; i++) {
        data[i] = (int8_t)((rand() % 256) - 128);
    }
}

// Benchmark configuration
struct BenchmarkConfig {
    int n;           // embedding dimension (must be divisible by 4)
    int nr;          // number of rows in Y matrix
    int nc;          // number of columns in X matrix
    int iterations;  // number of benchmark iterations
    int warmup;      // number of warmup iterations
};

void print_config(const BenchmarkConfig& config) {
    printf("=" "=%.78s\n", "===============================================================================");
    printf("Benchmark Configuration:\n");
    printf("=" "=%.78s\n", "===============================================================================");
    printf("  Embedding dimension (n)    : %d\n", config.n);
    printf("  Matrix Y rows (nr)         : %d\n", config.nr);
    printf("  Matrix X columns (nc)      : %d\n", config.nc);
    printf("  Iterations                 : %d\n", config.iterations);
    printf("  Warmup iterations          : %d\n", config.warmup);
    printf("\nMatrix sizes:\n");
    printf("  X (i2): %d x %d (%.2f KB)\n", config.nc, config.n, 
           (config.nc * config.n / 4) / 1024.0);
    printf("  Y (i8): %d x %d (%.2f KB)\n", config.nr, config.n,
           (config.nr * config.n) / 1024.0);
    printf("  S (f32): %d x %d (%.2f KB)\n", config.nr, config.nc,
           (config.nr * config.nc * sizeof(float)) / 1024.0);
    printf("\nGEMM Config:\n");
#if defined(ACT_PARALLEL)
    printf("  ACT_PARALLEL              : ON\n");
#else
    printf("  ACT_PARALLEL              : OFF\n");
#endif
    printf("  ROW_BLOCK_SIZE            : %d\n", ROW_BLOCK_SIZE);
    printf("  COL_BLOCK_SIZE            : %d\n", COL_BLOCK_SIZE);
    printf("  PARALLEL_SIZE             : %d\n", PARALLEL_SIZE);
    printf("=" "=%.78s\n\n", "===============================================================================");
}

void run_benchmark(const BenchmarkConfig& config) {
    // Allocate matrices
    printf("Allocating matrices...\n");
    
    // X matrix (i2 format): nc x n, but stored as nc x (n/4) bytes
    uint8_t* X = (uint8_t*)malloc(config.nc * config.n / 4);
    
    // Y matrix (i8 format): nr x n
    int8_t* Y = (int8_t*)malloc(config.nr * config.n);
    
    // Result matrix (float32): nr x nc
    float* S = (float*)malloc(config.nr * config.nc * sizeof(float));
    
    if (!X || !Y || !S) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(1);
    }
    
    // Initialize matrices with random data
    printf("Initializing matrices with random data...\n");
    srand(time(NULL));
    init_matrix_i2(X, config.n, config.nc);
    init_matrix_i8(Y, config.n, config.nr);
    memset(S, 0, config.nr * config.nc * sizeof(float));
    
    // Warmup
    printf("Running %d warmup iterations...\n", config.warmup);
    for (int i = 0; i < config.warmup; i++) {
        ggml_gemm_i2_i8_s(config.n, S, config.nc, X, Y, config.nr, config.nc);
    }
    
    // Benchmark
    printf("Running %d benchmark iterations...\n", config.iterations);
    double total_time = 0.0;
    double min_time = 1e20;
    double max_time = 0.0;
    
    for (int i = 0; i < config.iterations; i++) {
        double start = get_time_ns();
        ggml_gemm_i2_i8_s(config.n, S, config.nc, X, Y, config.nr, config.nc);
        double end = get_time_ns();
        
        double elapsed = end - start;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;
        
        if ((i + 1) % 100 == 0) {
            printf("  Progress: %d/%d iterations\n", i + 1, config.iterations);
        }
    }
    
    // Calculate statistics
    double avg_time_ns = total_time / config.iterations;
    double avg_time_ms = avg_time_ns / 1e6;
    double min_time_ms = min_time / 1e6;
    double max_time_ms = max_time / 1e6;
    
    // Calculate GFLOPS
    // For GEMM: nr x nc x n multiply-adds = 2 * nr * nc * n FLOPs
    double flops = 2.0 * config.nr * config.nc * config.n;
    double gflops = (flops / avg_time_ns);
    
    // Calculate throughput (tokens/s assuming each column is a token)
    double throughput = (config.nc * 1e9) / avg_time_ns;
    
    // Print results
    printf("\n");
    printf("=" "=%.78s\n", "===============================================================================");
    printf("Benchmark Results:\n");
    printf("=" "=%.78s\n", "===============================================================================");
    printf("  Average time  : %.3f ms\n", avg_time_ms);
    printf("  Min time      : %.3f ms\n", min_time_ms);
    printf("  Max time      : %.3f ms\n", max_time_ms);
    printf("  Std dev       : %.3f ms\n", sqrt((max_time_ms - min_time_ms) * (max_time_ms - min_time_ms) / 12));
    printf("\nPerformance:\n");
    printf("  GFLOPS        : %.2f\n", gflops);
    printf("  Throughput    : %.2f tokens/s\n", throughput);
    printf("  Latency/token : %.3f us\n", (avg_time_ms * 1000) / config.nc);
    printf("=" "=%.78s\n", "===============================================================================");
    
    // Cleanup
    free(X);
    free(Y);
    free(S);
}

void print_usage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("Options:\n");
    printf("  -n <size>    Embedding dimension (must be divisible by 4, default: 2048)\n");
    printf("  -r <rows>    Number of rows in matrix Y (default: 32)\n");
    printf("  -c <cols>    Number of columns in matrix X (default: 128)\n");
    printf("  -i <iters>   Number of iterations (default: 1000)\n");
    printf("  -w <warmup>  Number of warmup iterations (default: 10)\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    BenchmarkConfig config = {
        .n = 2048,
        .nr = 32,
        .nc = 128,
        .iterations = 1000,
        .warmup = 10
    };
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            config.n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            config.nr = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            config.nc = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            config.iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            config.warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate configuration
    if (config.n % 4 != 0) {
        fprintf(stderr, "Error: Embedding dimension (-n) must be divisible by 4\n");
        return 1;
    }
    
    if (config.n <= 0 || config.nr <= 0 || config.nc <= 0 || config.iterations <= 0) {
        fprintf(stderr, "Error: All size parameters must be positive\n");
        return 1;
    }
    
    // Run benchmark
    print_config(config);
    run_benchmark(config);
    
    return 0;
}
