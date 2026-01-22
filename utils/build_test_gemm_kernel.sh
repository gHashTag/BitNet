#!/bin/bash
# Build script for standalone GEMM kernel benchmark

set -e

echo "Building GEMM kernel benchmark..."

# Compiler settings
CXX=${CXX:-g++}
BUILD_DIR="../build"
SRC_DIR="../src"

# Create build directory if it doesn't exist
mkdir -p ${BUILD_DIR}

# Compiler flags
CXXFLAGS="-O3 -march=native -mtune=native -std=c++17 -fopenmp"
CXXFLAGS+=" -I.. -I../include"
CXXFLAGS+=" -I../3rdparty/llama.cpp/ggml/include"
CXXFLAGS+=" -I../3rdparty/llama.cpp/ggml/src"
CXXFLAGS+=" -I../3rdparty/llama.cpp/include"
CXXFLAGS+=" -DNDEBUG -ffast-math"

# Link flags
LDFLAGS="-lm -lpthread"

# Link with pre-built libraries
GGML_LIB_DIR="../build/3rdparty/llama.cpp/ggml/src"
GGML_SO="${GGML_LIB_DIR}/libggml.so"

if [ ! -f "${GGML_SO}" ]; then
    echo "⚠️  Warning: Cannot find libggml.so"
    echo "Please build the project first with: cmake --build build"
    exit 1
fi

LDFLAGS+=" -L${GGML_LIB_DIR} -lggml -Wl,-rpath,\$ORIGIN/../../${GGML_LIB_DIR}"
echo "Linking with libggml.so"

# Source files
SOURCES="./test_gemm_kernel.cpp"

# Output binary
OUTPUT="${BUILD_DIR}/test_gemm_kernel"

echo "Compiler: ${CXX}"
echo "Flags: ${CXXFLAGS}"
echo "Sources: ${SOURCES}"
echo ""

# Build
${CXX} ${CXXFLAGS} ${SOURCES} -o ${OUTPUT} ${LDFLAGS}

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo "Output: ${OUTPUT}"
    echo ""
    echo "Usage examples:"
    echo "  # Default test (n=2048, nr=32, nc=128, 1000 iterations)"
    echo "  ${OUTPUT}"
    echo ""
    echo "  # Custom matrix sizes"
    echo "  ${OUTPUT} -n 4096 -r 64 -c 256"
    echo ""
    echo "  # Quick test (fewer iterations)"
    echo "  ${OUTPUT} -i 100 -w 5"
    echo ""
    echo "  # Large-scale test"
    echo "  ${OUTPUT} -n 3200 -r 128 -c 512 -i 500"
    echo ""
else
    echo ""
    echo "❌ Build failed!"
    exit 1
fi
