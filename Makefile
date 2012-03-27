GCC44BIND=-arch=sm_21 --compiler-options "-O3" --use_fast_math
CUDAINC=-I/opt/cuda-toolkit/include 
CUDASDKINC=-I/opt/cuda-sdk/C/common/inc
CUTILINC=-I/opt/cuda-sdk/C/common/inc
CUTILLD=-L/opt/cuda-sdk/C/common/lib -L/opt/cuda-sdk/CUDALibraries/common/lib -lcuda -lcutil_x86_64 -lstdc++

VPATH=utils:algorithms:test:detail
BUILDDIR=./build
SRC=reduction.cu primitives.cu kdtree.cu kdtree_kernels.cu kdtree_node_array.cu kdtree_node_array_kernels.cu node_chunk_array.cu node_chunk_array_kernels.cu split_candidate_array.cu small_node_array.cu small_node_array_kernels.cu
OBJ=$(patsubst %.cu,build/%.o,$(SRC))

all: $(OBJ) libcukd.a build/ray_traversal

libcukd.a: $(OBJ)
	nvcc -lib -o libcukd.a $(OBJ)

build/%.o: %.cu
	nvcc -c $< -o $@ $(CUDAINC) $(CUDASDKINC) $(GCC44BIND) $(CUTILINC) $(CUTILLD) -I. 

build/ray_traversal: libcukd.a test/ray_traversal.cu
	nvcc -o $@ test/ray_traversal.cu $(OBJ) $(CUDAINC) $(CUDASDKINC) $(GCC44BIND) $(CUTILINC) $(CUTILLD) -L. -I. -lSDL -lcukd

build/benchmark: $(OBJ) test/benchmark.cu
	nvcc -o $@ test/benchmark.cu $(OBJ) $(CUDAINC) $(CUDASDKINC) $(GCC44BIND) $(CUTILINC) $(CUTILLD) -I. -lSDL

clean:
	rm -f build/*
