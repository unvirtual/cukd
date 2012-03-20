GCC44BIND=-arch=sm_21 --compiler-options "-O3" --use_fast_math
CUDAINC=-I/opt/cuda-toolkit/include -Icutl
CUDASDKINC=-I/opt/cuda-sdk/C/common/inc
CUTILINC=-I/opt/cuda-sdk/C/common/inc
CUTILLD=-L/opt/cuda-sdk/C/common/lib -L/opt/cuda-sdk/CUDALibraries/common/lib -lcuda -lcutil_x86_64 -lstdc++

VPATH=cutl/utils:cutl/algorithms:test:detail
BUILDDIR=./build
SRC=reduction.cu mesh.cu primitives.cu kdtree.cu kdtree_kernels.cu kdtree_node_array.cu kdtree_node_array_kernels.cu node_chunk_array.cu node_chunk_array_kernels.cu split_candidate_array.cu small_node_array.cu small_node_array_kernels.cu
OBJ=$(patsubst %.cu,build/%.o,$(SRC))

all: $(OBJ) build/large_node_stage build/test_full

build/%.o: %.cu
	nvcc -c $< -o $@ $(CUDAINC) $(CUDASDKINC) $(GCC44BIND) $(CUTILINC) $(CUTILLD) -I. 

build/large_node_stage: $(OBJ) test/large_node_stage.cu
	nvcc -o $@ test/large_node_stage.cu $(OBJ) $(CUDAINC) $(CUDASDKINC) $(GCC44BIND) $(CUTILINC) $(CUTILLD) -I. 

build/test_full: $(OBJ) test/test_full.cu
	nvcc -o $@ test/test_full.cu $(OBJ) $(CUDAINC) $(CUDASDKINC) $(GCC44BIND) $(CUTILINC) $(CUTILLD) -I. 

clean:
	rm -f build/*
