#
# Simulation of rainwater flooding
# 
#

# Compilers
CC=gcc
CUDACC=nvcc

# Flags for optimization and libs
FLAGS=-O3 -Wall
CUFLAGS=-O3 
LIBS=-lm
CULIBS=-lm rng.c

# Targets to build
OBJS=flood_seq flood_cuda

# Rules. By default show help
help:
	@echo
	@echo "Simulation of rainwater flooding"
	@echo
	@echo "make flood_seq	Build only the sequential version"
	@echo "make flood_cuda	Build only the CUDA version"
	@echo
	@echo "make all		Build all versions (Sequential & CUDA)"
	@echo "make debug		Build sequential version with demo output for small surfaces"
	@echo "make animation		Build the sequential version to produce the animation data"
	@echo "make clean		Remove targets"
	@echo "make test_seq		Build and run the sequential version with a simple input sequence (not suitable for DAS5)"
	@echo "make test_seq_remote	Build and run the sequential version with a simple input sequence on a compute node"
	@echo "make test_cuda		Build and run the CUDA version with a simple input sequence on a compute node"
	@echo

all: $(OBJS)

flood.o: flood.c
	$(CC) $(FLAGS) $(DEBUG) -c $< -o $@

flood_seq.o: flood_seq.c
	$(CC) $(FLAGS) $(DEBUG) -c $< -o $@

flood_seq: flood.o flood_seq.o
	$(CC) $(DEBUG) $^ $(LIBS) -o $@

flood_cuda.o: flood_cuda.cu
	$(CUDACC) $(CUFLAGS) $(DEBUG) -c $< -o $@

flood_cuda: flood.o flood_cuda.o
	$(CUDACC) $(DEBUG) $^ $(CULIBS) -o $@

# Remove the target files
clean:
	rm -rf $(OBJS) *.o

# Compile in debug mode (currently sequential version only)
debug:
	make FLAGS="$(FLAGS) -DDEBUG -g" flood_seq

# Compile to generate animation (currently sequential version only)
animation:
	make FLAGS="$(FLAGS) -DDEBUG -DANIMATION -g" flood_seq

animation_cuda:
	make CUFLAGS="$(CUFLAGS) -DDEBUG -DANIMATION -g" FLAGS="$(FLAGS) -DDEBUG -DANIMATION -g" flood_cuda

test_seq: flood_seq
	./flood_seq $(cat test_files/debug.in)

test_seq_remote: flood_seq
	prun -t 15:00 -np 1 -native '-C gpunode' ./flood_seq $$(cat test_files/debug.in)

test_cuda: flood_cuda
	prun -t 15:00 -np 1 -native '-C gpunode' ./flood_cuda $$(cat test_files/debug.in)
