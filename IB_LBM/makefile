PROG = IBLBM.x
SRC = IBLBM.cu
SRC2 = function.h
CUDA = /usr/local/cuda-11.3/bin/nvcc
#CUDA = /usr/local/cuda-11.3/bin/nvcc
#CUDA = nvcc
CUDAFLAGS = -arch sm_70  # -ccbin=/usr/bin/clang
CPPLAGS  = -O3 

compile: $(PROG)

$(PROG): $(SRC) $(SRC2)
	$(CUDA) $(SRC) $(CUDAFLAGS) $(CPPFLAGS) -o $@

clean:
	rm -rf *.o *.x out* *.log *.txt pout*
