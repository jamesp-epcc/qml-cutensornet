CUQUANTUM_ROOT=/work/ic081/ic081/jamesp-ic081/fraud/venv_cuquantum/lib/python3.11/site-packages/cuquantum
CUTENSOR_ROOT=/work/ic081/ic081/jamesp-ic081/fraud/venv_cuquantum/lib/python3.11/site-packages/cutensor
CUDA_HOME=/work/y07/shared/cirrus-software/nvidia/hpcsdk-22.11/Linux_x86_64/22.11/cuda

NVCCFLAGS=-I${CUQUANTUM_ROOT}/include -I${CUTENSOR_ROOT}/include -Xcompiler -fopenmp
CFLAGS=-I${CUQUANTUM_ROOT}/include -I${CUTENSOR_ROOT}/include -I${CUDA_HOME}/include -fopenmp
LDFLAGS=-L${CUQUANTUM_ROOT}/lib -L${CUTENSOR_ROOT}/lib -lcutensornet -lcutensor -fopenmp -lcudart

OBJS=build_matrix.o mps.o vdot_calculator.o

build_matrix: ${OBJS}
	mpicxx -o build_matrix ${OBJS} ${LDFLAGS}

build_matrix.o: build_matrix.cpp mps.h vdot_calculator.h
	mpicxx -c build_matrix.cpp -o build_matrix.o ${CFLAGS}

mps.o: mps.cu mps.h
	nvcc -c mps.cu -o mps.o ${NVCCFLAGS}

vdot_calculator.o: vdot_calculator.cu vdot_calculator.h mps.h
	nvcc -c vdot_calculator.cu -o vdot_calculator.o ${NVCCFLAGS}

clean:
	rm -f build_matrix *.o
