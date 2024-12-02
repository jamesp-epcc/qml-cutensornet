CUQUANTUM_ROOT=/work/ic081/ic081/jamesp-ic081/fraud/venv_cuquantum/lib/python3.11/site-packages/cuquantum
CUTENSOR_ROOT=/work/ic081/ic081/jamesp-ic081/fraud/venv_cuquantum/lib/python3.11/site-packages/cutensor

CFLAGS=-I${CUQUANTUM_ROOT}/include -I${CUTENSOR_ROOT}/include
LDFLAGS=-L${CUQUANTUM_ROOT}/lib -L${CUTENSOR_ROOT}/lib -lcutensornet -lcutensor

OBJS=build_matrix.o mps.o vdot_calculator.o

build_matrix: ${OBJS}
	nvcc -o build_matrix ${OBJS} ${LDFLAGS}

build_matrix.o: build_matrix.cu mps.h vdot_calculator.h
	nvcc -c build_matrix.cu -o build_matrix.o ${CFLAGS}

mps.o: mps.cu mps.h
	nvcc -c mps.cu -o mps.o ${CFLAGS}

vdot_calculator.o: vdot_calculator.cu vdot_calculator.h mps.h
	nvcc -c vdot_calculator.cu -o vdot_calculator.o ${CFLAGS}

clean:
	rm -f build_matrix *.o
