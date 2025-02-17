// FIXME: I think matrix dimensions are completely wrong for MPI runs.
// num_mps_x and num_mps_y correspond to the number of items in the input files,
// which is the ones the local process has, but they're used for the dimensions of
// the entire matrix, which should be global!
#include "mps.h"
#include "vdot_calculator.h"

#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/time.h>

#include <cuda_runtime.h>

#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define HANDLE_CUDA_ERROR(x)                           \
{                                                      \
    const auto err = x;                                \
    if (err != cudaSuccess) {                          \
	std::cout << "Error: " << cudaGetErrorString(err) << " in line " << __LINE__ << std::endl; \
	std::exit(1);                                  \
    }                                                  \
};

// gets the current time in seconds to a high resolution
static double getTime()
{
    const double micro = 1.0e-06;
    double wall_time;
    struct timeval tp;
    
    if (gettimeofday( &tp, NULL) == -1) {
    	wall_time = -1.0e0;
    }
    else {
	wall_time = (double) (tp.tv_sec + micro*tp.tv_usec);
    }

    return wall_time;
}

// loads the contents of a text file into a memory buffer (null-terminated)
// returns null on error
static char* loadFile(const char* filename)
{
    std::FILE* f;
    f = std::fopen(filename, "r");
    if (f == nullptr) return nullptr;

    std::fseek(f, 0, SEEK_END);
    int64_t len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    char* buf = new char[len + 1];
    std::fread(buf, 1, len, f);
    std::fclose(f);
    buf[len] = 0;
    return buf;
}

// parses an integer from the string pointer (ignoring whitespace)
// returns pointer to character after integer
static char* readInt(char* str, int& result)
{
    int p = 0;
    // skip whitespace
    while ((str[p]) && (std::isspace(str[p]))) p++;
    if (!std::isdigit(str[p])) return nullptr;
    int start = p;
    while (std::isdigit(str[p])) p++;
    int end = p;

    char numbuf[50];
    std::strncpy(numbuf, &str[start], end - start);
    numbuf[end - start] = 0;
    result = std::atoi(numbuf);

    return str + p;
}

// returns true if the next item in the string is 'None'
static bool isNone(char* str)
{
    int p = 0;
    // skip whitespace
    while ((str[p]) && (std::isspace(str[p]))) p++;
    if ((str[p] == 0) || (str[p+1] == 0) || (str[p+2] == 0) || (str[p+3] == 0) ||
	(str[p] != 'N') || (str[p+1] != 'o') || (str[p+2] != 'n') || (str[p+3] != 'e')) return false;
    return true;
}

static int sendMPS(MatrixProductState& mps, unsigned int dest)
{
    // FIXME: could allocate one of these per thread instead of reallocating locally
    // each time. Or store a serialisation buffer with each MPS.
    std::vector<complex_t> vec;
    mps.serialise(vec);
    return MPI_Send(vec.data(), vec.size(), MPI_C_DOUBLE_COMPLEX, dest, 0,
		    MPI_COMM_WORLD);
}

static int receiveMPS(MatrixProductState& mps, unsigned int source)
{
    MPI_Status status;
    // FIXME: could allocate one of these per thread instead of reallocating locally
    // each time. Or store a serialisation buffer with each MPS.
    std::vector<complex_t> vec;
    vec.resize(mps.getSerialisedSize());
    int err = MPI_Recv(vec.data(), vec.size(), MPI_C_DOUBLE_COMPLEX, source, 0,
		       MPI_COMM_WORLD, &status);
    if (err == MPI_SUCCESS) {
	mps.deserialise(vec);
    }
    return err;
}

static int sendRecvMPS(MatrixProductState& mps, unsigned int dest, unsigned int source)
{
    MPI_Status status;
    // FIXME: again, could manage this memory more efficiently
    std::vector<complex_t> sendVec, recvVec;
    mps.serialise(sendVec);
    recvVec.resize(sendVec.size());
    int err = MPI_Sendrecv(sendVec.data(), sendVec.size(), MPI_C_DOUBLE_COMPLEX,
			   dest, 0, recvVec.data(), recvVec.size(),
			   MPI_C_DOUBLE_COMPLEX, source, 0, MPI_COMM_WORLD,
			   &status);
    if (err == MPI_SUCCESS) {
	mps.deserialise(recvVec);
    }
    return err;
}

int main(int argc, char* argv[])
{
    unsigned int rank, numProcs;
    int mpiError;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, (int*)&numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, (int*)&rank);
    std::cout << "This is process " << rank << " of " << numProcs << std::endl; 
    
    int numThreads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
	numThreads = omp_get_num_threads();
    }
#endif
    int numGPUs;
    HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numGPUs));
    
    std::cout << "Running with " << numThreads << " threads and " << numGPUs << " GPUs" << std::endl;
    
    double truncation_error = 1e-16;
    
    if ((argc != 3) || ((strcmp(argv[2], "train")) && (strcmp(argv[2], "test")))) {
	std::cerr << "Usage: build_matrix <data directory> [train|test]" << std::endl;
	MPI_Finalize();
	return 1;
    }
    char* datadir = argv[1];
    char* trainOrTest = argv[2];
    char filenameBuffer[500];
    
    // read the input files
    sprintf(filenameBuffer, "%s/mps_x_%s_%d.txt", datadir, trainOrTest, rank);
    char* mps_x_str = loadFile(filenameBuffer);
    if (mps_x_str == nullptr) {
	std::cerr << "Error loading MPS X input file " << filenameBuffer << std::endl;
	MPI_Finalize();
	return 1;
    }
    
    sprintf(filenameBuffer, "%s/mps_y_%s_%d.txt", datadir, trainOrTest, rank);
    char* mps_y_str = loadFile(filenameBuffer);
    if (mps_y_str == nullptr) {
	std::cerr << "Error loading MPS Y input file " << filenameBuffer << std::endl;
	MPI_Finalize();
	return 1;
    }

    // extract number of MPSs and number of qubits from input files
    int num_mps_x, num_qubit_x;
    char *mps_x_ptr = readInt(mps_x_str, num_mps_x);
    if (!mps_x_ptr) {
	std::cerr << "Error parsing header of MPS X input file" << std::endl;
	MPI_Finalize();
	return 1;
    }
    mps_x_ptr = readInt(mps_x_ptr, num_qubit_x);
    if (!mps_x_ptr) {
	std::cerr << "Error parsing header of MPS X input file" << std::endl;
	MPI_Finalize();
	return 1;
    }

    int num_mps_y, num_qubit_y;
    char *mps_y_ptr = readInt(mps_y_str, num_mps_y);
    if (!mps_y_ptr) {
	std::cerr << "Error parsing header of MPS Y input file" << std::endl;
	MPI_Finalize();
	return 1;
    }
    mps_y_ptr = readInt(mps_y_ptr, num_qubit_y);
    if (!mps_y_ptr) {
	std::cerr << "Error parsing header of MPS Y input file" << std::endl;
	MPI_Finalize();
	return 1;
    }    
    
    // check that they match
    if (num_qubit_x != num_qubit_y) {
	std::cerr << "Dimension mismatch between MPS X and MPS Y" << std::endl;
	MPI_Finalize();
	return 1;
    }
    std::cout << "Loading " << num_mps_x << " MPSs of " << num_qubit_x << " qubits" << std::endl;

    // create MPS object for each MPS loaded
    std::vector<MatrixProductState*> mps_x, mps_y;
    mps_x.reserve(num_mps_x);
    mps_y.reserve(num_mps_y);
    
    // load Y first in case there are 'None' items
    for (int i = 0; i < num_mps_y; i++) {
	if (isNone(mps_y_ptr)) {
	    std::cout << "Found 'None' in mps Y input, adjusting count to " << i << std::endl;
	    num_mps_y = i;
	    break;
	}
	MatrixProductState* mps = new MatrixProductState(num_qubit_y, 12, 1.0 - truncation_error);
	mps_y_ptr = mps->loadFromString(mps_y_ptr, false);
	if (mps_y_ptr == nullptr) {
	    std::cerr << "Error loading MPS " << i << " from MPS Y" << std::endl;
	    MPI_Finalize();
	    return 1;
	}
	mps_y.push_back(mps);
    }

    std::cout << "Loaded Y MPS" << std::endl;
    
    // load conjugate of X MPSs
    for (int i = 0; i < num_mps_x; i++) {
	MatrixProductState* mps = new MatrixProductState(num_qubit_x, 12, 1.0 - truncation_error);
	mps_x_ptr = mps->loadFromString(mps_x_ptr, true);
	if (mps_x_ptr == nullptr) {
	    std::cerr << "Error loading MPS " << i << " from MPS X" << std::endl;
	    MPI_Finalize();
	    return 1;
	}
	mps_x.push_back(mps);
    }
    
    delete[] mps_x_str;
    delete[] mps_y_str;

    std::cout << "Allocating resources" << std::endl;
    
    // allocate storage for matrix
    // FIXME: derive this properly
    int matrix_w = num_mps_y * 4;
    int matrix_h = num_mps_x * 4;
    complex_t* gpuMatrix;
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&gpuMatrix, matrix_w * matrix_h * sizeof(complex_t)));
    HANDLE_CUDA_ERROR(cudaMemset((void*)gpuMatrix, 0, matrix_w * matrix_h * sizeof(complex_t)));
    complex_t* complexMatrix = new complex_t[matrix_w * matrix_h];
    double* matrix = new double[matrix_w * matrix_h];

    // instantiate a VdotCalculator for each thread
    std::vector<VdotCalculator*> vdcs;
    for (int i = 0; i < numThreads; i++) {
	//HANDLE_CUDA_ERROR(cudaSetDevice(i % numGPUs));
	vdcs.push_back(new VdotCalculator(CUDA_C_64F, CUTENSORNET_COMPUTE_64F, num_qubit_x, 2));
    }

    // work out sizes and counts
    unsigned int entriesPerChunk = num_mps_x;
    unsigned int xChunks = numProcs;
    unsigned int yChunks = xChunks; // FIXME: handle the case where these are not equal!
    unsigned int numProcsInRR = numProcs - (xChunks % yChunks);

    unsigned int iterations = yChunks; // FIXME: handle symmetry optimisation
    
    // compute matrix
    std::cout << "Computing matrix..." << std::endl;
    double t1 = getTime();

    // create local log file
    char logFilename[1000];
    sprintf(logFilename, "/work/ic081/ic081/jamesp-ic081/fraud/mpilog%d.txt", rank);
    std::ofstream logFile(logFilename);

    for (unsigned int it = 0; it < iterations; it++) {
	// fetch MPS for processes that don't fit in round robin
	for (unsigned int procSend = 0; procSend < (xChunks % yChunks); procSend++) {
	    unsigned int procRecv = numProcsInRR + procSend;
	    if (rank == procSend) {
		// send all mps_y via MPI
		for (int i = 0; i < mps_y.size(); i++) {
		    sendMPS(*mps_y[i], procRecv);
		}
	    }
	    if (rank == procRecv) {
		// receive all mps_y via MPI
		for (int i = 0; i < mps_y.size(); i++) {
		    receiveMPS(*mps_y[i], procSend);
		}
	    }
	}
	
#pragma omp parallel for
	for (int i = 0; i < num_mps_x; i++) {
	    int t = 0;
#ifdef _OPENMP
	    t = omp_get_thread_num();
#endif
	    // FIXME: before we can use multiple devices, we need multiple output
	    // buffers
	    //HANDLE_CUDA_ERROR(cudaSetDevice(t % numGPUs));
	    if (t == 0) std::cout << "Row " << i << std::endl;
	    for (int j = 0; j < num_mps_y; j++) {
		// FIXME: handle symmetry optimisation
		int x_index = i + entriesPerChunk * rank;
		int y_index = j + entriesPerChunk * ((rank + it) % yChunks);
		vdcs[t]->vdot(*mps_x[i], *mps_y[j], &gpuMatrix[(y_index * matrix_w) + x_index]);
	    }
	}

	// round robin message passing
	if (rank < numProcsInRR) {
	    // send and receive mps_y via MPI
	    for (int i = 0; i < mps_y.size(); i++) {
		unsigned int recvfrom = (rank+1) % numProcsInRR;
		unsigned int sendto = (rank-1) % numProcsInRR;
		logFile << "Rank " << rank << " sending to " << sendto << " and receiving from " << recvfrom << std::endl;
		logFile << "Before sending: " << std::endl;
		mps_y[i]->printTensors(logFile);
		sendRecvMPS(*mps_y[i], sendto, recvfrom);
		logFile << std::endl << "After receiving: " << std::endl;
		mps_y[i]->printTensors(logFile);
	    }
	}
    }

    double t2 = getTime();
    std::cout << "Matrix computation took " << (t2 - t1) << "s" << std::endl;

    // copy and convert matrix
    HANDLE_CUDA_ERROR(cudaMemcpy(complexMatrix, gpuMatrix, matrix_w * matrix_h * sizeof(complex_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < (matrix_w * matrix_h); i++) {
	complex_t overlap = complexMatrix[i];
	matrix[i] = (overlap * std::conj(overlap)).real();
    }
    double t3 = getTime();
    std::cout << "Matrix copying and conversion took " << (t3 - t2) << "s" << std::endl;

    if (rank == 0) {
	double* finalMatrix = new double[matrix_w * matrix_h];
	// reduce matrix across MPI processes
	MPI_Reduce(matrix, finalMatrix, matrix_w * matrix_h,
		   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	// write matrix to disk
	sprintf(filenameBuffer, "%s/matrix_%s.txt", datadir, trainOrTest);
	std::cout << "Writing matrix to output file" << std::endl;
	std::ofstream of(filenameBuffer);
	if (!of.good()) {
	    std::cerr << "Error opening output file" << filenameBuffer << std::endl;
	    MPI_Finalize();
	    return 1;
	}
	of << std::setprecision(20);
	of << "[ ";
	for (int i = 0; i < matrix_w; i++) {
	    of << "[ ";
	    for (int j = 0; j < matrix_h; j++) {
		of << finalMatrix[(j * matrix_w) + i];
		if (j < (matrix_h - 1)) of << ", ";
	    }
	    of << " ]";
	    if (i < (matrix_w - 1)) of << ",";
	    of << std::endl;
	}
	of << " ]" << std::endl;
	of.close();

	delete[] finalMatrix;
    }
    else {
	// reduce matrix across MPI processes
	MPI_Reduce(matrix, nullptr, matrix_w * matrix_h,
		   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // free resources
    delete[] matrix;
    delete[] complexMatrix;
    cudaFree(gpuMatrix);
		      
    for (int i = 0; i < mps_x.size(); i++) {
	delete mps_x[i];
    }
    for (int i = 0; i < mps_y.size(); i++) {
	delete mps_y[i];
    }

    for (int i = 0; i < vdcs.size(); i++) {
	delete vdcs[i];
    }

    logFile.close();

    MPI_Finalize();
    return 0;
}
