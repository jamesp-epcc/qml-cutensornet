#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "mps.h"

#define HANDLE_CUDA_ERROR(x)                           \
{                                                      \
    const auto err = x;                                \
    if (err != cudaSuccess) {                          \
	std::cout << "Error: " << cudaGetErrorString(err) << " in line " << __LINE__ << std::endl; \
	std::exit(1);                                  \
    }                                                  \
};

MatrixProductState::MatrixProductState(int32_t numQubits, int64_t maxVirtualExtent,
				       double truncationFidelity)
{
    numQubits_ = numQubits;
    physExtent_ = 2;
    truncationFidelity_ = truncationFidelity;
    zero_ = 1e-16;

    // initialise modes and extents arrays
    for (int i = 0; i < (numQubits_ + 1); i++) {
	extentsPerQubit_.push_back(1);
	virtualModes_.push_back(nextMode_++);
	if (i != numQubits_) physModes_.push_back(nextMode_++);
    }
    int64_t untruncatedMaxExtent = (int64_t)std::pow(physExtent_, numQubits_/2);
    maxVirtualExtent_ = maxVirtualExtent == 0 ? untruncatedMaxExtent : std::min(maxVirtualExtent, untruncatedMaxExtent);

    // work out maximum number of elements for each tensor
    size_t maxMaxTensorElements = 0;
    int64_t maxLeftExtent = 1;
    for (int i = 0; i < numQubits_; i++) {
	int64_t maxRightExtent = std::min(std::min((int64_t)std::pow(physExtent_, i+1),
						   (int64_t)std::pow(physExtent_, numQubits_-i-1)),
					  maxVirtualExtent_);
	size_t maxHere = physExtent_ * maxLeftExtent * maxRightExtent;
	maxTensorElements_.push_back(maxHere);
	if (maxHere > maxMaxTensorElements) maxMaxTensorElements = maxHere;
	maxLeftExtent = maxRightExtent;
    }
    
    // allocate and initialise tensors
    for (int i = 0; i < numQubits_; i++) {
	complex_t* data_h = new complex_t[maxTensorElements_[i]];
	void* data_d;
	size_t maxSize = maxTensorElements_[i] * sizeof(complex_t);
	HANDLE_CUDA_ERROR(cudaMalloc(&data_d, maxSize));
	std::memset(data_h, 0, maxSize);
	data_h[0] = complex_t(1.0, 0.0);
	HANDLE_CUDA_ERROR(cudaMemcpy(data_d, data_h, maxSize, cudaMemcpyHostToDevice));
	qubitTensor_.push_back(data_d);
	qubitTensorHost_.push_back(data_h);
    }
}

MatrixProductState::~MatrixProductState()
{
    for (int i = 0; i < numQubits_; i++) {
	delete[] qubitTensorHost_[i];
	cudaFree(qubitTensor_[i]);
    }
}

// returns pointer to next character after this tensor, or nullptr on error
char* MatrixProductState::loadFromString(char* str, bool conj)
{
    // parse out each tensor in turn
    // first pass, work out the sizes
    int p = 0;
    for (int i = 0; i < numQubits_; i++) {
	int shape[3] = { 0, 0, 2 };
	int depth = 0;

	int startp = p;
	do {
	    // skip whitespace before each token
	    while ((str[p]) && (std::isspace(str[p]))) p++;
	    
	    if (str[p] == '[') {
		if (depth == 1) shape[0]++;
		if ((depth == 2) && (shape[0] == 1)) shape[1]++;
		depth++;
		p++;
	    }
	    else if (str[p] == ']') {
		depth--;
		p++;
	    }
	    else {
		if ((depth != 0) && (str[p] == 0)) {
		    return nullptr;
		}
		p++;
	    }
	    if ((depth < 0) || (depth > 3)) {
		return nullptr;
	    }
	} while (depth > 0);
	//std::cout << "qubit " << i << " tensor is of shape (" << shape[0] << "," << shape[1] << "," << shape[2] << ")" << std::endl;

	// set qubit shape and sanity check
	if (extentsPerQubit_[i] != shape[0]) {
	    std::cerr << "Shape mismatch for qubit " << i << "!" << std::endl;
	    return nullptr;
	}
	extentsPerQubit_[i+1] = shape[1];
	
	// read qubit values
	p = startp;
	int k = 0;
	do {
	    // skip whitespace before each token
	    while ((str[p]) && (std::isspace(str[p]))) p++;
      	    if (str[p] == '[') {
		depth++;
		p++;
	    }
	    else if (str[p] == ']') {
		depth--;
		p++;
	    }
	    else if ((str[p] == '-') || (std::isdigit(str[p]))) {
		// found start of number. find end of real part
		int sor = p;
		while (((str[p] != '-') && (str[p] != '+')) || (str[p-1] == 'e') || (p == sor)) p++;
		int eor = p;
		int soi = p;
		if (str[p] == '+') soi = p + 1;
		// find end of imaginary part
		while (str[p] != 'j') p++;
		int eoi = p;
		p++; // skip past j

		char numbuf[100];
		std::memcpy(numbuf, &str[sor], eor - sor);
		numbuf[eor - sor] = 0;
		double re = std::atof(numbuf);
		std::memcpy(numbuf, &str[soi], eoi - soi);
		numbuf[eoi - soi] = 0;
		double im = std::atof(numbuf);
		if (conj) im = -im;
		qubitTensorHost_[i][k] = complex_t(re, im);
		k++;
	    }
	    else {
		std::cerr << "Unexpected character " << str[p] << std::endl;
		return nullptr;
	    }
	} while (depth > 0);

	// copy to GPU
	HANDLE_CUDA_ERROR(cudaMemcpy(qubitTensor_[i], qubitTensorHost_[i],
				     shape[0] * shape[1] * shape[2] * sizeof(complex_t),
				     cudaMemcpyHostToDevice));
    }
    return str + p;
}

void MatrixProductState::printTensors()
{
    for (int i = 0; i < numQubits_; i++) {
	size_t maxSize = sizeof(complex_t) * maxTensorElements_[i];
	HANDLE_CUDA_ERROR(cudaMemcpy(qubitTensorHost_[i], qubitTensor_[i],
				     maxSize, cudaMemcpyDeviceToHost));
	size_t actualSize = 2 * extentsPerQubit_[i] * extentsPerQubit_[i+1];
	std::cout << "qubit " << i << ": (" << extentsPerQubit_[i] << "x" << extentsPerQubit_[i+1] << "x2) [";
	for (size_t j = 0; j < actualSize; j++) {
	    std::cout << qubitTensorHost_[i][j] << ", ";
	}
	std::cout << "]" << std::endl;
    }
}

int MatrixProductState::getSerialisedSize()
{
    int arraySize = numQubits_ + 1; // additional space for holding extents
    for (int i = 0; i < numQubits_; i++) {
	arraySize += maxTensorElements_[i];
    }
    return arraySize;
}

void MatrixProductState::serialise(std::vector<complex_t> &vec)
{
    // work out size required
    int arraySize = getSerialisedSize();
    vec.resize(arraySize);

    // copy in tensor extents
    int k = 0;
    for (int i = 0; i < (numQubits_ + 1); i++) {
	vec[k] = complex_t(extentsPerQubit_[i]);
	k++;
    }

    // copy data in from GPU
    for (int i = 0; i < numQubits_; i++) {
	HANDLE_CUDA_ERROR(cudaMemcpy(vec.data() + k, qubitTensor_[i],
				     maxTensorElements_[i] * sizeof(complex_t),
				     cudaMemcpyDeviceToHost));
	k += maxTensorElements_[i];
    }
}

void MatrixProductState::deserialise(std::vector<complex_t> &vec)
{
    // sanity check size
    int arraySize = getSerialisedSize();
    if (vec.size() != arraySize) {
	std::cerr << "Error deserialising MPS: expected " << arraySize << " elements, got " << vec.size() << std::endl;
	return;
    }

    // read out extents
    int k = 0;
    for (int i = 0; i < (numQubits_ + 1); i++) {
	extentsPerQubit_[i] = int64_t(vec[k].real());
	k++;
    }

    // copy data directly to GPU
    for (int i = 0; i < numQubits_; i++) {
	HANDLE_CUDA_ERROR(cudaMemcpy(qubitTensor_[i], vec.data() + k,
				     maxTensorElements_[i] * sizeof(complex_t),
				     cudaMemcpyHostToDevice));
	k += maxTensorElements_[i];
    }
}

// mode 1 is used for result of vdot
int32_t MatrixProductState::nextMode_ = 2;
