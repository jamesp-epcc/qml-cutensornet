/*
 * Class representing a matrix product state, stored in GPU memory
 */

#include <complex>
#include <vector>

#include <cuda_runtime.h>
#include <cutensornet.h>

#ifndef _MPS_H_
#define _MPS_H_

typedef std::complex<double> complex_t;

class MatrixProductState {
 public:
    MatrixProductState(int32_t numQubits, int64_t maxVirtualExtent,
		       double truncationFidelity=1.0);
    ~MatrixProductState();

    
    const char* loadFromString(const char* str, bool conj=false);

    void printTensors();

 private:
    int32_t numQubits_;
    int64_t physExtent_;
    int64_t maxVirtualExtent_;

    double truncationFidelity_;
    double zero_;

    std::vector<int32_t> physModes_;
    std::vector<int32_t> virtualModes_;
    std::vector<int64_t> extentsPerQubit_;

    std::vector<size_t> maxTensorElements_;
    std::vector<void*> qubitTensor_;
    std::vector<complex_t*> qubitTensorHost_;

    static int32_t nextMode_;
};

#endif
