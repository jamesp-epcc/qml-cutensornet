#ifndef _VDOT_CALCULATOR_H_
#define _VDOT_CALCULATOR_H_

#include "mps.h"

#include <cuda_runtime.h>
#include <cutensornet.h>

class VdotCalculator {
 public:
    VdotCalculator(cudaDataType_t typeData, cutensornetComputeType_t typeCompute);
    ~VdotCalculator();

    complex_t vdot(MatrixProductState& mps1, MatrixProductState& mps2);
    
 private:
    cudaDataType_t typeData_;
    cutensornetComputeType_t typeCompute_;

    cutensornetHandle_t handle_;
    cudaStream_t stream_;

    int64_t workspaceSize_;
    cutensornetWorkspaceDescriptor_t workDesc_;
    void* workspace_;

    void* resultGPU_;
};

#endif
