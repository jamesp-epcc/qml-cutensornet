#ifndef _VDOT_CALCULATOR_H_
#define _VDOT_CALCULATOR_H_

#include "mps.h"

#include <cuda_runtime.h>
#include <cutensornet.h>

class VdotCalculator {
 public:
    VdotCalculator(cudaDataType_t typeData, cutensornetComputeType_t typeCompute,
		   int numQubits, int physExtent);
    ~VdotCalculator();

    void vdot(MatrixProductState& mps1, MatrixProductState& mps2, complex_t* result);
    
 private:
    cudaDataType_t typeData_;
    cutensornetComputeType_t typeCompute_;

    int numQubits_;
    int physExtent_;

    std::vector<int32_t> numModesIn_;
    std::vector<int64_t*> extentsIn_;
    std::vector<int64_t*> stridesIn_;
    std::vector<int32_t*> modesIn_;

    int64_t extentsOut_[1];
    int32_t modesOut_[1];

    std::vector<void*> rawDataIn_;
    
    cutensornetHandle_t handle_;
    cudaStream_t stream_;

    int64_t workspaceSize_;
    cutensornetWorkspaceDescriptor_t workDesc_;
    void* workspace_;

    cutensornetContractionOptimizerConfig_t optimizerConfig_;
};

#endif
