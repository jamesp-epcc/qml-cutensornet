#include "vdot_calculator.h"
#include "mps.h"

#include <iostream>

#define HANDLE_ERROR(x)                                \
{                                                      \
    const auto err = x;	                               \
    if (err != CUTENSORNET_STATUS_SUCCESS) {           \
	std::cout << "Error: " << cutensornetGetErrorString(err) << " in line " << __LINE__ << std::endl; \
	std::exit(1);                                  \
    }                                                  \
};

#define HANDLE_CUDA_ERROR(x)                           \
{                                                      \
    const auto err = x;                                \
    if (err != cudaSuccess) {                          \
	std::cout << "Error: " << cudaGetErrorString(err) << " in line " << __LINE__ << std::endl; \
	std::exit(1);                                  \
    }                                                  \
};

VdotCalculator::VdotCalculator(cudaDataType_t typeData, cutensornetComputeType_t typeCompute)
{
    typeData_ = typeData;
    typeCompute_ = typeCompute;

    // create handle
    HANDLE_ERROR(cutensornetCreate(&handle_));

    // create CUDA stream
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream_));
    
    // create workspace and set its memory
    HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle_, &workDesc_));
    // FIXME: for now, hard code this to be similar size to in test program
    // might want to compute it dynamically in future
    workspaceSize_ = 20 * 1024 * 1024;
    HANDLE_CUDA_ERROR(cudaMalloc(&workspace_, workspaceSize_));
    HANDLE_ERROR(cutensornetWorkspaceSetMemory(handle_,
					       workDesc_,
					       CUTENSORNET_MEMSPACE_DEVICE,
					       CUTENSORNET_WORKSPACE_SCRATCH,
					       workspace_,
					       workspaceSize_));

    HANDLE_CUDA_ERROR(cudaMalloc(&resultGPU_, sizeof(complex_t)));
}

VdotCalculator::~VdotCalculator()
{
    cutensornetDestroyWorkspaceDescriptor(workDesc_);
    cutensornetDestroy(handle_);

    cudaFree(workspace_);
    cudaFree(resultGPU_);
}

complex_t VdotCalculator::vdot(MatrixProductState& mps1, MatrixProductState& mps2)
{
    // FIXME: promote a lot of the locals here to class members so they don't
    // have to be allocated each time
    // sanity check
    if ((mps1.numQubits_ != mps2.numQubits_) ||
	(mps1.physExtent_ != mps2.physExtent_)) {
	std::cerr << "For vdot, both MPS must have same number of qubits and physical extent!" << std::endl;
	return complex_t(0.0, 0.0);
    }

    std::vector<int32_t> numModesIn;
    std::vector<int64_t*> extentsIn;
    std::vector<int64_t*> stridesIn;
    std::vector<int32_t*> modesIn;

    int64_t extentsOut[1] = { 1 };
    int32_t modesOut[1]; // = { nextMode_; };
    int numModesOut = 1;
    modesOut[0] = 1;
    
    // mps1 tensors
    for (int i = 0; i < mps1.numQubits_; i++) {
	numModesIn.push_back(4);
	
	int64_t* extents = new int64_t[4];
	extents[0] = mps1.extentsPerQubit_[i];
	extents[1] = mps1.extentsPerQubit_[i+1];
	extents[2] = mps1.physExtent_;
	extents[3] = 1;
	extentsIn.push_back(extents);

	int64_t* strides = new int64_t[4];
	strides[3] = 1;
	strides[2] = 1;
	strides[1] = extents[2] * strides[2];
	strides[0] = extents[1] * strides[1];
	stridesIn.push_back(strides);

	int32_t* modes = new int32_t[4];
	modes[0] = mps1.virtualModes_[i];
	modes[1] = mps1.virtualModes_[i+1];
	modes[2] = mps1.physModes_[i];
	modes[3] = modesOut[0];
	modesIn.push_back(modes);
    }

    // mps 2 tensors
    for (int i = 0; i < mps2.numQubits_; i++) {
	numModesIn.push_back(4);

	int64_t* extents = new int64_t[4];
	extents[0] = mps2.extentsPerQubit_[i];
	extents[1] = mps2.extentsPerQubit_[i+1];
	extents[2] = mps2.physExtent_;
	extents[3] = 1;
	extentsIn.push_back(extents);

	int64_t* strides = new int64_t[4];
	strides[3] = 1;
	strides[2] = 1;
	strides[1] = extents[2] * strides[2];
	strides[0] = extents[1] * strides[1];
	stridesIn.push_back(strides);
	
	int32_t* modes = new int32_t[4];
	modes[0] = mps2.virtualModes_[i];
	modes[1] = mps2.virtualModes_[i+1];
	modes[2] = mps1.physModes_[i]; // same physical modes as mps 1
	modes[3] = modesOut[0];
	modesIn.push_back(modes);
    }
    
    cutensornetNetworkDescriptor_t descNet;
    HANDLE_ERROR(cutensornetCreateNetworkDescriptor(handle_, mps1.numQubits_*2,
						    numModesIn.data(),
						    extentsIn.data(),
						    stridesIn.data(),
						    modesIn.data(), nullptr,
						    numModesOut,
						    extentsOut, nullptr, modesOut,
						    typeData_, typeCompute_,
						    &descNet));

    // create optimiser info
    cutensornetContractionOptimizerConfig_t optimizerConfig;
    HANDLE_ERROR( cutensornetCreateContractionOptimizerConfig(handle_, &optimizerConfig) );
    cutensornetContractionOptimizerInfo_t optimizerInfo;
    HANDLE_ERROR( cutensornetCreateContractionOptimizerInfo(handle_, descNet, &optimizerInfo) );

    // define contraction path explicitly
    cutensornetContractionPath_t path;
    path.numContractions = (mps1.numQubits_ * 2) - 1;
    path.data = new cutensornetNodePair_t[path.numContractions];
    int endMPS1 = mps1.numQubits_ - 1;
    int endMPS2 = mps1.numQubits_ + mps2.numQubits_ - 1;
    path.data[0].first = endMPS1; // contract rightmost ends of MPS1 and MPS2
    path.data[0].second = endMPS2;
    int k = 1;
    for (int i = 0; i < (mps1.numQubits_ - 1); i++) {
	endMPS1 -= 1; // one tensor removed from MPS1
	endMPS2 -= 2; // one tensor removed from MPS1 and another from MPS2

	// contract result of last iteration with end of MPS2
	path.data[k].first = endMPS2;
	path.data[k].second = endMPS2 + 1;
	k++;

	// contract intermediate result with end of MPS1
	path.data[k].first = endMPS1;
	path.data[k].second = endMPS2;
	k++;
    }
    HANDLE_ERROR( cutensornetContractionOptimizerInfoSetAttribute(handle_,
								  optimizerInfo,
								  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH,
								  &path,
								  sizeof(path)) );
    
    HANDLE_ERROR( cutensornetContractionOptimize(handle_, descNet, optimizerConfig,
						 workspaceSize_, optimizerInfo) );
    int64_t numSlices = 0;
    HANDLE_ERROR( cutensornetContractionOptimizerInfoGetAttribute( handle_, optimizerInfo, CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES, &numSlices, sizeof(numSlices)) );
								  
    // create contraction plan
    cutensornetContractionPlan_t plan;
    HANDLE_ERROR( cutensornetCreateContractionPlan(handle_, descNet, optimizerInfo,
						   workDesc_, &plan) );
    
    // create slices
    cutensornetSliceGroup_t sliceGroup{};
    HANDLE_ERROR( cutensornetCreateSliceGroupFromIDRange(handle_, 0, numSlices, 1, &sliceGroup) );

    // perform actual contraction
    std::vector<void*> rawDataIn;
    for (int i = 0; i < mps1.numQubits_; i++) {
	rawDataIn.push_back(mps1.qubitTensor_[i]);
    }
    for (int i = 0; i < mps2.numQubits_; i++) {
	rawDataIn.push_back(mps2.qubitTensor_[i]);
    }
    
    HANDLE_ERROR(cutensornetContractSlices(handle_, plan, rawDataIn.data(),
					   resultGPU_, 0, workDesc_,
					   sliceGroup, stream_));

    // copy back result
    complex_t result;
    HANDLE_CUDA_ERROR(cudaMemcpy(&result, resultGPU_, sizeof(complex_t),
				 cudaMemcpyDeviceToHost));
    
    // free resources
    delete[] path.data;
    for (int i = 0; i < (mps1.numQubits_ * 2); i++) {
	delete[] extentsIn[i];
	delete[] modesIn[i];
	delete[] stridesIn[i];
    }
    HANDLE_ERROR(cutensornetDestroySliceGroup(sliceGroup));
    HANDLE_ERROR(cutensornetDestroyContractionPlan(plan));
    HANDLE_ERROR(cutensornetDestroyContractionOptimizerInfo(optimizerInfo));
    HANDLE_ERROR(cutensornetDestroyContractionOptimizerConfig(optimizerConfig));
    HANDLE_ERROR(cutensornetDestroyNetworkDescriptor(descNet));

    return result;
}

