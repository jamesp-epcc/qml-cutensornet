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

VdotCalculator::VdotCalculator(cudaDataType_t typeData, cutensornetComputeType_t typeCompute, int numQubits, int physExtent)
{
    typeData_ = typeData;
    typeCompute_ = typeCompute;
    numQubits_ = numQubits;
    physExtent_ = physExtent;

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

    // create optimiser config
    HANDLE_ERROR( cutensornetCreateContractionOptimizerConfig(handle_, &optimizerConfig_) );

    // initialise the size/mode/stride arrays
    extentsOut_[0] = 1;
    modesOut_[0] = 1;

    int pmodeBase = 2;
    int vmode1Base = pmodeBase + numQubits_;
    int vmode2Base = vmode1Base + numQubits_ + 1;
    
    // mps1
    for (int i = 0; i < numQubits_; i++) {
	rawDataIn_.push_back(nullptr);

	numModesIn_.push_back(4);

	// extents and strides depend on the individual MPS's dimensions
	// so just allocate space now
	extentsIn_.push_back(new int64_t[4]);
	stridesIn_.push_back(new int64_t[4]);

	int32_t* modes = new int32_t[4];
	modes[0] = vmode1Base + i;
	modes[1] = vmode1Base + i + 1;
	modes[2] = pmodeBase + i;
	modes[3] = modesOut_[0];
	modesIn_.push_back(modes);
    }

    // mps2
    for (int i = 0; i < numQubits_; i++) {
	rawDataIn_.push_back(nullptr);

	numModesIn_.push_back(4);

	// extents and strides depend on the individual MPS's dimensions
	// so just allocate space now
	extentsIn_.push_back(new int64_t[4]);
	stridesIn_.push_back(new int64_t[4]);

	int32_t* modes = new int32_t[4];
	modes[0] = vmode2Base + i;
	modes[1] = vmode2Base + i + 1;
	modes[2] = pmodeBase + i;
	modes[3] = modesOut_[0];
	modesIn_.push_back(modes);
    }
}

VdotCalculator::~VdotCalculator()
{
    for (int i = 0; i < (numQubits_ * 2); i++) {
	delete[] extentsIn_[i];
	delete[] modesIn_[i];
	delete[] stridesIn_[i];
    }

    cutensornetDestroyContractionOptimizerConfig(optimizerConfig_);
    cutensornetDestroyWorkspaceDescriptor(workDesc_);
    cutensornetDestroy(handle_);

    cudaFree(workspace_);
}

void VdotCalculator::vdot(MatrixProductState& mps1, MatrixProductState& mps2, complex_t* result)
{
    // FIXME: promote a lot of the locals here to class members so they don't
    // have to be allocated each time
    // sanity check
    if ((mps1.numQubits_ != numQubits_) ||
	(mps2.numQubits_ != numQubits_) ||
	(mps1.physExtent_ != physExtent_) ||
	(mps2.physExtent_ != physExtent_)) {
	std::cerr << "For vdot, both MPS must have same number of qubits and physical extent!" << std::endl;
	return;
    }

    // mps1 tensors
    for (int i = 0; i < numQubits_; i++) {
	int64_t* extents = extentsIn_[i];
	extents[0] = mps1.extentsPerQubit_[i];
	extents[1] = mps1.extentsPerQubit_[i+1];
	extents[2] = mps1.physExtent_;
	extents[3] = 1;

	int64_t* strides = stridesIn_[i];
	strides[3] = 1;
	strides[2] = 1;
	strides[1] = extents[2] * strides[2];
	strides[0] = extents[1] * strides[1];
    }

    // mps 2 tensors
    for (int i = 0; i < numQubits_; i++) {
	int64_t* extents = extentsIn_[numQubits_ + i];
	extents[0] = mps2.extentsPerQubit_[i];
	extents[1] = mps2.extentsPerQubit_[i+1];
	extents[2] = mps2.physExtent_;
	extents[3] = 1;

	int64_t* strides = stridesIn_[numQubits_ + i];
	strides[3] = 1;
	strides[2] = 1;
	strides[1] = extents[2] * strides[2];
	strides[0] = extents[1] * strides[1];
    }
    
    cutensornetNetworkDescriptor_t descNet;
    HANDLE_ERROR(cutensornetCreateNetworkDescriptor(handle_, numQubits_*2,
						    numModesIn_.data(),
						    extentsIn_.data(),
						    stridesIn_.data(),
						    modesIn_.data(), nullptr, 1,
						    extentsOut_, nullptr, modesOut_,
						    typeData_, typeCompute_,
						    &descNet));

    // create optimiser info
    // re-use optimiser config, which never changes
    cutensornetContractionOptimizerInfo_t optimizerInfo;
    HANDLE_ERROR( cutensornetCreateContractionOptimizerInfo(handle_, descNet, &optimizerInfo) );
    // leave contraction path implicit, unlike in Pytket. It's slightly faster
    
    HANDLE_ERROR( cutensornetContractionOptimize(handle_, descNet, optimizerConfig_,
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
    for (int i = 0; i < numQubits_; i++) {
	rawDataIn_[i] = mps1.qubitTensor_[i];
	rawDataIn_[i + numQubits_] = mps2.qubitTensor_[i];
    }
    
    HANDLE_ERROR(cutensornetContractSlices(handle_, plan, rawDataIn_.data(),
					   (void*)result, 0, workDesc_,
					   sliceGroup, stream_));

    // free resources
    HANDLE_ERROR(cutensornetDestroySliceGroup(sliceGroup));
    HANDLE_ERROR(cutensornetDestroyContractionPlan(plan));
    HANDLE_ERROR(cutensornetDestroyContractionOptimizerInfo(optimizerInfo));
    HANDLE_ERROR(cutensornetDestroyNetworkDescriptor(descNet));
}

