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

int main(int argc, char* argv[])
{
    double truncation_error = 1e-16;
    
    if (argc != 4) {
	std::cerr << "Usage: build_matrix <mps X input file> <mps Y input file> <matrix output file>" << std::endl;
	return 1;
    }

    // read the input files
    char* mps_x_str = loadFile(argv[1]);
    if (mps_x_str == nullptr) {
	std::cerr << "Error loading MPS X input file" << std::endl;
	return 1;
    }
    
    char* mps_y_str = loadFile(argv[2]);
    if (mps_y_str == nullptr) {
	std::cerr << "Error loading MPS Y input file" << std::endl;
	return 1;
    }

    // extract number of MPSs and number of qubits from input files
    int num_mps_x, num_qubit_x;
    char *mps_x_ptr = readInt(mps_x_str, num_mps_x);
    if (!mps_x_ptr) {
	std::cerr << "Error parsing header of MPS X input file" << std::endl;
	return 1;
    }
    mps_x_ptr = readInt(mps_x_ptr, num_qubit_x);
    if (!mps_x_ptr) {
	std::cerr << "Error parsing header of MPS X input file" << std::endl;
	return 1;
    }

    int num_mps_y, num_qubit_y;
    char *mps_y_ptr = readInt(mps_y_str, num_mps_y);
    if (!mps_y_ptr) {
	std::cerr << "Error parsing header of MPS Y input file" << std::endl;
	return 1;
    }
    mps_y_ptr = readInt(mps_y_ptr, num_qubit_y);
    if (!mps_y_ptr) {
	std::cerr << "Error parsing header of MPS Y input file" << std::endl;
	return 1;
    }    
    
    // check that they match
    if ((num_mps_x != num_mps_y) || (num_qubit_x != num_qubit_y)) {
	std::cerr << "Dimension mismatch between MPS X and MPS Y" << std::endl;
	return 1;
    }
    std::cout << "Loading " << num_mps_x << " MPSs of " << num_qubit_x << " qubits" << std::endl;

    // create MPS object for each MPS loaded
    std::vector<MatrixProductState*> mps_x, mps_y;
    mps_x.reserve(num_mps_x);
    mps_y.reserve(num_mps_y);
    
    // load Y first in case there are 'None' items
    for (int i = 0; i < num_mps_y; i++) {
	MatrixProductState* mps = new MatrixProductState(num_qubit_y, 12, 1.0 - truncation_error);
	mps_y_ptr = mps->loadFromString(mps_y_ptr, false);
	if (mps_y_ptr == nullptr) {
	    std::cerr << "Error loading MPS " << i << " from MPS Y" << std::endl;
	    return 1;
	}
	mps_y.push_back(mps);
    }

    // load conjugate of X MPSs
    for (int i = 0; i < num_mps_x; i++) {
	MatrixProductState* mps = new MatrixProductState(num_qubit_x, 12, 1.0 - truncation_error);
	mps_x_ptr = mps->loadFromString(mps_x_ptr, true);
	if (mps_x_ptr == nullptr) {
	    std::cerr << "Error loading MPS " << i << " from MPS X" << std::endl;
	    return 1;
	}
	mps_x.push_back(mps);
    }
    
    delete[] mps_x_str;
    delete[] mps_y_str;

    // allocate storage for matrix
    double* matrix = new double[num_mps_x * num_mps_y];

    // compute matrix
    VdotCalculator vdc(CUDA_C_64F, CUTENSORNET_COMPUTE_64F);
    // FIXME: get ordering correct here
    for (int i = 0; i < num_mps_x; i++) {
	for (int j = 0; j < num_mps_y; j++) {
	    complex_t overlap = vdc.vdot(*mps_x[i], *mps_y[j]);
	    double kernel_entry = (overlap * std::conj(overlap)).real();
	    matrix[(j * num_mps_x) + i] = kernel_entry;
	}
    }

    // write matrix to disk
    ofstream of(argv[3]);
    if (!of.good()) {
	std::cerr << "Error opening output file" << std::endl;
	return 1;
    }
    of << std::setprecision(20);
    of << "[ ";
    // FIXME: get ordering correct here
    for (int i = 0; i < num_mps_x; i++) {
	of << "[ ";
	for (int j = 0; j < num_mps_y; j++) {
	    of << matrix[(j * num_mps_x) + i];
	    if (j < (num_mps_y - 1)) of << ", ";
	}
	of << " ]" << std::endl;
    }
    of << " ]" << std::endl;
    of.close();

    // free resources
    delete[] matrix;
    for (int i = 0; i < mps_x.size(); i++) {
	delete mps_x[i];
    }
    for (int i = 0; i < mps_y.size(); i++) {
	delete mps_y[i];
    }
    
    return 0;
}
