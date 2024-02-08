/**
 * @file
 * @brief File with macros and methods used by several OpenCL kernels and classes
 */

/*
 * hostKernelFunctions.h
 *
 *  Created on: 8 de nov. de 2017
 *      Author: fedsim
 */
#ifndef HOSTKERNELFUNCTIONS_H
#define HOSTKERNELFUNCTIONS_H

#include<OpenCLIPER/kernels/defs.h>

#ifdef __cplusplus
/// Macro used for disabling kernel-only code keyword global
#define global
/// Macro used for disabling kernel-only code keyword __global
#define __global
/// Macro for unsigned int data type name (different for kernel and non-kernel code)
#define uint cl_uint
#include <CL/cl.h>

extern "C" {
#else // Device code, C
// These functions are to be called from device code only
#define dimIndexType uint // dimIndexType defined as uint32_t only valid for C++ code
#endif // __CPLUSPLUS

//#define global

//#define DEBUG
//#define DEBUGKERNEL

#ifdef DEBUGKERNEL
    /// Macro for printing debugging information
    #define PRINTF(x) printf x
#else
    /// Macro for printing debugging information
    #define PRINTF(x) do {} while(0)
#endif // DEBUGKERNEL

// defines related to vector data types
// moved to defs.hpp because it used for checking validity of NDArray spatial dimensions
// (OpenCL vector operations are used for columns, rows or slices, so every spatial dimension has to be multiple of VECTORDATATYPESIZE/2, as
// every complex number is stored as 2 floats)
//
//#define VECTORDATATYPESIZE 16

/// Macro for defining the name of the vector type from its element data type and its number of elements
#define VECTORDATATYPEMACRO(baseType,size) {baseType ## size}
/// Data type used for vector kernel operations
#define VECTORDATATYPE float16
/// Half of the size of the vector data type defined previously
#define VECTORDATATYPEHALFSIZE (VECTORDATATYPESIZE)/2
/// Data type used for vector kernel operations on data with half size of the vector data type defined previously
#define HALFVECTORDATATYPE float8
/// OpenCL kernel operation for loading data from memory into a vector of the selected size
#define VLOADN vload16
/// OpenCL kernel operation for storing data into memory from a vector of the selected size
#define VSTOREN vstore16

/// Default kernel compile options
#define KERNELCOMPILEOPTS "-I../include/"

struct __attribute__((packed)) dimsAndStridesKnownFields {
    uint numSpatialDims;
    uint allSizesEqual;
    uint numCoils;
    uint numTemporalDims;
    uint firstTemporalDimSize;
};


// ---------------------------------------------------------------------------------------------------------------------------
// Internal functions
// ---------------------------------------------------------------------------------------------------------------------------

uint roundUp(uint numToRound, uint baseNumber);
uint getDimsAndStridesArrayOffsetInBytes(global const void* buffer);
global const uint* getDimsAndStridesArrayInBuffer(global const void* buffer);
global const struct dimsAndStridesKnownFields* getDimsAndStridesStruct(global const void* buffer);
global const uint* getStridesArray(global const void* buffer);


// ---------------------------------------------------------------------------------------------------------------------------
// Functions related to spatial dimensions
// ---------------------------------------------------------------------------------------------------------------------------

uint getNumSpatialDims(global const void* buffer);
uint getNumNDArrays(global const void* buffer);
uint getNDArray1DIndex(global const void* buffer, uint coilIndex, uint temporalDimIndexes[]);
uint getNDArrayTotalSize(global const void* buffer, uint NDArray1DIndex);
uint getNDArrayStride(global const void* buffer);
uint getSpatialDimStride(global const void* buffer, uint spatialDimIndex, uint NDArray1DIndex);
uint getSpatialDimSize(global const void* buffer, uint spatialDimIndex, uint NDArray1DIndex);


// ---------------------------------------------------------------------------------------------------------------------------
// Functions related to coils
// ---------------------------------------------------------------------------------------------------------------------------

uint getNumCoils(global const void* buffer);
uint getCoilDim(global const void* buffer);
uint getCoilStride(global const void* buffer, uint NDArray1DIndex);


// ---------------------------------------------------------------------------------------------------------------------------
// Functions related to temporal dimensions
// ---------------------------------------------------------------------------------------------------------------------------

uint getNumTemporalDims(global const void* buffer);
uint getTemporalDim(global const void* buffer, uint tempDim);
uint getTemporalDimStride(global const void* buffer, uint temporalDimIndex, uint NDArray1DIndex);
uint getTemporalDimSize(global const void* buffer, uint temporalDimIndex);


// ---------------------------------------------------------------------------------------------------------------------------
// Functions related to dimensions in general (not particular to spatial, coil or temporal dimensions)
// ---------------------------------------------------------------------------------------------------------------------------

uint getDimSize(global const void* buffer, uint dimIndex, uint NDArray1DIndex);
uint getElementStride(global const void* buffer, uint NDArray1DIndex);
uint getDimStride(global const void* buffer, uint dimIndex, uint NDArray1DIndex);
global const void* getNextElement(global const void* buffer, dimIndexType nDim, dimIndexType curNDArray, dimIndexType curOffset);


// ---------------------------------------------------------------------------------------------------------------------------
// These functions are to be called from device code only
// ---------------------------------------------------------------------------------------------------------------------------
#ifdef __OPENCL_C_VERSION
    void printComplex(float2 complex, char* name);
    void printVector(__constant char name[], float* vector, uint numberOfElements);
#endif //__OPENCL_C_VERSION

#ifdef __cplusplus
}  //extern "C"
#undef global
#undef __global
#undef uint
#endif // __cplusplus

#endif // HOSTKERNELFUNCTIONS_H
