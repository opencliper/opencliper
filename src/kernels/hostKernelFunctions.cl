/* Copyright (C) 2018 Federico Simmross Wattenberg,
 *                    Manuel Rodríguez Cayetano,
 *                    Javier Royuela del Val,
 *                    Elena Martín González,
 *                    Elisa Moya Sáez,
 *                    Marcos Martín Fernández and
 *                    Carlos Alberola López
 *
 * This file is part of OpenCLIPER.
 *
 * OpenCLIPER is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; version 3 of the License.
 *
 * OpenCLIPER is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with OpenCLIPER; If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *  Contact:
 *
 *  Federico Simmross Wattenberg
 *  E.T.S.I. Telecomunicación
 *  Universidad de Valladolid
 *  Paseo de Belén 15
 *  47011 Valladolid, Spain.
 *  fedsim@tel.uva.es
 */
#include <OpenCLIPER/kernels/hostKernelFunctions.h>

#ifdef __cplusplus
    #define global
    #define __global
    #define uint cl_uint
    extern "C" {
#endif  // __cplusplus

// ---------------------------------------------------------------------------------------------------------------------------
// Internal functions
// ---------------------------------------------------------------------------------------------------------------------------

/**
 * @brief Round up numToRound to multiple of baseNumber
 * @param[in] numToRound positive number to be rounded
 * @param[in] baseNumber value whose multiple nearest to numToRound is returned
 * @return rounded number (nearest multiple of baseNumber)
 */
uint roundUp(uint numToRound, uint baseNumber) {
    //assert(baseNumber);
    uint remainder = numToRound % baseNumber;
    PRINTF(("remainder: %d\n", remainder));
    if(remainder == 0) {
	return numToRound;
    }
    else {
	return numToRound + baseNumber - remainder;
    }
}

/**
 * @brief Return the offset (in bytes) from cl buffer start to first position of dimensions and strides array
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @return offset in bytes
 */
uint getDimsAndStridesArrayOffsetInBytes(global const void* buffer) {
    // first uint element of input buffer before data start position is offset in bytes for dimensions and strides vector
    uint dataDimsAndStridesArrayOffset = *((global uint*) buffer - 1);
    return dataDimsAndStridesArrayOffset;
}

/**
 * @brief Return a uint pointer to the array of data dimensions and strides stored inside an OpenCL buffer
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @return pointer to the array of dimensions and strides
 */
global const uint* getDimsAndStridesArrayInBuffer(global const void* buffer) {
    uint dimsAndStridesArrayOffsetInBytes = getDimsAndStridesArrayOffsetInBytes(buffer);
    global const uint* dimsAndStridesArray;
    dimsAndStridesArray = ((global const uint*) buffer) - (dimsAndStridesArrayOffsetInBytes / sizeof(uint));
    return dimsAndStridesArray;
}

/**
 * @brief Return a pointer to the struct of data dimensions and strides stored inside an OpenCL buffer
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @return pointer to the array of dimensions and strides
 */
global const struct dimsAndStridesKnownFields* getDimsAndStridesStruct(global const void* buffer) {
    return (global const struct dimsAndStridesKnownFields*) getDimsAndStridesArrayInBuffer(buffer);
}

/**
 * @brief returns a pointer to the start of strides array (included in dimsAndStridesInfo)
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @return the pointer to the start of the strides information of the dimsAndStridesInfo array
 */
global const uint* getStridesArray(global const void* buffer) {
    global const struct dimsAndStridesKnownFields* pDimsAndStridesStruct = getDimsAndStridesStruct(buffer);
    uint numOfNDArrays = getNumNDArrays(buffer);
    uint numOfTempDims = getNumTemporalDims(buffer);
    uint dimsArrayNumOfPos = NUMINITIALPOSITIONSDIMSINFO + numOfTempDims;
    if(pDimsAndStridesStruct->allSizesEqual == 1) {
	dimsArrayNumOfPos += getNumSpatialDims(buffer);
    }
    else {
	dimsArrayNumOfPos += getNumSpatialDims(buffer) * numOfNDArrays;
    }
    return ((global const uint*) pDimsAndStridesStruct) + dimsArrayNumOfPos;
}


// ---------------------------------------------------------------------------------------------------------------------------
// Functions related to spatial dimensions
// ---------------------------------------------------------------------------------------------------------------------------

/**
 * @brief Returns the number of spatial dimensions stored in dimensions info array
 *
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @return the number of spatial dimensions in the data object
 */
uint getNumSpatialDims(global const void* buffer) {
    return getDimsAndStridesStruct(buffer)->numSpatialDims;
}

/**
 * @brief Returns the number of NDArrays stored a given buffer contains
 *
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @return the number of NDArrays in the data object
 */
uint getNumNDArrays(global const void* buffer) {
    uint numOfNDArrays = 1;
    uint numCoils = getNumCoils(buffer);
    if(numCoils > 0) {
	numOfNDArrays *= numCoils;
    }
    uint numSpatialDims = getNumTemporalDims(buffer);
    if(numSpatialDims > 0) {
	for(uint i = 0; i < numSpatialDims; i++) {
	    numOfNDArrays *= getTemporalDimSize(buffer, i);
	}
    }
    return numOfNDArrays;
}

/**
 * Calculates a 1-dimensional index for accessing a specific NDArray from the vector of NDArrays corresponding to the specified coil
 * index and the indexes for the temporal dimensions
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @param[in] coilIndex index for the coils
 * @param[in] temporalDimIndexes vector of indexes for the selected frame
 * @return the 1-dimensional index for a NDArray
 */
uint getNDArray1DIndex(global const void* buffer, uint coilIndex, uint temporalDimIndexes[]) {
    global const struct dimsAndStridesKnownFields* pDimsAndStridesStruct = getDimsAndStridesStruct(buffer);
    uint numCoils = pDimsAndStridesStruct->numCoils;
    // coilIndex value 0 can be used to ignore coil dimensions (if numCoils is 0)
    // but coilIndex must be < numCoils if numCoils != 0
    if((numCoils != 0) && (coilIndex >= numCoils)) {
	// Invalid coil index, returns invalid NDArray 1D-index,
	return -1;
    }
    uint numTemporalDims = pDimsAndStridesStruct->numTemporalDims;
    for(uint temporalDimId = 0; temporalDimId < numTemporalDims; temporalDimId++) {
	if(temporalDimIndexes[temporalDimId] >= getTemporalDimSize(buffer, temporalDimId)) {
	    // Invalid temporal dim index for dimension number temporalDimId
	    return -1;
	}
    }
    uint index = 0, stride = 1;
    for(uint i = 0; i < numTemporalDims; i++) {
	index += temporalDimIndexes[i] * stride;
	stride *= (&pDimsAndStridesStruct->firstTemporalDimSize)[i];
    }
    if(numCoils != 0) {
	index = coilIndex + (numCoils * index);
    }
    return index;
}

/**
 * @brief Returns the size (product of sizes of spatial dimensions) of a specific NDArray (can be different for every NDArray)
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @param[in] NDArray1DIndex 1D-index for the NDArray
 * @return the size of the selected NDArray
 */
uint getNDArrayTotalSize(global const void* buffer, uint NDArray1DIndex) {
    uint numSpatialDims = getNumSpatialDims(buffer);
    uint acum = 1;
    for(uint spatialDimPos = 0; spatialDimPos < numSpatialDims; spatialDimPos ++) {
	acum *= getSpatialDimSize(buffer, spatialDimPos, NDArray1DIndex);
    }
    return acum;
}

/**
 * @brief ...
 *
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @return cl_uint
 */
uint getNDArrayStride(global const void* buffer) {
    global const struct dimsAndStridesKnownFields* pDimsAndStridesStruct = getDimsAndStridesStruct(buffer);
    if(pDimsAndStridesStruct->allSizesEqual == 1) {  // All NDArrays have the same size, only 1 group of strides valid for every NDArray
	return 0;
    }
    else {
	uint NDArrayStride;
	NDArrayStride = pDimsAndStridesStruct->numSpatialDims + pDimsAndStridesStruct->numTemporalDims;
	if(pDimsAndStridesStruct->numCoils != 0) {
	    NDArrayStride += 1;
	}
	return NDArrayStride;
    }
}

/**
 * @brief Returns the spatial stride (number of real elements between the first data element associated to two consecutive cols, rows, slices, etc.)
 * for a specific NDArray
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @param[in] spatialDimIndex the spatial dimension to be get (enum knownSpatialDimPos values can be used for
 * this parameter)
 * @param[in] NDArray1DIndex 1D-index for the NDArray
 * @return the spatial stride for the specific spatial dimension and NDArray
 */
uint getSpatialDimStride(global const void* buffer, uint spatialDimIndex, uint NDArray1DIndex) {
    uint spatialDimStridePos = FirstSpatialStridePos + spatialDimIndex + getNDArrayStride(buffer) * NDArray1DIndex;
    return getStridesArray(buffer)[spatialDimStridePos];
}

/**
 * @brief Returns the size of a specific spatial dimension of a NDArray element (can be different for every NDArray)
 *
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @param[in] spatialDimIndex the spatial dimension to be get (enum knownSpatialDimPos values can be used for
 * this parameter)
 * @param[in] NDArray1DIndex 1D-index for the NDArray
 * @return the size of the selected spatial dimension and NDArray
 */
uint getSpatialDimSize(global const void* buffer, uint spatialDimIndex, uint NDArray1DIndex) {
    global const struct dimsAndStridesKnownFields* pDimsAndStridesStruct = getDimsAndStridesStruct(buffer);
    uint numSpatialDims = pDimsAndStridesStruct->numSpatialDims;
    if(spatialDimIndex >= numSpatialDims) {  // XData
	// invalid spatialDimPos, return 0 as size for it
	return 0;
    }
    uint numTemporalDims = pDimsAndStridesStruct->numTemporalDims;
    uint spatialDimSizePos = FirstTemporalDimPos + numTemporalDims + spatialDimIndex;
    if(pDimsAndStridesStruct->allSizesEqual == 0) {
	spatialDimSizePos += numSpatialDims * NDArray1DIndex;
    }
    return ((global const uint*) pDimsAndStridesStruct)[spatialDimSizePos];
}


// ---------------------------------------------------------------------------------------------------------------------------
// Functions related to coils
// ---------------------------------------------------------------------------------------------------------------------------

/**
 * @brief Returns the number of coils stored in dimensions info array
 *
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @return the number of coils in the data object
 */
uint getNumCoils(global const void* buffer) {
    return getDimsAndStridesStruct(buffer)->numCoils;
}

/**
 * @brief Returns the dimension which holds the different coils in a given temporal moment, or -1 if there are no coils
 *
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @return the number of the coil dimension
 */
uint getCoilDim(global const void* buffer) {
    if(getNumCoils(buffer) != 0)
	return getNumSpatialDims(buffer);
    else
	return -1;
}

/**
 * @brief Returns the coil stride (number of real NDArray elements between the first NDArrays associated to two consecutive coils)
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @param[in] NDArray1DIndex 1D-index for the NDArray
 * @return the coil stride for the specific NDArray
 */
uint getCoilStride(global const void* buffer, uint NDArray1DIndex) {
    if(getNumCoils(buffer) == 0) {
	// if number of coills is 0, coil stride does not exist, method returns 0
	return 0;
    }
    uint numSpatialDims = getNumSpatialDims(buffer);
    uint coilStridePos = numSpatialDims + getNDArrayStride(buffer) * NDArray1DIndex;
    return getStridesArray(buffer)[coilStridePos];
}


// ---------------------------------------------------------------------------------------------------------------------------
// Functions related to temporal dimensions
// ---------------------------------------------------------------------------------------------------------------------------

/**
 * @brief Returns the number of temporal dimensions stored in dimensions info array
 *
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @return the number of temporal dimensions in the data object
 */
uint getNumTemporalDims(global const void* buffer) {
    return getDimsAndStridesStruct(buffer)->numTemporalDims;
}

/**
 * @brief Returns the linear dimension corresponding to the tempDim-th temporal dimension, or -1 if there are no temporal dimensions
 *
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @return the linear dimension number of the given temporal dimension
 */
uint getTemporalDim(global const void* buffer, uint tempDim) {
    if(getNumTemporalDims(buffer) != 0)
	return getNumSpatialDims(buffer) + ((getNumCoils(buffer) != 0)? 1:0) + tempDim;
    else
	return -1;
}

/**
 * @brief Returns the frame stride (number of real NDArray elements between the first NDArrays associated to two consecutive frames)
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @param[in] temporalDimIndex index of the selected frame dimension
 * @param[in] NDArray1DIndex 1D-index for the NDArray
 * @return the frame stride for the specific frame dimension and NDArray
 */
uint getTemporalDimStride(global const void* buffer, uint temporalDimIndex, uint NDArray1DIndex) {
    global const struct dimsAndStridesKnownFields* pDimsAndStridesStruct = getDimsAndStridesStruct(buffer);
    uint firstTemporalStridePos;
    uint numCoils = pDimsAndStridesStruct->numCoils;
    uint numSpatialDims = pDimsAndStridesStruct->numSpatialDims;
    uint numTemporalDims = pDimsAndStridesStruct->numTemporalDims;
    // Invalid temporalDimIndex, return 0 as stride for it
    if(temporalDimIndex >= numTemporalDims) {
	return 0;
    }
    firstTemporalStridePos = FirstSpatialStridePos + numSpatialDims;
    if(numCoils != 0) {
	firstTemporalStridePos++; // first temporal stride is stored after coil stride
    }
    uint temporalStridePosition = firstTemporalStridePos + temporalDimIndex + getNDArrayStride(buffer) * NDArray1DIndex;
    return getStridesArray(buffer)[temporalStridePosition];
}

/**
 * @brief Returns the size of a temporal dimension stored in dimensions info array
 *
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @param[in] temporalDimIndex index of the temporal dimension
 * @return size of the specified temporal dimension
 */
uint getTemporalDimSize(global const void* buffer, uint temporalDimIndex) {
    uint numTemporalDims = getNumTemporalDims(buffer);
    if(temporalDimIndex >= numTemporalDims) {
	// invalid frameDimPos, return 0 as size for it
	return 0;
    }
    uint temporalDimSizePos = FirstTemporalDimPos + temporalDimIndex;
    return getDimsAndStridesArrayInBuffer(buffer)[temporalDimSizePos];
}


// ---------------------------------------------------------------------------------------------------------------------------
// Functions related to dimensions in general (not particular to spatial, coil or temporal dimensions)
// ---------------------------------------------------------------------------------------------------------------------------

/**
 * @brief Returns the size of a specific dimension (be it spatial, coil or temporal) of a NDArray element (can be different for every NDArray)
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @param[in] dimIndex the dimension to get, spatial dimensions first, then coils, then temporal dimensions
 * @param[in] NDArray1DIndex 1D-index of the NDArray
 * @return the size of the selected dimension in the chosen NDArray
 */
uint getDimSize(global const void* buffer, uint dimIndex, uint NDArray1DIndex) {
    uint numSpatialDims = getNumSpatialDims(buffer);
    if(dimIndex < numSpatialDims)
	return getSpatialDimSize(buffer, dimIndex, NDArray1DIndex);
    else {
	dimIndex -= numSpatialDims;
	uint nCoils = getNumCoils(buffer);
	if(nCoils) {
	    if(dimIndex == 0)
		return nCoils;
	    else
		dimIndex--;
	}
	if(dimIndex < getNumTemporalDims(buffer))
	    return getTemporalDimSize(buffer, dimIndex);
	else
	    return 0; // invalid dimIndex, return 0 as size for it
    }
}

/**
 * @brief Returns the stride for the first element of a NDArray object
 *
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @param[in] NDArray1DIndex 1-dimensional index for selected NDArray (obtained from temporal dimensions indexes and coil index)
 * @return the stride for the first element of a NDArray object
 */
uint getElementStride(global const void* buffer, uint NDArray1DIndex) {
    uint elementStridePos = ElementStridePos + getNDArrayStride(buffer) * NDArray1DIndex;
    return getStridesArray(buffer)[elementStridePos];
}

/**
 * @brief Returns the stride (number of real elements between the first data element associated with two consecutive cols, rows, slices, coils (if any), frames, etc.)
 * for a specific NDArray
 * @param[in] buffer OpenCL buffer storing data and their dimensions and strides array
 * @param[in] dimIndex the spatial dimension to be get (enum knownSpatialDimPos values can be used for
 * this parameter)
 * @param[in] NDArray1DIndex 1D-index for the NDArray
 * @return the spatial stride for the specific spatial dimension and NDArray
 */
uint getDimStride(global const void* buffer, uint dimIndex, uint NDArray1DIndex) {
    global const struct dimsAndStridesKnownFields* pDimsAndStridesStruct = getDimsAndStridesStruct(buffer);
    uint numSpatialDims = pDimsAndStridesStruct->numSpatialDims;
    if(dimIndex < numSpatialDims)
	return getSpatialDimStride(buffer, dimIndex, NDArray1DIndex);
    else {
	dimIndex -= numSpatialDims;
	uint nCoils = getNumCoils(buffer);
	if(nCoils) {
	    if(dimIndex == 0)
		return getCoilStride(buffer, NDArray1DIndex);
	    else
		dimIndex--;
	}
	if(dimIndex < pDimsAndStridesStruct->numTemporalDims)
	    return getTemporalDimStride(buffer, dimIndex, NDArray1DIndex);
	else
	    return 0; // invalid dimIndex, return 0 as stride for it
    }
}

global const void* getNextElement(global const void* buffer, dimIndexType dimIndex, dimIndexType curNDArray, dimIndexType curOffset) {
    global const char* charBuffer = (global const char*) buffer;
    dimIndexType stride = getDimStride(buffer, dimIndex, curNDArray);
    return charBuffer + curOffset + stride;
}


// ---------------------------------------------------------------------------------------------------------------------------
// These functions are to be called from device code only
// ---------------------------------------------------------------------------------------------------------------------------
#ifdef __OPENCL_C_VERSION
void printComplex(float2 complex, char* name) {
    printf("%s: %f + %f j\n", name, complex.s0, complex.s1);
}

void printVector(__constant char name[], float* vector, uint numberOfElements) {
    printf("%s printVector starts...\n", name);
    printf("Number of elements: %u\n", numberOfElements);
    for(uint i = 0; i < numberOfElements - 1; i++) {
	//printf("i: %u\n", i);
	printf("%f, ", vector[i]);
    }
    printf("%f)\n", vector[numberOfElements - 1]);
    printf("printVector ends\n");
}
#endif // __OPENCL_C_VERSION

#ifdef __cplusplus
}  //extern "C"
#endif // __cplusplus
