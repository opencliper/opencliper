/* Copyright (C) 2018 Federico Simmross Wattenberg,
 *                    Manuel Rodríguez Cayetano,
 *                    Javier Royuela del Val,
 *                    Elena Martín González,
 *                    Elisa Moya Sáez,
 *                    Marcos Martín Fernández and
 *                    Carlos Alberola López
 *
 * This file is part of OpenCLIPER
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
#include <OpenCLIPER/MatVarInfo.hpp>
#include <OpenCLIPER/hostKernelFunctions.hpp>
#include <OpenCLIPER/NDArray.hpp>
#include <LPISupport/Utils.hpp>

// Uncomment to show class-specific debug messages
//#define MATVARINFO_DEBUG

#if !defined NDEBUG && defined MATVARINFO_DEBUG
    #define MATVARINFO_CERR(x) CERR(x)
#else
    #define MATVARINFO_CERR(x)
    #undef MATVARINFO_DEBUG
#endif

namespace OpenCLIPER {
/**
 * @brief Default constructor, sets all the class variables with values passed as method parameters
 *
 * @param[in] class_type matlab class type
 * @param[in] data_type matlab data type
 * @param[in] data array of data in matlab format
 * @param[in] opt matlab options
 */
MatVarInfo::MatVarInfo(enum matio_classes class_type, enum matio_types data_type, void* data, int opt) {
    commonInit(class_type, data_type, data, opt);
}

void MatVarInfo::commonInit(enum matio_classes class_type, enum matio_types data_type, void* data, int opt) {
    this->class_type = class_type;
    this->data_type = data_type;
    this->data = data;
    this->opt = opt;
}
/**
 * @brief Creates a MatVarInfo object containing an fixed size array of elements of an specific data type.
 *
 * @param[in] elementDataType data type of array elements
 * @param[in] numOfElements number of array elements
 * @return a pointer to a MatVarInfo object
 */
MatVarInfo::MatVarInfo(ElementDataType elementDataType, dimIndexType numOfElements) {
    this->elementDataType = elementDataType;
    opt = 0;
    if(elementDataType == TYPEID_COMPLEX) {
	realPart = (realType*) new realType[numOfElements];
	imagPart = (realType*) new realType[numOfElements];
	opt = MAT_F_COMPLEX;
	data = (mat_complex_split_t*) new mat_complex_split_t({realPart, imagPart});
	if(TYPEID_REAL == std::type_index(typeid(float))) {
	    class_type = MAT_C_SINGLE;
	    data_type = MAT_T_SINGLE;
	}
	else if(TYPEID_REAL == std::type_index(typeid(double))) {
	    class_type = MAT_C_DOUBLE;
	    data_type = MAT_T_DOUBLE;
	}
	else {
	    BTTHROW(std::invalid_argument("Data type not supported for complex data"), "MatVarInfo::MatVarInfo");
	}
    }
    else if(elementDataType == TYPEID_INDEX) {
	opt = 0;
	data = (dimIndexType*) new dimIndexType[numOfElements];
	if(elementDataType == std::type_index(typeid(unsigned int))) {
	    class_type = MAT_C_UINT32;
	    data_type = MAT_T_UINT32;
	}
	else if(elementDataType == std::type_index(typeid(unsigned long))) {
	    class_type = MAT_C_UINT64;
	    data_type = MAT_T_UINT64;
	}
	else {
	    BTTHROW(std::invalid_argument("Unsupported data type for index type"), "MatVarInfo::MatVarInfo");
	}
    }
    else if(elementDataType == TYPEID_REAL) {
	opt = 0;
	data = (realType*) new realType[numOfElements];
	if(TYPEID_REAL == std::type_index(typeid(float))) {
	    class_type = MAT_C_SINGLE;
	    data_type = MAT_T_SINGLE;
	}
	else if(TYPEID_REAL == std::type_index(typeid(double))) {
	    class_type = MAT_C_DOUBLE;
	    data_type = MAT_T_DOUBLE;
	}
	else {
	    BTTHROW(std::invalid_argument("Data type not supported for real type data"), "MatVarInfo::MatVarInfo");
	}
    }
    else if(elementDataType == TYPEID_CL_UCHAR) {
	opt = 0;
	data = (cl_uchar*) new cl_uchar[numOfElements];
	class_type = MAT_C_UINT8;
	data_type = MAT_T_UINT8;
    }
    else {
	BTTHROW(std::invalid_argument("Data type not supported"), "MatVarInfo::MatVarInfo");
    }
}

/**
 * @brief Stores the data contained in this NDArray into a matlab variable.
 *
 * @param[in] pMatVarInfo pointer to matlab array variable
 * @param[in] pDimsAndStridesInfo pointer to array with spatial and temporal dimensions (and their strides) info
 * @param[in] nDArrayOffsetInElements offset (in number of elements) from NDArray data array beginning to start reading from
 * @param[in] pNDArrayData generic pointer to NDArray data start
 */
void MatVarInfo::set(const dimIndexType* pDimsAndStridesInfo, dimIndexType nDArrayOffsetInElements, const NDArray* pNDArray,
	const void* pNDArrayData) {
    // First dimension in matlab is number of rows (height) and in OpenCLIPER is width (number of columns)
    // Number of dimensions in matlab (rank) is always >= 2 (a scalar has dimensions 1x1, a vector, 1xN)
    dimIndexType numRows = 1, numColumns = 1, numSlicesAndBeyond = 1;
    numRows = NDARRAYHEIGHT(pNDArray);
    numColumns = NDARRAYWIDTH(pNDArray);
    // The rest of dimensions are in the same order in matlab and in OpenCLIPER code, we process them as a unique dimension
    // (this way a number of spatial dimensions greater than 3 is supported)

    const std::vector<dimIndexType>* pDims = pNDArray->getDims();
    for(dimIndexType i = 2; i < pDims->size(); i++) {
	numSlicesAndBeyond *= pDims->at(i);
    }

    if(matlabStrides.size() != 0) {
	matlabStrides.erase(matlabStrides.begin(), matlabStrides.end());
    }

    // If number of spatial dimensions is 1 but second matlab dimension is > 1, the stride
    // for columns is the number of columns (matlab arrays are stored by rows)
    matlabStrides[matlabStridesKeys::row] = 1;
    matlabStrides[matlabStridesKeys::column] = matlabStrides[matlabStridesKeys::row] * numRows;
    matlabStrides[matlabStridesKeys::sliceAndBeyond] = matlabStrides[matlabStridesKeys::column] * numColumns;

    dimIndexType matlabElementOffset, nDArrayElementOffset;

    for(dimIndexType sliceAndBeyond = 0;  sliceAndBeyond < numSlicesAndBeyond; sliceAndBeyond ++) {
	for(dimIndexType row = 0; row < numRows; row++) {
	    for(dimIndexType column = 0; column < numColumns; column++) {
		// Matlab offset must include the nDArrayOffsetInElements to
		// store data of different NDArrays inside the same matlab data array
		// (all the NDArrays data of the same Data object are stored in an unique data array for non-complex data, 2 arrays
		// for complex data).
		matlabElementOffset = getMatlabStride(matlabStridesKeys::sliceAndBeyond) * sliceAndBeyond +
				      getMatlabStride(matlabStridesKeys::column) * column +
				      getMatlabStride(matlabStridesKeys::row) * row + nDArrayOffsetInElements;
		// nDArrayElementOffset for a NDArray data must not include nDArrayOffsetInElements as
		// data access for NDArray is based on a pointer specific to its OpenCL buffer or image (Data device contiguous
		// memory is not used for this access)
		nDArrayElementOffset = getSpatialDimStride(pDimsAndStridesInfo, knownSpatialDimPos::COLUMNS, 0) * column +
				       getSpatialDimStride(pDimsAndStridesInfo, knownSpatialDimPos::ROWS, 0) * row +
				       getSpatialDimStride(pDimsAndStridesInfo, knownSpatialDimPos::SLICES, 0) * sliceAndBeyond;
		MATVARINFO_CERR("row: " << row << "\tcolumn: " << column << std::endl);
		MATVARINFO_CERR("matvar->data[" << row << ", " << column << "]: ");
		dumpMatlabElement(matlabElementOffset, nDArrayElementOffset, pNDArrayData);
	    }
	}
    }
}

void MatVarInfo::dumpMatlabElement(dimIndexType matlabElementOffset, dimIndexType nDArrayElementOffset, const void* pNDArrayData) {
    if(elementDataType == TYPEID_COMPLEX) {
	mat_complex_split_t* complexData = (mat_complex_split_t*) data;

	// WARNING!! pointer arithmetic depends on base type (pointer + offset operation increments pointer in a number of bytes equal
	// to offset multiplied by the base data type size in bytes, you must not explicitly multiply offset by base data type size!!)
	realType* realPart = ((realType*) complexData->Re) + matlabElementOffset;
	realType* imagPart = ((realType*) complexData->Im) + matlabElementOffset;

	// type of base element (real, not complexType)
	realType* source = (realType*) pNDArrayData;

	// real and imaginary parts in OpenCL buffer/images stored in contiguous positions (nDArrayElementOffset*2: real part,
	// (nDArrayElementOffset*2) + 1: imaginary part, nDArrayElementOffset for elements of complexType => double offset to
	// accessing their real elements components

	*realPart = source[nDArrayElementOffset * 2];
	*imagPart = source[(nDArrayElementOffset * 2) + 1];
    }
    else if(elementDataType == TYPEID_INDEX) {
	dimIndexType* typedData = ((dimIndexType*) data) + matlabElementOffset;
	dimIndexType* source = (dimIndexType*) pNDArrayData;
	*typedData = (source)[nDArrayElementOffset];
    }
    else if(elementDataType == TYPEID_REAL) {
	realType* typedData = ((realType*) data) + matlabElementOffset;
	realType* source = (realType*) pNDArrayData;
	*typedData = (source)[nDArrayElementOffset];
    }
    else if(elementDataType == TYPEID_CL_UCHAR) {
	cl_uchar* typedData = ((cl_uchar *) data) + matlabElementOffset;
	cl_uchar* source = (cl_uchar *) pNDArrayData;
	*typedData = (source)[nDArrayElementOffset];
    }
    else {
	BTTHROW(std::invalid_argument("Data type not supported"), "MatVarInfo::dumpMatlabElement");
    }
}

/**
 * @brief Updates dimensions and rank class variables from a vector of dimensions
 *
 * @param[in] pDimsVectorArg pointer to vector of data dimensions
 * @param[in] swapColsAndRows true if columns and rows must be swapped (matlab arrays are stored by columns, C/C++ convention is store them by rows)
 */
void MatVarInfo::updateDimsAndRank(const std::vector<dimIndexType>* pDimsVectorArg) {
    // pDimsVector field still empty, must guarantee minimum rank of 2
    // and swap first and second dimensions, if both exist
    if(pDimsVector->size() == 0) {
	// data with 1 spatial dimension, minimum matlab rank is 2 (vector is an array of 1xN dimensions)
	if(pDimsVectorArg->size() == 1) {
	    pDimsVector->push_back(1);
	    pDimsVector->push_back(pDimsVectorArg->at(WIDTHPOS));
	}
	else {   // 2 or more spatial dimensions, first 2 swapped (matlab stores data by columns, OpenCLIPER by rows)
	    pDimsVector->push_back(pDimsVectorArg->at(HEIGHTPOS));
	    pDimsVector->push_back(pDimsVectorArg->at(WIDTHPOS));
	}
	// append third and following dimensions
	for(dimIndexType i = 2; i < pDimsVectorArg->size(); i++) {
	    pDimsVector->push_back(pDimsVectorArg->at(i));
	}
    }
    else {
	// pDimsVector field already contains some dimensions info, only
	// append new ones without reordering
	for(dimIndexType i = 0; i < pDimsVectorArg->size(); i++) {
	    pDimsVector->push_back(pDimsVectorArg->at(i));
	}
    }
    rank = pDimsVector->size();
}

/**
 * @brief Gets number of data dimensions
 *
 * @return the number of data dimensions
 */
size_t* MatVarInfo::getDims() {
    size_t* pDims;
    pDims = new size_t[rank];
    std::copy(pDimsVector->begin(), pDimsVector->end(), pDims);
    return pDims;
}

MatVarInfo::~MatVarInfo() {
    if(elementDataType == TYPEID_COMPLEX) {
	delete(static_cast<mat_complex_split_t*>(data));
	delete(realPart);
	delete(imagPart);
    }
    else if(elementDataType == std::type_index(typeid(dimIndexType))) {
	delete(static_cast<dimIndexType*>(data));
    }
    else if(elementDataType == TYPEID_REAL) {
	delete(static_cast<realType*> (data));
    }
    else {
	// unknown data type, cannot safely free anything
    }
}
} //namespace OpenCLIPER
#undef NDARRAY_DEBUG
