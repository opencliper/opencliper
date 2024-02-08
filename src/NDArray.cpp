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
/*
 * NDArray.cpp
 *
 *  Created on: 26 de oct. de 2016
 *      Author: manrod
 */
#include<fstream>
#include <OpenCLIPER/NDArray.hpp>
#include <OpenCLIPER/ConcreteNDArray.hpp>
#include <OpenCLIPER/hostKernelFunctions.hpp>
#include <OpenCLIPER/MatVarInfo.hpp>
#include <LPISupport/Utils.hpp>

// Uncomment to show class-specific debug messages
//#define NDARRAY_DEBUG

#if !defined NDEBUG && defined NDARRAY_DEBUG
    #define NDARRAY_CERR(x) CERR(x)
#else
    #define NDARRAY_CERR(x)
    #undef NDARRAY_DEBUG
#endif

namespace OpenCLIPER {
/**
 * @brief Constructor without parameters, initializes class fields.
 *
 * Pointer fields are set to nullptr, vector fields are created but empty.
 */
NDArray::NDArray() {
    //pHostData = unique_ptr<vector<complexType>>(new vector<complexType>());
    pDims = std::unique_ptr<std::vector<dimIndexType>>(new std::vector<dimIndexType>());
}

/**
 * @brief Reads a matlab variable and store its contents as a NDArray subclass object.
 *
 * @param[in] matvar matlab array variable read from file
 * @param[in] numOfSpatialDims number of dimensions of matlab array variable that are used as data spatial dimensions
 * @param[in] nDArrayOffsetInElements offset (in number of elements) from matlab variable beginning to start reading from
 */
void NDArray::loadMatlabHostData(matvar_t* matvar, dimIndexType numOfSpatialDims, dimIndexType nDArrayOffsetInElements) {
    // First dimension in matlab is number of rows (height) and in OpenCLIPER is width (number of columns)
    // Number of dimensions in matlab (rank) is always >= 2 (a scalar has dimensions 1x1, a vector, 1xN)
    dimIndexType numRows = 1, numColumns = 1, numSlicesAndBeyond = 1;
    // If number of spatial dimensions is 1 this matvar dimension is the number of columns for a Data object
    if(numOfSpatialDims == 1) {
	numColumns = matvar->dims[0];
	pDims->push_back(numColumns);
    }
    // if number of spatial dimensions is >= 2 first matvar dimension is number of rows for a Data object (only added if
    // it is > 1), second matvar dimension is number of columns for a Data object
    else {
	numColumns = matvar->dims[1];
	pDims->push_back(numColumns);
	if((numOfSpatialDims > 1) && (matvar->dims[0] > 1)) {
	    numRows = matvar->dims[0];
	    pDims->push_back(numRows);
	}
    }
    // The rest of dimensions are in the same order in matlab and in OpenCLIPER code, we process them as a unique dimension
    // (this way a number of spatial dimensions greater than 3 is supported)

    for(dimIndexType i = 2; i < numOfSpatialDims; i++) {
	numSlicesAndBeyond *= matvar->dims[i];
	pDims->push_back(matvar->dims[i]);
    }
    std::map<MatVarInfo::matlabStridesKeys, dimIndexType> matlabStrides;

    if(matlabStrides.size() != 0) {
	matlabStrides.erase(matlabStrides.begin(), matlabStrides.end());
    }

    // If number of spatial dimensions is 1 but second matlab dimension is > 1, the stride
    // for columns is the number of columns (matlab arrays are stored by rows)
    matlabStrides[MatVarInfo::matlabStridesKeys::row] = 1;
    matlabStrides[MatVarInfo::matlabStridesKeys::column] = numRows * matlabStrides[MatVarInfo::matlabStridesKeys::row];
    matlabStrides[MatVarInfo::matlabStridesKeys::sliceAndBeyond] = matlabStrides[MatVarInfo::matlabStridesKeys::column] * numColumns;
    matlabStrides[MatVarInfo::matlabStridesKeys::dataElement] = Mat_SizeOf(matvar->data_type);

    dimIndexType idx;
    dimIndexType dataElementStride = matlabStrides[MatVarInfo::matlabStridesKeys::dataElement];

    for(dimIndexType sliceAndBeyond = 0;  sliceAndBeyond < numSlicesAndBeyond; sliceAndBeyond ++) {
	for(dimIndexType row = 0; row < numRows; row++) {
	    for(dimIndexType column = 0; column < numColumns; column++) {
		idx = matlabStrides[MatVarInfo::matlabStridesKeys::sliceAndBeyond] * sliceAndBeyond +
		      matlabStrides[MatVarInfo::matlabStridesKeys::column] * column +
		      matlabStrides[MatVarInfo::matlabStridesKeys::row] * row + nDArrayOffsetInElements;
		NDARRAY_CERR("row: " << row << "\tcolumn: " << column << std::endl);
		NDARRAY_CERR("matvar->data[" << idx << "]: ");
		loadMatlabHostDataElement(matvar, idx * dataElementStride);
	    }
	}
    }
}

/**
 * @brief Destructor, frees all previously allocated memory.
 */
NDArray::~NDArray() {
    NDARRAY_CERR("~NDArray() begins..." << std::endl);
    NDARRAY_CERR("~NDArray() ends" << std::endl);
}

/**
 * @brief Gets number of dimensions of stored data (0 if there is no data stored).
 *
 * @return number of dimensions of stored data
 */
const numberOfDimensionsType NDArray::getNDims() const {
    if(pDims != nullptr)
	return pDims->size();
    else
	return 0;
}

/**
 * @brief Gets total size of stored data (product of every data dimension).
 *
 * @return number of elements of stored data
 */
const index1DType NDArray::size() const {
    index1DType size = 1;
    for(index1DType i = 0; i < pDims->size(); i++) {
	size = size * pDims->at(i);
    }
    return size;
}

/**
 * @brief Converts data of pHostData field (vector of data) to a text representation.
 *
 * @param[in] title title for data text representation
 * @return string with text representation of data in pHostData
 */
const std::string NDArray::hostDataToString(std::string title) const {
    void* pElementsArray = this->getHostDataAsVoidPointer();
    return nDArrayElementsToString(title, pElementsArray);
}

/**
 * @brief Converts array of data elements to a text representation.
 *
 * Maximum number of spatial dimensions supported is 3.
 * @param[in] title title for data text representation
 * @param[in] pElementsArray pointer to array of elements
 * @return string with text representation of data in pHostImage
 */
const std::string NDArray::nDArrayElementsToString(std::string title, const void* pElementsArray) const {
    std::stringstream ss, ss2;
    ss << title;
    ss << "(widthpos, heightpos, depthpos)";
    if(getDims() == nullptr) {
	ss << "empty";
	return ss.str();
    }
    else {
	ss << std::endl;
    }

    if(getDims()->size() > 3)
	BTTHROW(std::invalid_argument("Incorrect number of data dimensions: " +
				      std::to_string(getDims()->size()) +
				      "(maximum supported is 3)"), "NDArray::nDArrayElementsToString");
    dimIndexType width, height, depth;
    width = NDARRAYWIDTH(this);
    height = NDARRAYHEIGHT(this);
    depth = NDARRAYDEPTH(this);
    if(depth == 0)
	depth = 1;

    index1DType index1D;
    std::string elementRepresentation;

    for(dimIndexType slice = 0;  slice < depth; slice++) {
	for(dimIndexType y = 0;  y < height; y++) {
	    for(dimIndexType x = 0;  x < width; x++) {
		index1D = (x + y * width + slice * height * width);
		ss << title << " (" << x << "," << y << "," << slice << "): ";
		elementRepresentation = elementToString(pElementsArray, index1D);
		ss << elementRepresentation << std::endl;
		ss2 << elementRepresentation << " ";
	    }
	    ss2 << std::endl;
	}
	ss2 << std::endl;
    }
    return ss.str() + ss2.str();
}

/**
 * @brief Converts data of dims field (dimensions vector, vector of dimIndexType datatype data) to a text representation.
 * @param[in] title title for data text representation
 * @return string with text representation of spatial dimensions values in dims vector
 */
const std::string NDArray::dimsToString(std::string title) const {
    std::stringstream ss;
    ss << title;
    ss << "(width, height, depth, ...)" << ": ";
    if((pDims == nullptr) || (pDims->size() == 0)) {
	ss << "empty";
	return ss.str();
    }
    ss << "(";
    for(dimIndexType i = 0; i < pDims->size() - 1; i++) {
	ss << std::to_string(pDims->at(i)) << ",";
    }
    ss << std::to_string(pDims->at(pDims->size() - 1)) << ")";
    return ss.str();
}

/**
 * @brief Calculates element size in bytes depending on its data type.
 * @param[in] elementDataType data type of base element
 * @return element size in bytes
 */
dimIndexType NDArray::getElementSize(ElementDataType elementDataType) {
    dimIndexType elementSize;
    if(elementDataType == TYPEID_COMPLEX)
	return sizeof(complexType);
    if(elementDataType == TYPEID_REAL)
	return sizeof(realType);
    if(elementDataType == TYPEID_INDEX)
	return elementSize = sizeof(dimIndexType);
    if(elementDataType == TYPEID_CL_UCHAR)
	return elementSize = sizeof(cl_uchar);
    //ostream stringstream;
    std::stringstream errorStringStream;
    errorStringStream << "NDArray::getElementSize, element data type not supported: " << elementDataType.name() << std::endl;
    BTTHROW(std::invalid_argument(errorStringStream.str()), "NDArray::getElementSize");
    return 0;
}

/**
 * @brief Method for creating a subclass of NDArray depending on the data type of the base element, data for the NDArray is read from
 * a file in raw format.
 * @param[in] completeFileName name of the file to be read for getting data for the NDArray
 * @param[in,out] pSpatialDims vector with NDArray spatial dimensions (move semantics, ownership of the vector is transferred from
 * caller to NDArray and parameter value will be nullptr after executing this method)
 * @param[in] elementDataType data type of base element
 * @return pointer to the new object subclass of NDArray
 */
NDArray* NDArray::createNDArray(const std::string &completeFileName, std::vector<dimIndexType>*& pSpatialDims, ElementDataType
				elementDataType) {
    NDArray* pLocalNDArray = nullptr;

    if(elementDataType == TYPEID_COMPLEX) {
	pLocalNDArray = new ConcreteNDArray<complexType>(completeFileName, pSpatialDims);
    }
    else if(elementDataType == TYPEID_REAL) {
	pLocalNDArray = new ConcreteNDArray<realType>(completeFileName, pSpatialDims);
    }
    else if(elementDataType == TYPEID_INDEX) {
	pLocalNDArray = new ConcreteNDArray<dimIndexType>(completeFileName, pSpatialDims);
    }
    else if(elementDataType == TYPEID_CL_UCHAR) {
	pLocalNDArray = new ConcreteNDArray<cl_uchar>(completeFileName, pSpatialDims);
    }
    else {
	std::stringstream errorStringStream;
	errorStringStream << "Element data type not supported: " << elementDataType.name() << std::endl;
	BTTHROW(std::invalid_argument(errorStringStream.str()), "NDArray::createNDArray");
    }
    return pLocalNDArray;
}

/**
 * @brief Method for creating a subclass of NDArray depending on the data type of the base element, data for the NDArray is read from
 * a file in CLF format.
 * @param[in] f stream to data file previously open
 * @param[in,out] pSpatialDims vector with NDArray spatial dimensions (move semantics, ownership of the vector is transferred from
 * caller to NDArray and parameter value will be nullptr after executing this method)
 * @param[in] elementDataType data type of base element
 * @return pointer to the new object subclass of NDArray
 */
NDArray* NDArray::createNDArray(std::fstream &f, std::vector<dimIndexType>*& pSpatialDims, ElementDataType
				elementDataType) {
    NDArray* pLocalNDArray = nullptr;

    if(elementDataType == TYPEID_COMPLEX) {
	pLocalNDArray = new ConcreteNDArray<complexType>(f, pSpatialDims);
    }
    else if(elementDataType == TYPEID_REAL) {
	pLocalNDArray = new ConcreteNDArray<realType>(f, pSpatialDims);
    }
    else if(elementDataType == TYPEID_INDEX) {
	pLocalNDArray = new ConcreteNDArray<dimIndexType>(f, pSpatialDims);
    }
    else if(elementDataType == TYPEID_CL_UCHAR) {
	pLocalNDArray = new ConcreteNDArray<cl_uchar>(f, pSpatialDims);
    }
    else {
	std::stringstream errorStringStream;
	errorStringStream << "Element data type not supported: " << elementDataType.name() << std::endl;
	BTTHROW(std::invalid_argument(errorStringStream.str()), "NDArray::createNDArray");
    }
    return pLocalNDArray;
}

/**
 * @brief Method for creating a subclass of NDArray depending on the data type of the base element, data and dimensions for the
 * NDArray are copied from another NDArray.
 * @param[in] pSourceData pointer to the NDArray that is used as source of data and spatial dimensions
 * @param[in] copyData selects if the data is copied (true) or not (in this case only dimensions are copied and memory is
 * allocated according to the dimensions
 * @param[in] elementDataType data type of base element
 * @return pointer to the new object subclass of NDArray
 */
NDArray* NDArray::createNDArray(const NDArray* pSourceData, bool copyData, ElementDataType elementDataType) {
    NDArray* pLocalNDArray = nullptr;
    if(elementDataType == TYPEID_COMPLEX) {
	pLocalNDArray = new ConcreteNDArray<complexType>(pSourceData, copyData);
    }
    else if(elementDataType == TYPEID_REAL) {
	pLocalNDArray = new ConcreteNDArray<realType>(pSourceData, copyData);
    }
    else if(elementDataType == TYPEID_INDEX) {
	pLocalNDArray = new ConcreteNDArray<dimIndexType>(pSourceData, copyData);
    }
    else if(elementDataType == TYPEID_CL_UCHAR) {
	pLocalNDArray = new ConcreteNDArray<cl_uchar>(pSourceData, copyData);
    }
    else {
	std::stringstream errorStringStream;
	errorStringStream << "Element data type not supported: " << elementDataType.name() << std::endl;
	BTTHROW(std::invalid_argument(errorStringStream.str()), "NDArray::createNDArray");
    }
    return pLocalNDArray;
}

/**
 * @brief Method for creating a subclass of NDArray depending on the data type of the base element, data and dimensions for the
 * NDArray are got from a matlab variable (previously read from a file).
 * @param[in] matvar matlab array variable
 * @param[in] numOfSpatialDims number of dimensions of matlab array variable that are used as data spatial dimensions
 * @param[in] nDArrayOffsetInElements offset (in number of elements) from matlab variable beginning to start reading from
 */
NDArray* NDArray::createNDArray(matvar_t* matvar, dimIndexType numOfSpatialDims, dimIndexType nDArrayOffsetInElements) {
    NDArray* pLocalNDArray;
    switch(matvar->class_type) {
	case MAT_C_SINGLE:
	    if(matvar->isComplex) {
		pLocalNDArray = new ConcreteNDArray<std::complex<float>>(matvar, numOfSpatialDims, nDArrayOffsetInElements);
	    }
	    else {
		pLocalNDArray = new ConcreteNDArray<float>(matvar, numOfSpatialDims, nDArrayOffsetInElements);
	    }
	    break;
	/*
	 * If we want to support double instead of float data (still disabled)
	case MAT_T_DOUBLE:
	    if (matvar->isComplex) {
	        pLocalNDArray = new ConcreteNDArray<complex<double>>(matvar, numOfSpatialDims, nDArrayOffsetInElements);
	    } else {
	        pLocalNDArray = new ConcreteNDArray<double>(matvar, numOfSpatialDims, nDArrayOffsetInElements);
	    }
	    break;
	*/
	case MAT_C_UINT32:
	    //pLocalNDArray = new UnsignedNDArray(matvar, numOfSpatialDims, nDArrayOffsetInElements);
	    pLocalNDArray = new ConcreteNDArray<dimIndexType>(matvar, numOfSpatialDims, nDArrayOffsetInElements);
	    break;
	case MAT_C_UINT8:
	    //pLocalNDArray = new UnsignedNDArray(matvar, numOfSpatialDims, nDArrayOffsetInElements);
	    pLocalNDArray = new ConcreteNDArray<cl_uchar>(matvar, numOfSpatialDims, nDArrayOffsetInElements);
	    break;
	default:
	    BTTHROW(std::invalid_argument("Unsupported element data type in matlab data (supported data types are complex real single precision, non-complex real single precision and non-complex 32 bits unsigned integer"), "NDArray::createNDArray");
    }
    return pLocalNDArray;
}

/**
 * @brief Method for creating a subclass of NDArray with empty data but spatial dimensions set
 * @param[in,out] pSpatialDims vector with NDArray spatial dimensions (move semantics, ownership of the vector is transferred from
 * caller to NDArray and parameter value will be nullptr after executing this method)
 * @param[in] elementDataType data type of base element
 * @return pointer to the new object subclass of NDArray
 */
NDArray* NDArray::createNDArray(std::vector<dimIndexType>*& pSpatialDims, ElementDataType elementDataType) {
    NDArray* pLocalNDArray = nullptr;
    if(elementDataType == TYPEID_COMPLEX) {
	pLocalNDArray = new ConcreteNDArray<complexType>(pSpatialDims);
    }
    else if(elementDataType == TYPEID_REAL) {
	pLocalNDArray = new ConcreteNDArray<realType>(pSpatialDims);
    }
    else if(elementDataType == TYPEID_INDEX) {
	pLocalNDArray = new ConcreteNDArray<dimIndexType>(pSpatialDims);
    }
    else if(elementDataType == TYPEID_CL_UCHAR) {
	pLocalNDArray = new ConcreteNDArray<cl_uchar>(pSpatialDims);
    }
    else {
	std::stringstream errorStringStream;
	errorStringStream << "Element data type not supported: " << elementDataType.name() << std::endl;
	BTTHROW(std::invalid_argument(errorStringStream.str()), "NDArray::createNDArray");
    }
    return pLocalNDArray;
}

/**
 * @brief Method to create a subclass of NDArray with data copied from a void* and given spatial dimensions
 * @param[in] sourceData void* from which to copy data from
 * @param[in,out] pSpatialDims vector with NDArray spatial dimensions (move semantics, ownership of the vector is transferred from
 * caller to NDArray and parameter value will be nullptr after executing this method)
 * @param[in] elementDataType data type of base element
 * @return pointer to the new object subclass of NDArray
 */
NDArray* NDArray::createNDArray(const void* sourceData, std::vector<dimIndexType>*& pSpatialDims, ElementDataType elementDataType) {
    size_t totalLength = getElementSize(elementDataType);

    for(auto i: *pSpatialDims)
	totalLength *= i;

    NDArray* pLocalNDArray = createNDArray(pSpatialDims, elementDataType);
    ::memcpy(pLocalNDArray->getHostDataAsVoidPointer(), sourceData, totalLength);

    return pLocalNDArray;
}


/**
 * @brief Create an object of an NDArray subclass with given data and spatial dimensions
 *
 * @param[in,out] pSpatialDims vector with NDArray spatial dimensions (move semantics, ownership of the vector is transferred from
 * caller to NDArray and parameter value will be nullptr after executing this method)
 * @param[in] pData vector of data elements stored in host memory (move semantics, ownership of the vector is transferred from
 * caller to NDArray and parameter value will be nullptr after executing this method)

 * @return pointer to the new object subclass of NDArray
 */
template<typename T>
NDArray* NDArray::createNDArray(std::vector<dimIndexType>*& pSpatialDims, std::vector<T>*& pData) {
    NDArray* pLocalNDArray = new ConcreteNDArray<T>(pSpatialDims, pData);
    return pLocalNDArray;
}
template NDArray* NDArray::createNDArray(std::vector<dimIndexType>*& pSpatialDims, std::vector<complexType>*& pData);
template NDArray* NDArray::createNDArray(std::vector<dimIndexType>*& pSpatialDims, std::vector<realType>*& pData);
template NDArray* NDArray::createNDArray(std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pData);
template NDArray* NDArray::createNDArray(std::vector<dimIndexType>*& pSpatialDims, std::vector<cl_uchar>*& pData);

/**
 * @brief Create an object of an NDArray subclass with given one-dimensional data
 *
 * @param[in] pData vector of data elements stored in host memory (move semantics, ownership of the vector is transferred from
 * caller to NDArray and parameter value will be nullptr after executing this method)

 * @return pointer to the new object subclass of NDArray
 */
template<typename T>
NDArray* NDArray::createNDArray(std::vector<T>*& pData) {
    auto pSpatialDims = new std::vector<dimIndexType>(1);
    (*pSpatialDims)[0] = pData->size();
    NDArray* pLocalNDArray = new ConcreteNDArray<T>(pSpatialDims, pData);
    delete(pSpatialDims);
    return pLocalNDArray;
}
template NDArray* NDArray::createNDArray(std::vector<complexType>*& pData);
template NDArray* NDArray::createNDArray(std::vector<realType>*& pData);
template NDArray* NDArray::createNDArray(std::vector<dimIndexType>*& pData);
template NDArray* NDArray::createNDArray(std::vector<cl_uchar>*& pData);

} //namespace OpenCLIPER
#undef NDARRAY_DEBUG
