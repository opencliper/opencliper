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
 * ConcreteNDArray.cpp
 *
 *  Created on: 26 de oct. de 2016
 *      Author: manrod
 */
#include<fstream>
#include<OpenCLIPER/ConcreteNDArray.hpp>
#include<LPISupport/Utils.hpp>
#include<typeinfo>

// Uncomment to show class-specific debug messages
//#define CONCRETENDARRAY_DEBUG

#if !defined NDEBUG && defined CONCRETENDARRAY_DEBUG
    #define CONCRETENDARRAY_CERR(x) CERR(x)
#else
    #define CONCRETENDARRAY_CERR(x)
    #undef CONCRETENDARRAY_DEBUG
#endif

namespace OpenCLIPER {

/**
 * @brief Constructor without parameters, initializes class fields. It calls superclass default constructor.
 */
template <class T>
ConcreteNDArray<T>::ConcreteNDArray(): NDArray() {
}

/**
 * @brief Constructor for storing spatial dimensions and empty data in class fields (element data type is complexType).
 *
 * This constructor has move semantics (in spite of not using && notation):
 * after call, parameters memory deallocation is responsibility of this class
 * (parameters are set to nullptr at the end of the method).
 * @param[in,out] pSpatialDims vector with sizes of each spatial dimension
 */
template <>
ConcreteNDArray<complexType>::ConcreteNDArray(std::vector<dimIndexType>*& pSpatialDims) {
    setDims(pSpatialDims);
    complexType complexZero(0.0, 0.0);
    std::vector <complexType>* pHostComplexDataLocal = new std::vector <complexType>(this->size(), complexZero);
    setHostData(pHostComplexDataLocal);
}

/**
 * @brief Constructor for storing spatial dimensions and empty data in class fields (element data type is dimIndexType).
 *
 * This constructor has move semantics (in spite of not using && notation):
 * after call, parameters memory deallocation is responsibility of this class
 * (parameters are set to nullptr at the end of the method).
 * @param[in,out] pSpatialDims vector with sizes of each spatial dimension
 */
template <>
ConcreteNDArray<dimIndexType>::ConcreteNDArray(std::vector<dimIndexType>*& pSpatialDims) {
    setDims(pSpatialDims);
    std::vector <dimIndexType>* pHostUnsignedDataLocal = new std::vector <dimIndexType>(this->size(), 0);
    setHostData(pHostUnsignedDataLocal);
}

/**
 * @brief Constructor for storing spatial dimensions and empty data in class fields (element data type is realType).
 *
 * This constructor has move semantics (in spite of not using && notation):
 * after call, parameters memory deallocation is responsibility of this class
 * (parameters are set to nullptr at the end of the method).
 * @param[in,out] pSpatialDims vector with sizes of each spatial dimension
 */
template <>
ConcreteNDArray<realType>::ConcreteNDArray(std::vector<dimIndexType>*& pSpatialDims) {
    setDims(pSpatialDims);
    std::vector <realType>* pHostLocal = new std::vector <realType>(this->size(), 0.0);
    setHostData(pHostLocal);
}

/**
 * @brief Constructor for storing spatial dimensions and empty data in class fields (element data type is cl_uchar).
 *
 * This constructor has move semantics (in spite of not using && notation):
 * after call, parameters memory deallocation is responsibility of this class
 * (parameters are set to nullptr at the end of the method).
 * @param[in,out] pSpatialDims vector with sizes of each spatial dimension
 */
template <>
ConcreteNDArray<cl_uchar>::ConcreteNDArray(std::vector<dimIndexType>*& pSpatialDims) {
    setDims(pSpatialDims);
    std::vector <cl_uchar>* pHostUnsignedDataLocal = new std::vector <cl_uchar>(this->size(), 0);
    setHostData(pHostUnsignedDataLocal);
}

/**
 * @brief Constructor for storing spatial dimensions and data in class fields.
 * This constructor has move semantics (in spite of not using && notation):
 * after call, parameters memory deallocation is responsibility of this class
 * (parameters are set to nullptr at the end of the method).
 * @param[in,out] pSpatialDims vector with sizes of each spatial dimension
 * @param[in,out] pHostData vector of \<T\> type data elements stored in host memory
 */
template <class T>
ConcreteNDArray<T>::ConcreteNDArray(std::vector<dimIndexType>*& pSpatialDims, std::vector<T>*& pHostData) {
    setDims(pSpatialDims);
    setHostData(pHostData);
}

/**
 * @brief Constructor that reads data for the ConcreteNDArray object from a file in raw format.
 * @param[in] completeFileName name of the file to be read for getting data
 * @param[in,out] pSpatialDims vector with NDArray spatial dimensions (move semantics, ownership of the vector is transferred from caller to this object)
 */
template <class T>
ConcreteNDArray<T>::ConcreteNDArray(const std::string &completeFileName, std::vector<dimIndexType>*& pSpatialDims) {
    std::vector<T>* pTempData;
    setDims(pSpatialDims); // store dimension vector and set parameter to nullptr (move semantics)
    pTempData = new std::vector<T>; // vector for data
    pTempData->resize(this->size()); //size(): product of size of each dimension
    std::ifstream f(completeFileName, std::ios::binary);
    if(!f.good()) {
	BTTHROW(std::invalid_argument(completeFileName + " cannot be read\n"), "ConcreteNDArray::ConcreteNDArray");
    }
#if !defined NDEBUG && defined CONCRETENDARRAY_DEBUG
    // get length of file:
    f.seekg(0, f.end);
    int fileLength = f.tellg();
    f.seekg(0, f.beg);
    CONCRETENDARRAY_CERR(completeFileName << " file length: " << fileLength << " bytes\n");
#endif
    f.read(reinterpret_cast<char*>(pTempData->data()), this->size() * sizeof(T));
    // store data and set parameter to nullptr (move semantics)
    setHostData(pTempData);
    f.close();
}

/**
 * @brief Constructor that reads data for the ConcreteNDArray object from a file in CFL format.
 * @param[in] f stream to data file previously open
 * @param[in,out] pSpatialDims vector with NDArray spatial dimensions (move semantics, ownership of the vector is transferred from caller to this object)
 */
template <class T>
ConcreteNDArray<T>::ConcreteNDArray(std::fstream &f, std::vector<dimIndexType>*& pSpatialDims) {
    std::vector<T>* pTempData = new std::vector<T>; // vector for data
    setDims(pSpatialDims); // store dimension vector and set parameter to nullptr (move semantics)
    pTempData->resize(size()); // size of vector is the number of elements; size(): product of size of each dimension
    LPISupport::Utils::readBytesFromFile(f, reinterpret_cast<char*>(pTempData->data()), this->size()* sizeof(T));
    // store data and set parameter to nullptr (move semantics)
    setHostData(pTempData);
}

/**
 * @brief Constructor that creates a copy of a ConcreteNDArray object  with complexType data type elements (dimensions are copied always,
 * image data only if copyData parameter is true).
 * @param[in] pSourceData ConcreteNDArray object source of spatial and temporal dimensions (complexType elements)
 * @param[in] copyData data (not only dimensions) are copied if this parameter is true (default value: false)
 */
template <>
ConcreteNDArray<complexType>::ConcreteNDArray(const NDArray* pSourceData, bool copyData) {
    std::vector<dimIndexType>* pLocalDims = new std::vector<dimIndexType>(*(pSourceData->getDims()));
    setDims(pLocalDims);
    std::vector<complexType>* pLocalHostData;
    if(copyData) {
	const ConcreteNDArray<complexType>* pTypedSourceData = static_cast<const ConcreteNDArray<complexType>*>(pSourceData);
	pLocalHostData = new std::vector<complexType>(*(pTypedSourceData->getHostData()));
    }
    else {
	// Create image data initialized to a vector of complex values (0.0, 0.0) and with a number of values equal to the
	// multiplication of dims vector values
	complexType zeroElement(0.0, 0.0);
	pLocalHostData = new std::vector<complexType>(pSourceData->size(), zeroElement);
    }
    setHostData(pLocalHostData);
}

/**
 * @brief Constructor that creates a copy of a ConcreteNDArray object with dimIndexType data type elements (dimensions are copied always,
 * image data only if copyData parameter is true).
 * @param[in] pSourceData ConcreteNDArray object source of spatial and temporal dimensions (dimIndexType elements)
 * @param[in] copyData data (not only dimensions) are copied if this parameter is true (default value: false)
 */
template <>
ConcreteNDArray<dimIndexType>::ConcreteNDArray(const NDArray* pSourceData, bool copyData) {
    std::vector<dimIndexType>* pLocalDims = new std::vector<dimIndexType>(*(pSourceData->getDims()));
    setDims(pLocalDims);
    std::vector<dimIndexType>* pLocalHostData;
    if(copyData) {
	const ConcreteNDArray<dimIndexType>* pTypedSourceData = static_cast<const ConcreteNDArray<dimIndexType>*>(pSourceData);
	pLocalHostData = new std::vector<dimIndexType>(*(pTypedSourceData->getHostData()));
    }
    else {
	dimIndexType zeroElement = 0;
	pLocalHostData = new std::vector<dimIndexType>(pSourceData->size(), zeroElement);
    }
    setHostData(pLocalHostData);
}

/**
 * @brief Constructor that creates a copy of a ConcreteNDArray object with realType data type elements (dimensions are copied always,
 * image data only if copyData parameter is true).
 * @param[in] pSourceData ConcreteNDArray object source of spatial and temporal dimensions (realType elements)
 * @param[in] copyData data (not only dimensions) are copied if this parameter is true (default value: false)
 */
template <>
ConcreteNDArray<realType>::ConcreteNDArray(const NDArray* pSourceData, bool copyData) {
    std::vector<dimIndexType>* pLocalDims = new std::vector<dimIndexType>(*(pSourceData->getDims()));
    setDims(pLocalDims);
    std::vector<realType>* pLocalHostData;
    if(copyData) {
	const ConcreteNDArray<realType>* pTypedSourceData = static_cast<const ConcreteNDArray<realType>*>(pSourceData);
	pLocalHostData = new std::vector<realType>(*(pTypedSourceData->getHostData()));
    }
    else {
	realType zeroElement = 0.0;
	pLocalHostData = new std::vector<realType>(pSourceData->size(), zeroElement);
    }
    setHostData(pLocalHostData);
}

/**
 * @brief Constructor that creates a copy of a ConcreteNDArray object with dimIndexType data type elements (dimensions are copied always,
 * image data only if copyData parameter is true).
 * @param[in] pSourceData ConcreteNDArray object source of spatial and temporal dimensions (cl_uchar elements)
 * @param[in] copyData data (not only dimensions) are copied if this parameter is true (default value: false)
 */
template <>
ConcreteNDArray<cl_uchar>::ConcreteNDArray(const NDArray* pSourceData, bool copyData) {
    std::vector<dimIndexType>* pLocalDims = new std::vector<dimIndexType>(*(pSourceData->getDims()));
    setDims(pLocalDims);
    std::vector<cl_uchar>* pLocalHostData;
    if(copyData) {
	const ConcreteNDArray<cl_uchar>* pTypedSourceData = static_cast<const ConcreteNDArray<cl_uchar>*>(pSourceData);
	pLocalHostData = new std::vector<cl_uchar>(*(pTypedSourceData->getHostData()));
    }
    else {
	cl_uchar zeroElement = 0;
	pLocalHostData = new std::vector<cl_uchar>(pSourceData->size(), zeroElement);
    }
    setHostData(pLocalHostData);
}

/**
 * @brief Constructor for reading data from a matlab variable
 * @param[in] matvar matlab array variable read from file
 * @param[in] numOfSpatialDims number of dimensions of matlab array variable that are used as data spatial dimensions
 * @param[in] nDArrayOffsetInElements  offset (in number of elements) from matlab variable beginning to start reading from
 */
template <class T>
ConcreteNDArray<T>::ConcreteNDArray(matvar_t* matvar, dimIndexType numOfSpatialDims, dimIndexType nDArrayOffsetInElements) {
    loadMatlabHostData(matvar, numOfSpatialDims, nDArrayOffsetInElements);
    // Save a copy of the pointer to data array (pHostData->data()) in pHostBuffer and pHostImage
    // as they do not contain a valid pointer to mapped host memory until Data::setApp method is called
    CONCRETENDARRAY_CERR(hostDataToString("ConcreteNDArray: "));
}

/**
 * @brief Gets one element from a matlab variable (previously read from a matlab file), base data type is complexType
 * @param[in] matvar matlab array variable previously read from file
 * @param[in] offsetInBytes offset from beginning of matlab variable (in bytes) where element data must be read
 */
template <>
inline void ConcreteNDArray<complexType>::loadMatlabHostDataElement(matvar_t* matvar, dimIndexType offsetInBytes) {
    mat_complex_split_t* complex_data = (mat_complex_split_t*) matvar->data;
    char* pCharRealPart = (char*) complex_data->Re;
    char* pCharImagPart = (char*) complex_data->Im;
    complexType complexElement;
    realType realPart, imagPart;
    realPart = *((realType*)(pCharRealPart + offsetInBytes));
    imagPart = *((realType*)(pCharImagPart + offsetInBytes));
    complexElement = {realPart, imagPart};
    pHostData->push_back(complexElement);
    CONCRETENDARRAY_CERR(realPart << "+" << imagPart << "i" << std::endl);
}

/**
 * @brief Gets one element from a matlab variable (previously read from a matlab file), base data type is dimIndexType
 * @param[in] matvar matlab array variable previously read from file
 * @param[in] offsetInBytes offset from beginning of matlab variable (in bytes) where element data must be read
 */
template <>
inline void ConcreteNDArray<dimIndexType>::loadMatlabHostDataElement(matvar_t* matvar, dimIndexType offsetInBytes) {
    char* data = (char*)matvar->data;
    dimIndexType element;
    element = *((dimIndexType*)(data + offsetInBytes));
    pHostData->push_back(element);
    CONCRETENDARRAY_CERR(element << std::endl);
}

/**
 * @brief Gets one element from a matlab variable (previously read from a matlab file), base data type is realType
 * @param[in] matvar matlab array variable previously read from file
 * @param[in] offsetInBytes offset from beginning of matlab variable (in bytes) where element data must be read
 */
template <>
inline void ConcreteNDArray<realType>::loadMatlabHostDataElement(matvar_t* matvar, dimIndexType offsetInBytes) {
    char* data = (char*)matvar->data;
    realType element;
    element = *((realType*)(data + offsetInBytes));
    pHostData->push_back(element);
    CONCRETENDARRAY_CERR(element << std::endl);
}

/**
 * @brief Gets one element from a matlab variable (previously read from a matlab file), base data type is cl_uchar
 * @param[in] matvar matlab array variable previously read from file
 * @param[in] offsetInBytes offset from beginning of matlab variable (in bytes) where element data must be read
 */
template <>
inline void ConcreteNDArray<cl_uchar>::loadMatlabHostDataElement(matvar_t* matvar, dimIndexType offsetInBytes) {
    char* data = (char*)matvar->data;
    cl_uchar element;
    element = *((cl_uchar*)(data + offsetInBytes));
    pHostData->push_back(element);
    CONCRETENDARRAY_CERR(element << std::endl);
}

/**
 * @brief Destructor, frees all previously allocated memory
 */
template <class T>
ConcreteNDArray<T>::~ConcreteNDArray() {
    //CONCRETENDARRAY_CERR("~ConcreteNDArray() begins..." << std::endl);
    if(pHostData != nullptr) {
	pHostData.reset(nullptr); // pHostData is a smart pointer
    }
    //CONCRETENDARRAY_CERR("~ConcreteNDArray() ends" << std::endl);
}

/**
 * @brief Converts one data element of an NDArray to a text representation
 * @param[in] pElementsArray pointer to array of elements
 * @param[in] index1D 1-dimensional index for element from array
 * @return string with text representation of data element
 */
template <class T> const std::string ConcreteNDArray<T>::elementToString(const void* pElementsArray, dimIndexType index1D) const {
    std::string stringValue;
    if(typeid(T) == typeid(complexType)) {
	complexType* pTypedArray;
	pTypedArray = (complexType*) pElementsArray;
	stringValue = "(" + std::to_string(pTypedArray[index1D].real()) + "," +
		      std::to_string(pTypedArray[index1D].imag()) + ")";
    }
    else if(typeid(T) == typeid(dimIndexType)) {
	dimIndexType* pTypedArray;
	pTypedArray = (dimIndexType*) pElementsArray;
	stringValue = std::to_string(pTypedArray[index1D]);
    }
    else if(typeid(T) == typeid(realType)) {
	realType* pTypedArray;
	pTypedArray = (realType*) pElementsArray;
	stringValue = std::to_string(pTypedArray[index1D]);
    }
    else if(typeid(T) == typeid(cl_uchar)) {
	cl_uchar* pTypedArray;
	pTypedArray = (cl_uchar*) pElementsArray;
	stringValue = std::to_string(pTypedArray[index1D]);
    }
    else {
	BTTHROW(std::invalid_argument("element data type not supported in elementToString method: " + std::string(typeid(T).name())), "ConcreteNDArray::elementToString");
    }
    return stringValue;
}

/**
 * @brief Calculates rows, columns, slices, ... offsets for accessing images/volumes stored as 1D arrays of \<T\> type
 * elements
 * @return pointer to vector variable storing strides
 */
template <class T>
std::vector <dimIndexType>* ConcreteNDArray<T>::calcUnaligned1DArrayStridesFromNDArrayDims() const {
    std::vector<dimIndexType>* pSpatialDimsStridesVector = new std::vector<dimIndexType>();
    // dimensions order is columns, rows, slices, ...
    dimIndexType acumStride;
    if (typeid(T) == typeid(complexType)) {
        acumStride = 1;// every column has 2 floats (real and imaginary part of complex number) but strides should come in num. of complex elements
    } else if (typeid(T) == typeid(realType)) {
        acumStride = 1;// every column has 1 float (instead of 2 floats, real and imaginary part of complex number, for a ComplexNDArray)
    } else if (typeid(T) == typeid(dimIndexType)) {
        acumStride = 1;// every column has 1 uint (instead of 2 floats, real and imaginary part of complex number, for a ComplexNDArray)
    } else if (typeid(T) == typeid(cl_uchar)) {
        acumStride = 1;// every column has 1 uint (instead of 2 floats, real and imaginary part of complex number, for a ComplexNDArray)
    } else {
	BTTHROW(std::invalid_argument("Unsupported type in calcUnaligned1DArrayStridesFromNDArrayDims method: "
				      + std::string(typeid(this).name())), "ConcreteNDArray::calcUnaligned1DArrayStridesFromNDArrayDims");
    }
    for(dimIndexType spatialDimId = 0; spatialDimId < getNDims(); spatialDimId++) {
	pSpatialDimsStridesVector->push_back(acumStride);
	acumStride *= getDims()->at(spatialDimId);  // new stride is equal to previous stride * previous dimension
    }
    return pSpatialDimsStridesVector;
}

#include<OpenCLIPER/ConcreteNDArrayPrototypes.hpp>
} //namespace OpenCLIPER
#undef CONCRETENDARRAY_DEBUG
