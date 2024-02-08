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
 * XData.cpp
 *
 *  Created on: 27 de oct. de 2016
 *      Author: manrod
 */

#include <opencv2/opencv.hpp>
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/MatVarDimsData.hpp>

#include <iostream>
#include <cstring> // for memcpy
#include <thread>  // std::thread

// Uncomment to show class-specific debug messages
#define XDATA_DEBUG

#if !defined NDEBUG && defined XDATA_DEBUG
    #define XDATA_CERR(x) CERR(x)
#else
    #define XDATA_CERR(x)
    #undef XDATA_DEBUG
#endif


namespace OpenCLIPER {

//*********************************
// Constructors and destructor
//*********************************

//------------
// Empty Data
//------------

/**
 * @brief Constructor that creates an empty XData object.
 *
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, ElementDataType elementDataType) : Data(elementDataType) {
    this->pCLapp = pCLapp;
}


//-------------------------
// From a bunch of NDarrays
//-------------------------

/**
 * @brief Constructor that creates an XData object from a single NDArray.
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in,out] pNDArray pointer to NDArray object
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, NDArray*& pNDArray, ElementDataType elementDataType): Data(elementDataType) {

    std::vector<dimIndexType>* pDynDims = new std::vector<dimIndexType>({1});
    std::vector<NDArray*>* pNDArrays = new std::vector<NDArray*>(1);

    (*pNDArrays)[0] = pNDArray;
    internalSetData(pNDArrays);
    setDynDims(pDynDims);
    pNDArray = nullptr;

    setApp(pCLapp, true);
}

/**
 * @brief Constructor that creates an XData object from a vector of NDArrays.
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in,out] pNDArrays pointer to vector of pointers to NDArray objects
 * @param[in,out] pDynDims pointer to vector of temporal dimensions
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<NDArray*>*& pNDArrays, std::vector<dimIndexType>*& pTempDims, ElementDataType elementDataType): Data(pNDArrays, pTempDims, elementDataType) {

    setApp(pCLapp, true);
}


//------------------------------------------------------
// From given dimensions, uninitialized: single NDArray
//------------------------------------------------------

/**
 * @brief Constructor that creates an uninitialized XData object containing a single, 1-dimensional NDArray from given width
 *
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] width size of the NDArray along the first dimension
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, ElementDataType elementDataType): Data(width, elementDataType) {
    setApp(pCLapp, false);
}

/**
 * @brief Constructor that creates an uninitialized XData object containing a single, 2-dimensional NDArray from given width and height
 *
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] width size of the NDArray along the first dimension
 * @param[in] height size of the NDArray along the second dimension
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, ElementDataType elementDataType): Data(width, height, elementDataType) {
    setApp(pCLapp, false);
}

/**
 * @brief Constructor that creates an uninitialized XData object containing a single, 3-dimensional NDArray from given width, height and depth
 *
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] width size of the NDArray along the first dimension
 * @param[in] height size of the NDArray along the second dimension
 * @param[in] depth size of the NDArray along the third dimension
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, dimIndexType depth, ElementDataType elementDataType): Data(width, height, depth, elementDataType) {
    setApp(pCLapp, false);
}

/**
 * @brief Constructor that creates an uninitialized XData object containing a single NDArray from given spatial dimensions
 *
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in,out] pSpatialDims pointer to vector of spatial dimensions
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<dimIndexType>*& pSpatialDims, ElementDataType elementDataType): Data(pSpatialDims, elementDataType) {
    setApp(pCLapp, false);
}

//------------------------------------------------------------------------------------
// From given dimensions, uninitialized: several NDArrays with no temporal dimensions
//------------------------------------------------------------------------------------

/**
 * @brief Constructor that creates an uninitialized XData object containing several equally-sized NDArrays with given spatial dimensions and no temporal dimensions
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in] numNDArrays number of NDArrays the new XData object should contain
 * @param[in,out] pSpatialDims pointer to vector of spatial dimensions
 * @param[in] elementDataType data type of base element
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType numNDArrays, std::vector<dimIndexType>*& pSpatialDims, ElementDataType elementDataType):
       Data(numNDArrays, pSpatialDims, elementDataType) {
    setApp(pCLapp, false);
}

/**
 * @brief Constructor that creates an uninitialized XData object containing several arbitrarily-sized NDArrays with given spatial dimensions and no temporal dimensions.
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in,out] pSpatialDims pointer to vector of pointers to spatial dimensions vectors
 * @param[in] elementDataType data type of base element
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<std::vector<dimIndexType>*>*& pSpatialDims, ElementDataType elementDataType):
       Data(pSpatialDims, elementDataType) {
    setApp(pCLapp, false);
}


//---------------------------------------------------------------------------------------------
// From given dimensions, uninitialized: several NDArrays with spatial and temporal dimensions
//---------------------------------------------------------------------------------------------

/**
 * @brief Constructor that creates an uninitialized XData object containing several equally-sized NDArrays with given spatial and temporal dimensions
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in] numNDArrays number of NDArrays the new XData object should contain
 * @param[in,out] pSpatialDims pointer to vector of spatial dimensions
 * @param[in,out] pTempDims pointer to vector of temporal dimensions
 * @param[in] elementDataType data type of base element
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType numNDArrays, std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, ElementDataType elementDataType):
       Data(numNDArrays, pSpatialDims, pTempDims, elementDataType) {
    setApp(pCLapp, false);
}

/**
 * @brief Constructor that creates an uninitialized XData object containing several arbitrarily-sized NDArrays with given spatial and temporal dimensions.
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in,out] pSpatialDims pointer to vector of pointers to spatial dimensions vectors
 * @param[in,out] pTempDims pointer to vector of temporal dimensions
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<std::vector<dimIndexType>*>*& pSpatialDims,
	    std::vector<dimIndexType>*& pTempDims, ElementDataType elementDataType, bool initZero):
       Data(pSpatialDims, pTempDims, elementDataType) {
    //setApp(pCLapp, false);
    //setApp(pCLapp, true);
    if (initZero)
	setApp(pCLapp, true);
    else
	setApp(pCLapp, false);
}


//---------------------------------------------------------
// From given dimensions and an std::vector: single NDArray
//---------------------------------------------------------

/**
 * @brief Constructor that creates an XData object containing a single, 1-dimensional NDArray from given width and a data vector
 *
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] width size of the NDArray along the first dimension
 * @param[in,out] pData pointer to the data to be transferred to the NDArray
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 */
template<typename T>
XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, std::vector<T>*& pData): Data(width, pData) {
    setApp(pCLapp, true);
}

template XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, std::vector<complexType>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, std::vector<realType>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, std::vector<dimIndexType>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, std::vector<cl_uchar>*& pData);

/**
 * @brief Constructor that creates an XData object containing a single, 2-dimensional NDArray from given width and height, and a data vector
 *
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] width size of the NDArray along the first dimension
 * @param[in] height size of the NDArray along the second dimension
 * @param[in,out] pData pointer to the data to be transferred to the NDArray
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 */
template<typename T>
XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, std::vector<T>*& pData): Data(width, height, pData) {
    setApp(pCLapp, true);
}

template XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, std::vector<complexType>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, std::vector<realType>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, std::vector<dimIndexType>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, std::vector<cl_uchar>*& pData);

/**
 * @brief Constructor that creates an XData object containing a single, 3-dimensional NDArray from given width, height and depth, and a data vector
 *
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] width size of the NDArray along the first dimension
 * @param[in] height size of the NDArray along the second dimension
 * @param[in] depth size of the NDArray along the third dimension
 * @param[in,out] pData pointer to the data to be transferred to the NDArray
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 */
template<typename T>
XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, dimIndexType depth, std::vector<T>*& pData): Data(width, height, depth, pData) {
    setApp(pCLapp, true);
}

template XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, dimIndexType depth, std::vector<complexType>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, dimIndexType depth, std::vector<realType>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, dimIndexType depth, std::vector<dimIndexType>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, dimIndexType depth, std::vector<cl_uchar>*& pData);

/**
 * @brief Constructor that creates an XData object containing a single NDArray from given spatial dimensions and a data vector
 *
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in,out] pSpatialDims pointer to vector of spatial dimensions
 * @param[in,out] pData pointer to the data to be transferred to the NDArray
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 */
template<typename T>
XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<dimIndexType>*& pSpatialDims, std::vector<T>*& pData): Data(pSpatialDims, pData) {
    setApp(pCLapp, true);
}

template XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<dimIndexType>*& pSpatialDims, std::vector<complexType>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<dimIndexType>*& pSpatialDims, std::vector<realType>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<dimIndexType>*& pSpatialDims, std::vector<cl_uchar>*& pData);


//-----------------------------------------------------------------------------------------------------------
// From given dimensions and a vector of std::vectors: severall NDArrays with spatial and temporal dimensions
//-----------------------------------------------------------------------------------------------------------
/**
 * @brief Constructor that creates an XData object containing several NDArrays equal in size from given spatial dimensions and a data vector
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in,out] pSpatialDims pointer to vector of spatial dimensions (all NDArrays equal in size)
 * @param[in,out] pTempDims pointer to vector of temporal dimensions
 * @param[in,out] pData pointer to the data to be transferred to the NDArray: one std::vector<>* per NDArray
 * @param[in] elementDataType data type of base element
 */
template<typename T>
XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<T>*>*& pData): Data(pSpatialDims, pTempDims, pData) {
    setApp(pCLapp, true);
}

template XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<complexType>*>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<realType>*>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<dimIndexType>*>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<cl_uchar>*>*& pData);

/**
 * @brief Constructor that creates a Data object containing several NDArrays of different sizes from given spatial dimensions and a data vector
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in,out] pSpatialDims pointer to vector of vectors of spatial dimensions: one std::vector<>* per nDArray
 * @param[in,out] pTempDims pointer to vector of temporal dimensions
 * @param[in,out] pData pointer to the data to be transferred to the NDArray: one std::vector<>* per NDArray
 * @param[in] elementDataType data type of base element
 */
template<typename T>
XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<T>*>*& pData): Data(pSpatialDims, pTempDims, pData) {
    setApp(pCLapp, true);
}

template XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<complexType>*>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<realType>*>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<dimIndexType>*>*& pData);
template XData::XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<cl_uchar>*>*& pData);


//------------------------------------------------------------
// From another Data or Data-derived object (copy contructors)
//------------------------------------------------------------

/**
 * @brief Constructor that creates an empty XData from another Data object.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData Data object source of spatial and temporal dimensions
 * @param[in] copyData if true, also copy data (not only dimensions) from sourceData object to this object
 * @param[in] copyDataToDevice if true host memory data are copied to device memory
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data>& sourceData, bool copyData, bool copyDataToDevice): Data(sourceData, copyData, copyDataToDevice) {
	setApp(pCLapp, copyDataToDevice);// setApp called from Data::internalSetData called from Data::setData called by Data constructor
}

/**
 * @brief Constructor that creates an uninitialized XData object from a given Data object.
 * Element data type for the new object is specified by the caller
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData Data object source of spatial and temporal dimensions
 * @param[in] newElementDataType Data type of the newly created object
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data>& sourceData, ElementDataType newElementDataType): Data(sourceData, newElementDataType) {
    setApp(pCLapp, false);// setApp called from Data::internalSetData called from Data::setData called by Data constructor
}

/**
 * @brief Constructor that creates an XData object from another XData object.
 * @param[in] pCLapp shared_ptr to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData XData object source of spatial and temporal dimensions
 * @param[in] copyData if true also data (not only dimensions) are copied from sourceData object to this object host memory
 * @param[in] copyDataToDevice if true host memory data are copied to device memory
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<XData>& sourceData, bool copyData, bool copyDataToDevice): Data(sourceData, copyData, copyDataToDevice) {
    setApp(pCLapp, copyData); // setApp called from Data::internalSetData called from Data::setData called by Data constructor
    //setApp(pCLapp, true);
}

/**
 * @brief Constructor that creates an uninitialized XData object from another XData object.
 * Element data type for the new object is specified by the caller
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData Object to copy data structure from
 * @param[in] newElementDataType Data type of the newly created object
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<XData>& sourceData, ElementDataType newElementDataType): Data(sourceData, newElementDataType) {
    setApp(pCLapp, false);
}

/**
 * @brief Constructor that creates an uninitialized XData object from a KData object.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData Data object source of spatial and temporal dimensions
 * @throw std::invalid_argument if all KData NDArrays do not have the same spatial dimensions
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<KData>& sourceData) {
    if(sourceData->getAllSizesEqual() == 0)
	BTTHROW(std::invalid_argument("XData copy constructor with KData source not valid if KData NDArrays do not have the same spatial dimensions"), "XData::XData");

    std::vector<std::vector<dimIndexType>*>* pArraysDims = new std::vector<std::vector<dimIndexType>*>();
    for(dimIndexType i = 0; i < sourceData->getDynDimsTotalSize(); i++) {
	std::vector<dimIndexType>* pNDArrayDims;
	pNDArrayDims = new std::vector<dimIndexType>(*(sourceData->getData()->at(0)->getDims()));
	pArraysDims->push_back(pNDArrayDims);
    }

    std::vector<dimIndexType>* pDynDims = new std::vector<dimIndexType>(*(sourceData->getDynDims()));
    createFromDimensions(pArraysDims->size(), pArraysDims, pDynDims, sourceData->getElementDataType());

    setApp(pCLapp, false);
}

/**
 * @brief Constructor that creates an XData object from a group of image files (every file contains 1 x-space image).
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] fileNames vector of names of the image files
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 */
/*
XData::XData(const std::shared_ptr<CLapp>& pCLapp, const std::string fileName, ElementDataType elementDataType,
			 SyncSource host2DeviceSync): Data(elementDataType) {
    std::vector<std::string> fileNames;
    fileNames.push_back(fileName);
    load(fileNames);
    setApp(pCLapp, host2DeviceSync);
}
*/
/**
 * @brief Constructor that creates an XData object from a group of image files (every file contains 1 x-space image).
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] fileNames vector of names of the image files
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, const std::vector<std::string> &fileNames, ElementDataType elementDataType): Data(elementDataType) {
    load(fileNames);
    setApp(pCLapp, true);
}

/**
 * @brief Constructor that loads data from a group of files in raw format (see @ref loadRawHostData).
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] dataFileNamePrefix prefix common to names of all files to be read
 * @param[in,out] pArraysDims pointer to vector of vectors with spatial dimensions of every data to be read into a NDArray
 * @param[in,out] pDynDims pointer to vector with temporal dimensions
 * @param[in] framesFileNameSuffix suffix for file name part related to frames
 * @param[in] fileNameExtension extension common to all file names
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, const std::string& dataFileNamePrefix,
	     std::vector<std::vector< dimIndexType >*>*& pArraysDims, std::vector <dimIndexType>*& pDynDims,
	     const std::string& framesFileNameSuffix, const std::string& fileNameExtension, ElementDataType elementDataType):
    Data(elementDataType) {
    loadRawHostData(dataFileNamePrefix, pArraysDims, pDynDims, framesFileNameSuffix, fileNameExtension);
    setApp(pCLapp, true);
}

/**
 * @brief Constructor that creates an XData object from a file in matlab format (containing two variables: dimensions and x-image data).
 *
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] fileName name of the data file
 */
XData::XData(const std::shared_ptr<CLapp>& pCLapp, const std::string& fileName, ElementDataType elementDataType): Data() {
    std::map<std::string, matvar_t*>* pMatlabVariablesMap;
    try {
	XDATA_CERR("Trying to read image file...\n");
	commonFieldInitialization(elementDataType);
	pNDArraysForGet = new std::vector<const NDArray*>;
	pNDArrays = nullptr;
	std::vector<std::string> fileNames;
	fileNames.push_back(fileName);
	load(fileNames);
	XDATA_CERR("Done\n");
    }
    catch(std::exception &e) {    // It is not an image file, trying to read a matlab file
	XDATA_CERR("Trying to read matlab file...\n");
	pMatlabVariablesMap = Data::readMatlabVariablesFromFile(fileName);
	XDATA_CERR("Done\n");
	matvar_t* pDimsMatVar = nullptr, *pXDataMatVar = nullptr;

	// Read of matlab variable storing number of spatial and temporal dimensions of data (if not exists it is an unrecoverable error and method aborts)
	try {
	    pDimsMatVar = pMatlabVariablesMap->at(MatVarDimsData::matVarNameDims);
	}
	catch(std::out_of_range& e) {
	    BTTHROW(std::invalid_argument("Error: no Dims variable in matlab file"), "XData::XData");
	}

	dimIndexType nSpatialDims = 0, nTemporalDims = 0, nCoils = 0, numOfNDArraysToBeRead = 0; //, firstTemporalDimIndex = 0;
	std::vector<dimIndexType>* pSpatialDimsFullySampled = new std::vector<dimIndexType>();
	pMatVarDimsData.reset(new MatVarDimsData(pCLapp, pDimsMatVar));
	pMatVarDimsData->getAllDims(pSpatialDimsFullySampled, nCoils, pDynDims.get());
	//readDimsMatlabVar(pDimsMatVar, pSpatialDimsFullySampled, nCoils, pDynDims.get());
	nSpatialDims = pSpatialDimsFullySampled->size();
	delete(pSpatialDimsFullySampled);
	nTemporalDims = pDynDims->size();

	// Read of matlab variable storing XData data (if not exists it is an unrecoverable error and method is aborted)
	try {
	    pXDataMatVar = pMatlabVariablesMap->at(matVarNameXData);
	    // out_of_range exception if there is no value with the key "XData" in the map
	}
	catch(std::out_of_range& e) {
	    BTTHROW(std::invalid_argument("Error: no XData variable in matlab file"), "XData::XData");
	}

	// Note on valid temporal dimensions: if XData only contains one x-image, size of
	// temporal dimensions array is 0 (and rank of XData variable must be equal to the number of
	// spatial dimensions) or is 1 with a stored value of 1 at this position (and rank of XData
	// variable must be equal to the number of spatial dimensions plus 1)
	if((nCoils <= 1) && (static_cast<dimIndexType>(pXDataMatVar->rank) == (nSpatialDims + nTemporalDims))) {
	    // No coils dimensions, only 1 coil used
	    //firstTemporalDimIndex = nSpatialDims; // Spatial dimensions indexes from 0 to nSpatialDims - 1
	}
	else {   // Incorrect number of dimensions for XData
	    XDATA_CERR("Error: incorrect dimensions for XData\n");
	    std::ostringstream oss;
	    oss << "Invalid XData dimensions (" << pXDataMatVar->rank << "), should be " << (nSpatialDims + nTemporalDims);
	    oss << " (number of spatial dims: " << nSpatialDims;
	    oss << ", number of temporal dims: " << nTemporalDims << ")" << std::endl;
	    BTTHROW(std::out_of_range(oss.str()), "XData::XData");
	}

	// Number of XData NDArrays to be read is the product of all temporal dimensions
	numOfNDArraysToBeRead = getDynDimsTotalSize();
	XDATA_CERR("XData reading data...\n");
	((Data*)(this))->loadMatlabHostData(pXDataMatVar, nSpatialDims, numOfNDArraysToBeRead);
	XDATA_CERR("Done\n");
    }
    setApp(pCLapp, true);
}

/**
 * @brief Virtual copy "constructor"
 * @param[in] deepCopy copy stored data if true; only data structure if false (shallow copy)
 * @return A copy of this object
 */
std::shared_ptr<Data> XData::clone(bool deepCopy) const {
    return std::make_shared<XData>(pCLapp, shared_from_this(), deepCopy);
}

/**
 * @brief Virtual copy "constructor"
 * @param[in] deepCopy copy stored data if true; only data structure if false (shallow copy)
 * @return A copy of this object
 */
XData* XData::cloneHostOnly() const {
	// true => copy data from source Data host memory to destination Data host memory
	// false => not allocate device memory nor copy data from host to device in destination Data
    return new XData(pCLapp, shared_from_this(), true, false);
}

/**
 * @brief Virtual copy "constructor"
 * @param[in] newElementDataType data type of the newly created object. Shallow copy is implied.
 * @return An empty copy of this object that contains elements of type "newElementDataType"
 */
std::shared_ptr<Data> XData::clone(ElementDataType newElementDataType) const {
    return std::make_shared<XData>(pCLapp, shared_from_this(), newElementDataType);
}


/**
 * @brief Loads data from a group of files in raw format.
 *
 * Files must have a name with the format \<fileNamePrefix\>\<framesFileNameSuffix\>\<i\>\<fileNameExtension\>
 * where \<i\> is the number of frame of the image stored in one file.
 * @param[in] fileNamePrefix prefix common to names of all files to be read
 * @param[in,out] pArraysDims pointer to vector of vectors with spatial dimensions of every data to be read into a NDArray
 * @param[in,out] pDynDims pointer to vector with temporal dimensions
 * @param[in] framesFileNameSuffix suffix for file name part related to frames
 * @param[in] fileNameExtension extension common to all file names
 */
void XData::loadRawHostData(const std::string& fileNamePrefix, std::vector<std::vector< dimIndexType >*>*& pArraysDims,
			    std::vector <dimIndexType>*& pDynDims, const std::string& framesFileNameSuffix,
			    const std::string& fileNameExtension) {
    index1DType numFrames = 1;
    for(index1DType i = 0; i < pDynDims->size(); i++) {
	numFrames = numFrames * pDynDims->at(i);
    }

    std::stringstream variableSuffixStream;
    std::vector<std::string> fileNameSuffixes;
    if(numFrames > 1) {
	for(dimIndexType frame = 0; frame < numFrames; frame++) {
	    variableSuffixStream << framesFileNameSuffix << std::setfill('0') << std::setw(2) << frame ; // _frameFF
	    fileNameSuffixes.push_back(variableSuffixStream.str());
	    variableSuffixStream.str(""); // emtpy string, we don't want to accumulate strings among iterations
	}
    }
    std::string dataFileNamePrefixWithDims = OpenCLIPER::Data::buildFileNamePrefix(fileNamePrefix, pArraysDims->at(0));
    (reinterpret_cast<Data*>(this))->loadRawData(dataFileNamePrefixWithDims, pArraysDims, pDynDims, fileNameSuffixes,
	    fileNameExtension);
    XDATA_CERR("XData size: " << getData()->size() << std::endl);
}

/**
 * @brief Save data of NDArray objects to a group of files (every NDArray contains one image and is stored into a file).
 *
 * Files of this group have names with the format
 * \<fileNamePrefix\>\<dims\>\<framesFileNameSuffix\>\<frameNumber\>\<fileNameExtension\>,
 * where \<dims\> is a string with the format \<width\>x\<height\>x\<depth\> and \<frameNumber\> is the index
 * of the time frame of image adquisition (starting at 0).
 * @param[in] syncSource set the data source used for saving (IMAGE_ONLY or BUFFER_ONLY)
 * @param[in] fileNamePrefix fixed part of the file name before the variable part
 * @param[in] framesFileNameSuffix part of the file name before the frame number
 * @param[in] fileNameExtension extension common to all file names
 */
void XData::saveRawHostData(const SyncSource syncSource, const std::string& fileNamePrefix,
			    const std::string& framesFileNameSuffix,
			    const std::string& fileNameExtension) {
    dimIndexType numFrames = getDynDimsTotalSize();
    std::stringstream variableSuffixStream;
    std::vector<std::string> fileNameSuffixes;
    for(dimIndexType frame = 0; frame < numFrames; frame++) {
	variableSuffixStream << framesFileNameSuffix << std::setfill('0') << std::setw(2) << frame ; // _frameFF
	fileNameSuffixes.push_back(variableSuffixStream.str());
	variableSuffixStream.str(""); // emtpy string, we don't want to accumulate strings among iterations
    }

    this->device2Host();

    (reinterpret_cast<Data*>(this))->saveRawData(fileNamePrefix, fileNameSuffixes, fileNameExtension);
    XDATA_CERR("XData size: " << getData()->size() << std::endl);
}

/**
 * @brief Loads a group of images from files in standard image formats like png, jpeg, tiff, etc. (only formats supported by IL
 * library).
 * @param[in] fileNames vector of names for the files to be read data from
 * @throw std::invalid_argument if image cannot be loaded or images have more than 2 spatial dimensions or datatype of image elements is not supported
 */
void XData::load(const std::vector<std::string> &fileNames) {

    // NDArray collection to be stored in this XData
    auto nDArrays = new std::vector<NDArray*>;


    cv::Mat image;
    for(auto&& fileName : fileNames) {
	// Color images are nonsense to us. Have OpenCV convert color images to grayscale
	// Also, convert 16/32-bpp images to 8-bpp (OpenCV does this by default).
	image = cv::imread(fileName, cv::IMREAD_GRAYSCALE);
	if(image.data == NULL)
	    BTTHROW(std::invalid_argument("Unable to load image \"" + fileName + "\""), "XData::load");

	// Get image dimensions (guaranteed to be >=2)
	if(image.dims > 2)
	    BTTHROW(std::invalid_argument("Image files of more than 2 dimensions per file are unsupported at this time"), "XData::load");

	// Build vector of dimensions
	auto pDims = new std::vector<dimIndexType>(2);
	(*pDims)[WIDTHPOS] = image.cols;
	(*pDims)[HEIGHTPOS] = image.rows;
	index1DType numElements = image.cols * image.rows;

	// The NDArray (i.e. image in this case) we are currently processing
	NDArray* currentNDArray;
	// Create empty NDArray of appropriate size
	currentNDArray = NDArray::createNDArray(pDims, elementDataType);

	// Fill and normalize NDArray from image data
	if(elementDataType == TYPEID_COMPLEX) {
	    for(dimIndexType i = 0; i < numElements; i++) {
		static_cast<complexType*>(currentNDArray->getHostDataAsVoidPointer())[i].real(image.data[i] / 255.0);
		static_cast<complexType*>(currentNDArray->getHostDataAsVoidPointer())[i].imag(0.0);
	    }
	}
	else if(elementDataType == TYPEID_REAL) {
	    for(dimIndexType i = 0; i < numElements; i++) {
		static_cast<realType*>(currentNDArray->getHostDataAsVoidPointer())[i] = image.data[i] / 255.0;
	    }
	}
	else {
	    BTTHROW(std::invalid_argument("unsupported data type in XData::load"), "XData::load");
	}

	// add the loaded NDArray to this Data's collection of NDArrays
	nDArrays->push_back(currentNDArray);
    }

    // Set temporal dimension (number of images)
    auto pDynDims = new std::vector<dimIndexType>({ static_cast<dimIndexType>(fileNames.size()) });
    setDynDims(pDynDims);

    // Set data (contents of image sequence)
    setData(nDArrays);
}

/**
 * @brief Save 1 image in a file with name <i>fileName</i> (image format depending on file name extension,
 * only DevIL supported formats can be used)
 * @param[in] fileName name of the file to read data from
 * @param[in] syncSource device memory type field to read image data from (OpenCL buffer or image)
 */
void XData::save(const std::string& fileName) {
    std::vector<std::string> fileNames = {fileName};
    save(fileNames);
}

/**
 * @brief Save images in files with names obtained from <i>fileNames</i> vector (image format depending on file name extension,
 * only OpenCV supported formats can be used)
 * @param[in] fileNames vector of names for the files to read data from
 * @param[in] syncSource device memory type field to read image data from (OpenCL buffer or image)
 * @throw std::invalid_argument if number of filenames toes not match number of images or value of syncSource parameter is SyncSource::ALL or image data is empty
 * or element data type is not supported (only real or complex elements are valid) or file cannot be written
 */
void XData::save(const std::vector<std::string> &fileNames) {
    // Some sanity checks before attempting to save...
    std::ostringstream outputstream;
    if(fileNames.size() != getDynDimsTotalSize()) {
	outputstream << errorPrefix << "save: Number of filenames (" << fileNames.size() << ") does not match the number of images (" << getDynDimsTotalSize() << ") in this Data object.";
	BTTHROW(std::invalid_argument(outputstream.str()), "XData::save");
    }
    if(pNDArrays == nullptr) {
	outputstream << errorPrefix << "save: Image data empty";
	BTTHROW(std::invalid_argument(outputstream.str()), "XData::save");
    }
    if(getNumNDArrays() < 1) {
	outputstream << errorPrefix << "save: Image data empty";
	BTTHROW(std::invalid_argument(outputstream.str()), "XData::save");
    }

    // Temporary OpenCV matrix to save image data
    cv::Mat mat;

    // Traverse all images in this Data object
    dimIndexType nDims;
    for(dimIndexType currentNDArray = 0; currentNDArray < getDynDimsTotalSize(); currentNDArray++) {
	nDims = getData()->at(currentNDArray)->getNDims();

	// Can't save non-2D NDArrays as images
	if(nDims != 2) {
	    std::stringstream s;
	    s << errorPrefix << "save: NDArray " << currentNDArray << " is " << nDims << "D. Can only save 2D NDArrays as images. Skipping this NDArray...\n";
	    XDATA_CERR(s.str());
	    continue;
	}

	// This should never happen...
	assert(getData()->at(currentNDArray)->getDims() != nullptr);
	assert(getData()->at(currentNDArray)->getDims()->size() == 2);

	this->device2Host();

	// Create CV matrix of appropriate size. As for now, we only need unsigned 8 bits per pixel w/o alpha.
	dimIndexType width = NDARRAYWIDTH(getData()->at(currentNDArray));
	dimIndexType height = NDARRAYHEIGHT(getData()->at(currentNDArray));
	//index1DType numElements=width*height;
	mat = cv::Mat(width, height, CV_8U);	// old content is automatically de-referenced

	// Select source pointer from buffer or image data
	const void* pSource;
	pSource = getHostBuffer(currentNDArray);

	if(pSource == nullptr) {
	    std::stringstream s;
	    s << errorPrefix << "save: host data of NDArray " << currentNDArray << " is empty. Skipping this NDArray...\n";
	    XDATA_CERR(s.str());
	    continue;
	}

	// Convert source data from complex or real to uint8, normalize and store in OpenCV matrix
	if(elementDataType == TYPEID_COMPLEX) {
	    const complexType* pSourceComplex = static_cast<const complexType*>(pSource);

	    index1DType currentPixel = 0;
	    for(dimIndexType y = 0; y < height; y++)
		for(dimIndexType x = 0; x < width; x++)
		    mat.at<uchar>(y, x) = cv::saturate_cast<uchar> (255.0 * abs(pSourceComplex[currentPixel++]));
	}
	else if(elementDataType == TYPEID_REAL) {
	    const realType* pSourceReal = static_cast<const realType*>(pSource);

	    index1DType currentPixel = 0;
	    for(dimIndexType y = 0; y < height; y++)
		for(dimIndexType x = 0; x < width; x++)
		    mat.at<uchar>(y, x) = cv::saturate_cast<uchar> (255.0 * abs(pSourceReal[currentPixel++]));
	}
	else
	    BTTHROW(std::invalid_argument(std::string(errorPrefix) + "save: data type not supported"), "XData::save");

	// Write image file
	if(!cv::imwrite(fileNames[currentNDArray], mat))
	    BTTHROW(std::runtime_error(std::string(errorPrefix) + "save: error writing image file '" + fileNames[currentNDArray] + "'"), "XData::save");
    }
}

/**
 * @brief Saves a group of images (1 per frame) as a group of files with a common prefix name (<i>fileName</i>) and a
 * suffix name (<i>suffixName</i>) followed by an index (from 0 to number of frames -1).
 *
 * Image format depends on file name extension, only OpenCV supported formats can be used.
 *
 * @param[in] prefixName prefix name used for all the generated names
 * @param[in] suffixName string suffix added to the prefix file name
 * @param[in] syncSource device memory type field to read image data from (OpenCL buffer or image)
 */
void XData::save(const std::string& prefixName, const std::string& extension) {
    // Generate output file names (one file name per frame)
    std::vector<std::string> outputFileNames;
    std::string name;
    for(unsigned int i = 0; i < getDynDimsTotalSize(); i++) {
	name = "";
	name = name.append(prefixName);
	name.append(std::to_string(i));
	name.append(".");
	name.append(extension);
	outputFileNames.push_back(name);
    }
    save(outputFileNames);
}

void saveRawDataForThread(std::shared_ptr<Data> xData, std::string fileName) {
    xData->saveRawData(fileName);
    std::string headerFileName = LPISupport::Utils::basename(fileName) + ".hdr";
    std::dynamic_pointer_cast<XData>(xData)->saveCFLHeader(headerFileName);
    //delete(xData);
}

void XData::saveCFLData(const std::string &fileName, bool asyncSave) {
    device2Host();
    if (asyncSave) {
	dataForSaving = clone(true);
	XDATA_CERR("XData::saveCFLData(): myDataHandle="<<getHandle()<<"; dataForSaving->myDataHandle="<<dataForSaving->getHandle()<<"\n");
	pCLapp->host2Device(dataForSaving->myDataHandle);
	std::thread threadDataSave = std::thread(saveRawDataForThread, dataForSaving, fileName + ".cfl");

	// Necesario para que la nueva thread sea independiente de la original y pueda desaparecer, liberando
	// sus recursos, cuando acabe su función (aunque esto impide hacer un join desde la thread padre)
	threadDataSave.detach();
    } else {
	saveRawData(fileName + ".cfl");
	std::string headerFileName = LPISupport::Utils::basename(fileName) + ".hdr";
	saveCFLHeader(headerFileName);
    }
}

void XData::saveCFLHeader(const std::string &fileName) {
    std::fstream f;
    LPISupport::Utils::openFile(fileName, f, std::ofstream::out|std::ofstream::trunc, "XData::saveCFLHeader");
    const std::vector<dimIndexType>* pSpatialDims = getNDArray(0)->getDims(); // All NDArrays must have the same sptial dimensions
    for (uint i=0; i < pSpatialDims->size(); i++) {
	f << pSpatialDims->at(i) << " ";
    }
    f << "1 "; // 1 coil
    f << getDynDims()->at(0) << " "; // Only 1 temporal dimension supported
    f << "1";
    f << std::endl;
    f.close();
}
/**
 * @brief Stores image/volume spatial and temporal dimensions in DataDimsAndStrides field (DataDimsAndStrides data type is valid to be used for
 * kernel parameter).
 */
void XData::calcDataDims() {
    Data::calcDataDims();
    pDataDimsAndStridesVector->at(NumCoilsPos) = 0;
}

/**
 * @brief Saves data to a file in matlab format
 *
 * @param[in] fileName name of the file tha data will be saved to
 * @param[in] syncSource source of data: (OpenCL buffer or image)
 * @throw std::invalid_argument if matlab file cannot be written or image data is empty
 */
void XData::matlabSave(const std::string& fileName) {
    mat_t* matfp;
    matfp = Mat_CreateVer(fileName.c_str(), NULL, MAT_FT_DEFAULT);
    if(NULL == matfp) {
	BTTHROW(std::invalid_argument(std::string("Error creating MAT file")  + fileName), "XData::matlabSave");
    }

    this->device2Host();

    // Save dims variable
    if(pMatVarDimsData == nullptr) {
	// Create matlab Dims variable
	pMatVarDimsData.reset(new MatVarDimsData(pCLapp, getNDArray(0)->getDims(), getAllSizesEqual(), 0, getDynDims()));
    }
    pMatVarDimsData->matlabSaveVariable(matfp);

    // Saving image data
    // Only image sequences with the same spatial dimensions are supported
    auto dynDimsTotalSize = getDynDimsTotalSize();
    dimIndexType numMatVarElements = dynDimsTotalSize * pNDArrays->at(0)->size();
    if(NDARRAYWIDTH(pNDArrays->at(0)) == 0) {
	BTTHROW(std::invalid_argument("Invalid data size (width is 0)"), "XData::matlabSave");
    }

    MatVarInfo* pMatVarInfo = new MatVarInfo(elementDataType, numMatVarElements);
    // Update matlab dimensions and rank with spatial dimensions (number of coils not exists for XData)
    pMatVarInfo->updateDimsAndRank(pNDArrays->at(0)->getDims());
    // Update matlab dimensions and rank with temporal dimensions
    pMatVarInfo->updateDimsAndRank(getDynDims());
    fillMatlabVarInfo(matfp, matVarNameXData, pMatVarInfo);
    Mat_Close(matfp);
    delete(pMatVarInfo);
}

/**
 * @brief Creates a new XData object with data following a pattern (depending on typeOfGenData parameter).
 *
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] width width of the data (number of columns)
 * @param[in] height height of the data (number of rows)
 * @param[in] numFrames number of frames (number of arrays of size width x height)
 * @param[in] elementDataType Data type of vector elements stored in this object (default value is a complex type)
 * @param[in] typeOfGenData type of generated data (CONSTANT: same value for all the elements, SEQUENTIAL: consecutive odd values for
 * real type elements and for real part of complex elements and consecutive even values for elements imaginary part)
 * @return a pointer to the new XData object
 * @throw std::invalid_argument if element data type is not supported (only real or complex types are valid)
 */
XData* XData::genTestXData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, dimIndexType numFrames, ElementDataType elementDataType,
			   TypeOfGenData typeOfGenData) {
    XDATA_CERR("width: " << width << std::endl << "height: " << height << std::endl);
    std::vector<dimIndexType>* pDimsInputImage;
    std::vector<NDArray*>* pObjNDArraysImage = new std::vector<NDArray*>();
    std::vector<dimIndexType>* pInputDynDims = new std::vector<dimIndexType>({numFrames});
    realType realElement;
    if(typeOfGenData == CONSTANT) {
	realElement = 2.0;
    }
    else {
	realElement = 1.0;
    }
    if(elementDataType == TYPEID_REAL) {
	realType elementImage;
	for(dimIndexType dynId = 0; dynId < pInputDynDims->at(0); dynId++) {
	    std::vector <realType>* pImageData;
	    pImageData = new std::vector<realType>();
	    for(index1DType i = 0; i < (width * height) * 2; i += 2) {
		elementImage = realElement; // (1, 3, 5, 7, 9, ...)
		pImageData->push_back(elementImage);
		if(typeOfGenData != CONSTANT) {
		    realElement += 2.0;
		}
	    }
	    pDimsInputImage = new std::vector<dimIndexType>({width, height});
	    NDArray* pObjNDArrayImage = NDArray::createNDArray<realType>(pDimsInputImage, pImageData);
	    XDATA_CERR("Initial data" << std::endl);
	    XDATA_CERR("Objeto creado, ndims: " << (dimIndexType) pObjNDArrayImage->getNDims() << std::endl);
	    // pObjNDArrays only needed for Data constructor but now is abstract => can't be instantiated
	    // vector<NDArray*>* pObjNDArrays = new vector<NDArray*>;
	    // pObjNDArrays->push_back(pObjNDArray);
	    XDATA_CERR("Creando objeto de la clase XData" << std::endl);

	    pObjNDArraysImage->push_back(pObjNDArrayImage);
	}
	Data* pXData = (Data*) new XData(pCLapp, TYPEID_REAL);
	pXData->setData(pObjNDArraysImage);
	pXData->setDynDims(pInputDynDims);
	pCLapp->addData(pXData, true);
	return (static_cast<XData*>(pXData));
    }
    else if(elementDataType == TYPEID_COMPLEX) {
	complexType elementImage;
	for(dimIndexType dynId = 0; dynId < pInputDynDims->at(0); dynId++) {
	    std::vector <complexType>* pImageData;
	    pImageData = new std::vector<complexType>();
	    for(index1DType i = 0; i < (width * height) * 2; i += 2) {
		elementImage.real(realElement); // (1, 3, 5, 7, ...)
		elementImage.imag(realElement + 1); // (2, 4, 6, 8, ...)
		pImageData->push_back(elementImage);
		if(typeOfGenData != CONSTANT) {
		    realElement += 2.0;
		}
	    }
	    pDimsInputImage = new std::vector<dimIndexType>({width, height});
	    NDArray* pObjNDArrayImage = NDArray::createNDArray<complexType>(pDimsInputImage, pImageData);
	    XDATA_CERR("Initial data" << std::endl);
	    XDATA_CERR("Objeto creado, ndims: " << (numberOfDimensionsType) pObjNDArrayImage->getNDims() << std::endl);
	    // pObjNDArrays only needed for Data constructor but now is abstract => can't be instantiated
	    // vector<NDArray*>* pObjNDArrays = new vector<NDArray*>;
	    // pObjNDArrays->push_back(pObjNDArray);
	    XDATA_CERR("Creando objeto de la clase XData: " << dynId << std::endl);

	    pObjNDArraysImage->push_back(pObjNDArrayImage);
	}
	Data* pXData = new XData(pCLapp, TYPEID_COMPLEX);
	pXData->setData(pObjNDArraysImage);
	pXData->setDynDims(pInputDynDims);
	pCLapp->addData(pXData, true);
	delete(pObjNDArraysImage);
	delete(pInputDynDims);
	delete(pDimsInputImage);
	return (dynamic_cast<XData*>(pXData));
    }

    BTTHROW(std::invalid_argument("data type not supported for XData"), "XData::genTestXData");

    // we should never reach here. Just silence a warning (the compiler gets confused by the use of BTTHROW instead of plain throw.
    delete(pObjNDArraysImage);
    delete(pInputDynDims);
    return nullptr;
}

} // end namespace

#undef XDATA_DEBUG
