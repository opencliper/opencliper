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
// Avoiding compiler errors due to multiple include of header files
#ifndef INCLUDE_OPENCLIPER_NDARRAY_HPP_
#define INCLUDE_OPENCLIPER_NDARRAY_HPP_

#include <memory>
#include <cstddef>
#include <iostream>
#include <map>
#include <OpenCLIPER/defs.hpp>
// #include <LPISupport/Utils.hpp>
#include <typeinfo>
#include <OpenCLIPER/MatVarInfo.hpp>

namespace OpenCLIPER {
/// @brief class NDArray - n-dimensional matrix of data (abstract class, data type of data elements
/// is specific of subclasses).
class NDArray {
	// Needed for accessing fields with write permission (get methods are read-only)
	/// see @ref Data
	friend class Data;
	/// see @ref XData
	friend class XData;
	/// see @ref KData
	friend class KData;
	/// see @ref DeviceDataProperties
	friend class DeviceDataProperties;
	/// see @ref MatVarDimsData
	friend class MatVarDimsData;
    public:
	// Constructors and destructor
	NDArray();
	virtual ~NDArray();

	static NDArray* createNDArray(const std::string &completeFileName, std::vector<dimIndexType>*& pSpatialDims, ElementDataType elementDataType);
	static NDArray* createNDArray(std::fstream &f, std::vector<dimIndexType>*& pSpatialDims, ElementDataType elementDataType);
	static NDArray* createNDArray(const NDArray* pSourceData, bool copyData, ElementDataType elementDataType);
	static NDArray* createNDArray(const void* pSourceData, std::vector<dimIndexType>*& pSpatialDims, ElementDataType elementDataType);
	static NDArray* createNDArray(matvar_t* matvar, dimIndexType numOfSpatialDims, dimIndexType nDArrayOffsetInElements);
	static NDArray* createNDArray(std::vector<dimIndexType>*& pSpatialDims, ElementDataType elementDataType);
	template<typename T>
	static NDArray* createNDArray(std::vector<dimIndexType>*& pSpatialDims, std::vector<T>*& pData);

	template<typename T>
	static NDArray* createNDArray(std::vector<T>*& pData);

	// Getters
	const numberOfDimensionsType getNDims() const;
	static dimIndexType          getElementSize(ElementDataType elementDataType);
	const index1DType            size() const;

	/**
	* @brief Gets vector with dimensions of stored data (vector size is number of dimensions).
	* @return vector with dimensions of stored data
	*/
	const std::vector<dimIndexType>* getDims() const {
	    return (pDims.get());
	}


	// Data displaying
	const std::string hostDataToString(std::string title) const;
	const std::string dimsToString(std::string title) const;

    protected:
	/**
	 * @brief Calculates strides for accesing the NDArray as a contigouos memory data (using NDArray spatial dimensions).
	 */
	virtual std::vector<dimIndexType>* calcUnaligned1DArrayStridesFromNDArrayDims() const = 0;

	/**
	* @brief Gets pointer to data stored in host memory (vector of elements) as a void pointer.
	* @return raw pointer to data stored in host memory
	*/
	virtual void* getHostDataAsVoidPointer() const = 0;

	/**
	* @brief Sets pDims (spatial dimensions) field. Use move semantics, parameter value will be nullptr after executing this method.
	* @param[in,out] pDims reference to pointer to new vector with data spatial dimensions
	*/

	void setDims(std::vector<dimIndexType>*& pDims) {
	    this->pDims.reset(pDims);
	    pDims = nullptr;
	}

	/**
	 * @brief Loads a data element from a matlab variable (previously read from a matlab file).
	 * @param[in] matvar matlab variable
	 * @param[in] offset offset to start reading from
	 */
	virtual void loadMatlabHostDataElement(matvar_t* matvar, dimIndexType offset) = 0;

	void loadMatlabHostData(matvar_t* matvar, dimIndexType numOfSpatialDims, dimIndexType nDArrayOffsetInElements);

	DataHandle getDataHandle() const{
		return myDataHandle;
	}

	void setDataHandle(DataHandle dataHandle) {
		this->myDataHandle = dataHandle;
	}

    private:
	const std::string nDArrayElementsToString(std::string title, const void* pArrayElements) const;
	/**
	* @brief Pure virtual method that converts one element of an array of data to a text representation (representation depends on the data type of
	* base element so this method must be defined in NDArray subclasses).
	* @param[in] pElementsArray pointer to array of elements
	* @param[in] index1D 1-dimensional index for element from array
	* @return string with text representation of data element
	*/
	virtual const std::string elementToString(const void* pElementsArray, dimIndexType index1D) const = 0;

	// Attributes

	/** @brief vector with the size of each data spatial dimension. */
	std::unique_ptr<std::vector<dimIndexType>> pDims = nullptr;

	// Data handle of Data that contains this NDArray
	DataHandle myDataHandle = INVALIDDATAHANDLE;

};
}
#endif
