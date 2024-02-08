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
#ifndef INCLUDE_OPENCLIPER_ConcreteNDArray_HPP_
#define INCLUDE_OPENCLIPER_ConcreteNDArray_HPP_

#include <memory>
#include <cstddef>
#include <iostream>
#include <OpenCLIPER/defs.hpp>
#include <OpenCLIPER/NDArray.hpp>

#undef LPICL_DEBUG
namespace OpenCLIPER {
//class NDArray;
/// class ConcreteNDArray n-dimensional array of \<T\> type data
template <class T>
class ConcreteNDArray : public NDArray {
	// Needed for accessing fields with write permission (get methods are read-only)
	/// see @ref Data
	friend class Data;
	// see @ref ComplexData
	//friend class ComplexData;
	/// see @ref KData
	friend class XData;
	/// see @ref KData
	friend class KData;
	/// see @ref NDArray
	friend class NDArray;

    public:
	ConcreteNDArray();
	// Don't remove this doxygen comments!! (needed here and in also in .cpp parameterized constructors, doxygen bug)
	/**
	 * @brief Constructor for storing spatial dimensions and empty data in class fields.
	 *
	 * This constructor has move semantics (in spite of not using && notation):
	 * after call, parameters memory deallocation is responsibility of this class
	 * (parameters are set to nullptr at the end of the method).
	 * @param[in,out] pSpatialDims vector with sizes of each spatial dimension
	 */
	explicit ConcreteNDArray(std::vector<dimIndexType>*& pSpatialDims);
	ConcreteNDArray(std::vector<dimIndexType>*& pSpatialDims, std::vector<T>*& pHostData);
	ConcreteNDArray(const std::string &completeFileName, std::vector<dimIndexType>*& pSpatialDims);
    ConcreteNDArray(std::fstream &f, std::vector<dimIndexType>*& pSpatialDims);

	// Don't remove this doxygen comments!! (needed here and in also in .cpp parameterized constructors, doxygen bug)
	/**
	 * @brief Constructor that creates a copy of a ConcreteNDArray object (dimensions are copied always,
	 * data only if copyData parameter is true).
	 * @param[in] pSourceData ConcreteNDArray object source of spatial and temporal dimensions
	 * @param[in] copyData data (not only dimensions) are copied if this parameter is true (default value: false)
	 */
	ConcreteNDArray(const NDArray* pSourceData, bool copyData = false);
	ConcreteNDArray(matvar_t* matvar, dimIndexType numOfSpatialDims, dimIndexType nDArrayOffsetInElements);
	~ConcreteNDArray();

	// Getters

	/**
	 * @brief Returns the pointer to data in host memory as a void pointer
	 * @return a void pointer to data stored as a vector in host memory
	 */
	virtual void* getHostDataAsVoidPointer() const {
	    return pHostData.get()->data();
	}

	/**
	 * @brief Gets pointer to data stored in host memory (vector of elements)
	 * @return raw pointer to data stored in host
	 */
	const std::vector<T>* getHostData() const {
	    return pHostData.get();
	}

    protected:
	std::vector <dimIndexType>* calcUnaligned1DArrayStridesFromNDArrayDims() const ;
	void loadMatlabHostDataElement(matvar_t* matvar, dimIndexType offsetInBytes);
	/**
	 * @brief Sets hostData field. Uses move semantics.
	 * @param[in,out] pHostData reference to pointer to vector of \<T\> type data
	 */
	void setHostData(std::vector<T>*& pHostData) {
	    // gets ownership of pHostData, releases owned poiner
	    this->pHostData.reset(pHostData);
	    // set original pointer to null (release does not do it automatically)
	    pHostData = nullptr;
	}

    private:

	const std::string elementToString(const void* elementsArray, dimIndexType index1D) const;
	// Attributes

	/** Data in host memory as a vector of \<T\> type elements */
	std::unique_ptr<std::vector<T>> pHostData = std::unique_ptr<std::vector<T>>(new std::vector<T>());
};
}
#endif
