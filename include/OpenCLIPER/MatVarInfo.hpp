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
#ifndef INCLUDE_OPENCLIPER_MATVARINFO_HPP_
#define INCLUDE_OPENCLIPER_MATVARINFO_HPP_

#include<OpenCLIPER/defs.hpp>
#include <typeinfo>
#include <map>
#include <matio.h>
//#include<OpenCLIPER/NDArray.hpp>
namespace OpenCLIPER {
/**
 * @brief Class that stores matlab data and its meta-data (matlab class type, matlab data type, etc.)
 *
 */
class NDArray;
class MatVarInfo {
    public:
	/// Enum with keys for matlabStrides map.
	enum matlabStridesKeys {
	    /// key for the stride between two consecutive data elements (number of stored real element values between them:
	    /// 2 for data elements of complex type, 1 otherwise)
	    dataElement,
	    /// key for the stride between consecutive columns (number of stored real element values between them)
	    column,
	    /// key for the stride between consecutive rows (number of stored real element values between them)
	    row,
	    /// key for the stride between consecutive slices (number of stored real element values between them)
	    sliceAndBeyond
	};

	MatVarInfo(enum matio_classes class_type, enum matio_types data_type, void* data, int opt);
	MatVarInfo(ElementDataType elementDataType, dimIndexType numOfElements);
	~MatVarInfo();

	/**
	 * @brief Gets class_type class variable
	 *
	 * @return value class_type class variable
	 */
	enum matio_classes getClassType() {
	    return class_type;
	}

	/**
	 * @brief Gets data_type class variable
	 *
	 * @return value data_type class variable
	 */
	enum matio_types getDataType() {
	    return data_type;
	}

	/**
	 * @brief Gets matlab rank (number of dimensions of a matrix)
	 *
	 * @return value of matlab rank.
	 */
	unsigned int getRank() {
	    return rank;
	}

	size_t* getDims();

	/**
	 * @brief Gets pointer to data in matlab format
	 *
	 * @return pointer to data in matlab format
	 */
	void* getData() {
	    return data;
	}

	/**
	 * @brief Gets matlab opt class variable
	 *
	 * @return the value of matlab opt class variable
	 */
	int getOpt() {
	    return opt;
	}

	/**
	 * @brief Get matlab stride by key from matlabStrides field (strides are used for accessing elements inside matlab array).
	 * @param[in] key key of the stride to be got
	 * @return stride corresponding to the key parameter value
	 */
	dimIndexType getMatlabStride(matlabStridesKeys key) {
	    return matlabStrides[key];
	}

	void set(const dimIndexType* pDimsAndStridesInfo, dimIndexType nDArrayOffsetInElements, const NDArray* pNDArray,
		const void* pNDArrayData);
	void updateDimsAndRank(const std::vector<dimIndexType>* pDimsVectorArg);
    private:
	void dumpMatlabElement(dimIndexType matlabElementOffset, dimIndexType nDArrayElementOffset, const void* pNDArrayData);
	void commonInit(enum matio_classes class_type, enum matio_types data_type, void* data, int opt);

	/// Data type of stored data (delete a void pointer is not allowed, destructor must know the type of stored data)
	ElementDataType elementDataType = std::type_index(typeid(void));

	/// enumerated value storing matlab class type
	enum matio_classes class_type;

	/// enumerated value storing matlab data type
	enum matio_types data_type;

	/// matlab rank. Note: rank is a signed int in matio.h, but a negative number of dimensions makes no sense.
	/// Since we do comparisons with unsigned variables all the time, we store it as an unsigned
	unsigned int rank = 0;

	/// vector with dimensions of array
	std::vector<size_t>* pDimsVector = new std::vector<size_t>();

	/// array data in matlab format
	void* data;

	/// matlab options
	int opt;

	/// Vector for real parts of complex numbers
	realType* realPart;
	/// Vector for imaginary parts of complex numbers
	realType* imagPart;

	/** @brief Map of strides for accessing matlab variables with 1D, 2D or 3D data. */
	std::map<matlabStridesKeys, dimIndexType> matlabStrides;
};
}
#endif // INCLUDE_OPENCLIPER_MATVARINFO_HPP_
