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
template ConcreteNDArray<complexType>::ConcreteNDArray();
template ConcreteNDArray<dimIndexType>::ConcreteNDArray();
template ConcreteNDArray<realType>::ConcreteNDArray();
template ConcreteNDArray<cl_uchar>::ConcreteNDArray();

template ConcreteNDArray<complexType>::ConcreteNDArray(std::vector<dimIndexType>*& pSpatialDims, std::vector<complexType>*& pHostData);
template ConcreteNDArray<dimIndexType>::ConcreteNDArray(std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pHostData);
template ConcreteNDArray<realType>::ConcreteNDArray(std::vector<dimIndexType>*& pSpatialDims, std::vector<realType>*& pHostData);
template ConcreteNDArray<cl_uchar>::ConcreteNDArray(std::vector<dimIndexType>*& pSpatialDims, std::vector<cl_uchar>*& pHostData);

template ConcreteNDArray<complexType>::ConcreteNDArray(const std::string &completeFileName,
	std::vector<dimIndexType>*& pSpatialDims);
template ConcreteNDArray<dimIndexType>::ConcreteNDArray(const std::string &completeFileName,
	std::vector<dimIndexType>*& pSpatialDims);
template ConcreteNDArray<realType>::ConcreteNDArray(const std::string &completeFileName,
	std::vector<dimIndexType>*& pSpatialDims);
template ConcreteNDArray<cl_uchar>::ConcreteNDArray(const std::string &completeFileName,
	std::vector<dimIndexType>*& pSpatialDims);

template ConcreteNDArray<complexType>::ConcreteNDArray(std::fstream &f, std::vector<dimIndexType>*& pSpatialDims);
template ConcreteNDArray<dimIndexType>::ConcreteNDArray(std::fstream &f, std::vector<dimIndexType>*& pSpatialDims);
template ConcreteNDArray<realType>::ConcreteNDArray(std::fstream &f, std::vector<dimIndexType>*& pSpatialDims);
template ConcreteNDArray<cl_uchar>::ConcreteNDArray(std::fstream &f, std::vector<dimIndexType>*& pSpatialDims);

template ConcreteNDArray<complexType>::ConcreteNDArray(matvar_t* matvar, dimIndexType numOfSpatialDims, dimIndexType nDArrayOffsetInElements);
template ConcreteNDArray<dimIndexType>::ConcreteNDArray(matvar_t* matvar, dimIndexType numOfSpatialDims, dimIndexType nDArrayOffsetInElements);
template ConcreteNDArray<realType>::ConcreteNDArray(matvar_t* matvar, dimIndexType numOfSpatialDims, dimIndexType nDArrayOffsetInElements);
template ConcreteNDArray<cl_uchar>::ConcreteNDArray(matvar_t* matvar, dimIndexType numOfSpatialDims, dimIndexType nDArrayOffsetInElements);

template ConcreteNDArray<complexType>::~ConcreteNDArray();
template ConcreteNDArray<dimIndexType>::~ConcreteNDArray();
template ConcreteNDArray<realType>::~ConcreteNDArray();
template ConcreteNDArray<cl_uchar>::~ConcreteNDArray();

template const std::string ConcreteNDArray<complexType>::elementToString(const void* pElementsArray, dimIndexType index1D) const;
template const std::string ConcreteNDArray<dimIndexType>::elementToString(const void* pElementsArray, dimIndexType index1D) const;
template const std::string ConcreteNDArray<realType>::elementToString(const void* pElementsArray, dimIndexType index1D) const;
template const std::string ConcreteNDArray<cl_uchar>::elementToString(const void* pElementsArray, dimIndexType index1D) const;
