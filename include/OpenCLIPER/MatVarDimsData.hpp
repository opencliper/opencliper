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

#ifndef INCLUDE_OPENCLIPER_MATVARDIMSDATA_HPP_
#define INCLUDE_OPENCLIPER_MATVARDIMSDATA_HPP_

#include <OpenCLIPER/Data.hpp>

namespace OpenCLIPER {
class MatVarDimsData : public Data {
    public:
	enum dimsAndStridesKnownPos {
	    NSD_POS = 0, /// array position storing the number of spatial dimensions
	    ALLSIZESEQUAL_POS = 1, /// array position storing 1 if all NDArrays have the same size (still not supported)
	    NCOILS_POS = 2, /// array position storing number of coils (must be 0 for x-image data
	    NTD_POS = 3 /// array position storing the number of temporal dimensions
	};

	/// @brief Name for variable inside matlab files (specific of MatVardimsData class)
	static constexpr const char* matVarNameDims = "Dims";
	/**
	* @brief Deault constructor without parameters
	*/
	MatVarDimsData() {};

	MatVarDimsData(const std::shared_ptr<CLapp>& pCLapp, matvar_t* pMatlabVar);
	MatVarDimsData(const std::shared_ptr<CLapp>& pCLapp, const std::vector<dimIndexType>* pDataSpatialDims, dimIndexType allSizesEqual, numCoilsType nCoils,
		       const std::vector<dimIndexType>* pDataTemporalDims);

	MatVarDimsData(const std::shared_ptr<CLapp>&, const std::shared_ptr<MatVarDimsData>& sourceData, bool copyData);
	MatVarDimsData(const std::shared_ptr<CLapp>&, const std::shared_ptr<MatVarDimsData>& sourceData, ElementDataType newElementDataType);

	virtual ~MatVarDimsData() {};

	virtual std::shared_ptr<Data> clone(bool deepCopy) const override;
	virtual std::shared_ptr<Data> clone(ElementDataType newElementDataType) const override;

	void getAllDims(std::vector<dimIndexType>* pCompleteSpatialDimsVector, numCoilsType& numCoils, std::vector<dimIndexType>* pTemporalDimsVector);
	void matlabSaveVariable(mat_t* matfp);
	void calcDataDims() {
	    Data::calcDataDims();
	}

    protected:
	// Inherit shared_ptr counter from base class
	std::shared_ptr<MatVarDimsData> shared_from_this() const {
	    return std::dynamic_pointer_cast<MatVarDimsData> (std::const_pointer_cast<Data>(Data::shared_from_this()));
	}

    private:
	static constexpr const char* errorPrefix = "OpenCLIPER::Trajectories::";
};
} /* namespace OpenCLIPER */

#endif // INCLUDE_OPENCLIPER_MATVARDIMSDATA_HPP_
