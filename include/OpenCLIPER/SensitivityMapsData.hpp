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
#ifndef SENSITIVITYMAPSDATA_HPP
#define SENSITIVITYMAPSDATA_HPP

#include <OpenCLIPER/defs.hpp>
#include <OpenCLIPER/Data.hpp>

namespace OpenCLIPER {
/**
 * @brief Class for storing data of sensitivity maps of the coils used for image adquisition.
 *
 * This class can be used to store a group of sensitivity maps, every map associated to one coil used for
 * image adquisition. Load and save formats include matlab .mat format and OpenCLIPER raw format.
 *
 * OpenCLIPER raw format is a binary format where complex numbers of the image data are stored as a pair of real type numbers (first real
 * part, then imaginary part) using C language convention. The order for storing elements of the image data array is by columns (first columns of
 * the same row, then rows, slices and coils).
 */
class SensitivityMapsData: public Data {
    public:
	/// @brief Name for variable inside matlab files (specific of SensitivityMapsData class)
	static constexpr const char* matVarNameSensitivityMaps =  "SensitivityMaps";

	explicit SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp);

	SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp, std::vector<std::vector<dimIndexType>*>*& pArraysDims, numCoilsType nCoils);
	SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp, std::vector<NDArray*>*& pData, numCoilsType nCoils);

	SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<SensitivityMapsData>& sourceData, bool copyData = false);
	SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<SensitivityMapsData>& sourceData, ElementDataType newElementDataType);

	SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp, const std::string& dataFileNamePrefix,
			    std::vector<std::vector< dimIndexType >*>*& pArraysDims,
			    numCoilsType numCoils, const std::string& coilsFileNameSuffix = "_coil",
			    const std::string& fileNameExtension = ".raw");
	SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp, const std::string &fileName, const std::vector<dimIndexType>* pArraySpatialDims,
			dimIndexType numCoils);
	SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp, matvar_t* pMatlabVar,
			const std::vector<dimIndexType>* pSpatialDimsFullySampled, dimIndexType nCoils);
	virtual ~SensitivityMapsData();

       	// Virtual "constructors"
	virtual std::shared_ptr<Data> clone(bool deepCopy) const override;
	virtual std::shared_ptr<Data> clone(ElementDataType newElementDataType) const override;

	// Inherit shared_ptr counter from base class
	std::shared_ptr<SensitivityMapsData> shared_from_this() {
	    return std::dynamic_pointer_cast<SensitivityMapsData>(Data::shared_from_this());
	}

	/**
	 * @brief Gets nCoils class field.
	 * @return the number of coils
	 */
	const numCoilsType getNCoils() const {
	    return nCoils;
	}

	/**
	 * @brief Sets nCoils class field.
	 * @param[in] nCoils pointer to new for number of available coils
	 */
	void setNCoils(numCoilsType nCoils) {
	    this->nCoils = nCoils;
	}

	void calcDataDims();

    protected:
	// Inherit shared_ptr counter from base class
	std::shared_ptr<SensitivityMapsData> shared_from_this() const {
	    return std::dynamic_pointer_cast<SensitivityMapsData> (std::const_pointer_cast<Data>(Data::shared_from_this()));
	}

    private:
	static constexpr const char* errorPrefix = "OpenCLIPER::SensitivityMapsData::";
	void loadRawHostData(const std::string &dataFileNamePrefix, std::vector<std::vector< dimIndexType >*>*& pArraysDims,
			     numCoilsType numCoils, const std::string &coilsFileNameSuffix = "_coil", const std::string &fileNameExtension = ".raw");
	void saveCFLHeader(const std::string &fileName);
	void checkMatVarDims(matvar_t* matvar, const std::vector<dimIndexType>* pSpatialDimsFullySampled, dimIndexType nCoils);
	/// Total number of available image coils
	numCoilsType nCoils = 1;
};
}
/* namespace OpenCLIPER */
#endif // SENSITIVITYMAPSDATA_HPP
