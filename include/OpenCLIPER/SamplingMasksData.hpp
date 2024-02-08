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
#ifndef INCLUDE_OPENCLIPER_SAMPLINGMASKSDATA_HPP_
#define INCLUDE_OPENCLIPER_SAMPLINGMASKSDATA_HPP_

#include <OpenCLIPER/defs.hpp>
#include <OpenCLIPER/Data.hpp>
#include <OpenCLIPER/CLapp.hpp>

namespace OpenCLIPER {
/**
 * @brief Class for storing data of sampling masks used for image adquisition (number of masks is equal to the
 * number of different time frames in which images have been captured).
 *
 * The format used for external sources of sampling masks information (external format) for every sampling mask is an
 * array with a number of elements equal to the number of coil supported capture lines, and each element stores a 1
 * if element index is equal to the number of a line captured and 0 if element index is equal to the number of a line
 * not captured.
 *
 * The internal format used for every sampling mask is an array containing the indexes of the image lines not captured in
 * its time frame.
 *
 * External format example for a image of 8 lines: {1, 0, 1, 1, 0, 0, 0, 1}, corresponding internal format {1, 4, 5, 6}.
 *
 * OpenCLIPER raw format is a binary format where integer numbers of the sampling mask (external format) are stored using
 * C language convention. The order for storing elements of the image data array is by columns (first columns of the same
 * row, then rows, slices and temporal frames).
 *
 */
class SamplingMasksData: public Data {
	// Needed for accessing fields with write permission (get methods are read-only)
	/// @see KData
	friend class KData;
    public:
	/// @brief Name for variables inside matlab files (specific of SamplingMasksData class)
	static constexpr const char* matVarNameSamplingMasks =  "SamplingMasks";

	/// Format of sampling masks
	enum MasksFormat {
	    /// External old format: list of numbers 0 for a line not captured and 1 for captured (list size is number of lines of image)
	    ROWMASK,
	    /// Internal old format: list of numbers of rows not captured (must be blanked in reconstructed images)
	    ROWSTOBLANK,
	    /// New format: list of numbers 0 for a pixel not captured and 1 for captured (list size is number of pixels of image)
	    PIXELMASK
	};
	// Constructors
	SamplingMasksData(const std::shared_ptr<CLapp>& pCLapp, Data*& pMasks, dimIndexType kDataNumCols,
		ElementDataType elementDataTye = TYPEID_INDEX);
	SamplingMasksData(const std::shared_ptr<CLapp>& pCLapp, std::vector<NDArray*>*& pData,
		std::vector<dimIndexType>*& pDynDims, dimIndexType kDataNumCols, ElementDataType elementDataTye = TYPEID_INDEX);
	SamplingMasksData(const std::shared_ptr<CLapp>& pCLapp, const std::string& dataFileNamePrefix,
			  std::vector<std::vector< dimIndexType >*>*& pArraysDims, std::vector <dimIndexType>*& pDynDims,
			  dimIndexType kDataNumCols, const std::string &framesFileNameSuffix = "_frame",
			  const std::string &fileNameExtension = ".raw", ElementDataType elementDataTye = TYPEID_INDEX);
	SamplingMasksData(const std::shared_ptr<CLapp>& pCLapp, const std::string &fileName,
			const std::vector<dimIndexType>* pArraySpatialDims, const std::vector <dimIndexType>* pTemporalDims, dimIndexType kDataNumCols,
			ElementDataType elementDataType);
	SamplingMasksData(const std::shared_ptr<CLapp>& pCLapp, matvar_t* pMatlabVar,
		const std::vector<dimIndexType>* pKDataSpatialDimensions);
	SamplingMasksData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<SamplingMasksData>& sourceData, bool copyData = false);
	SamplingMasksData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<SamplingMasksData>& sourceData, ElementDataType newElementDataType);

	virtual ~SamplingMasksData();

       	// Virtual "constructors"
	virtual std::shared_ptr<Data> clone(bool deepCopy) const override;
	virtual std::shared_ptr<Data> clone(ElementDataType newElementDataType) const override;

	// Getters
	/**
	 * @brief Gets pointer to sampling masks data
	 *
	 * @return pointer to sampling masks data
	 */
	const Data* getMasks() const {
	    return pMasks.get();
	}

	/**
	 * @brief Gets the masksFormat class variable
	 *
	 * @return the value of the masksFormat class variable
	 */
	MasksFormat getMasksFormat() const {
	    return masksFormat;
	}

	/**
	 * @brief Gets the masksFormat class variable
	 *
	 * @return the value of the masksFormat class variable
	 */
	dimIndexType getKDataNumCols() {
	    return kDataNumCols;
	}

    protected:
	// Inherit shared_ptr counter from base class
	std::shared_ptr<SamplingMasksData> shared_from_this() const {
	    return std::dynamic_pointer_cast<SamplingMasksData> (std::const_pointer_cast<Data>(Data::shared_from_this()));
	}

	void rowmask2rowstoblank();
	void rowstoblank2rowmask();
	void rowmask2pixelmask();

    private:
	static constexpr const char* errorPrefix = "OpenCLIPER::SamplingMasksData::";

	// Setters
	/**
	 * @brief Sets the pMasks (pointer to vector of sampling masks data) class variable
	 * @param[in] pMasks pointer to new vector of sampling masks data
	 */
	void setMasks(Data*& pMasks) {
	    this->pMasks.reset(pMasks);
	    pMasks = nullptr;
	}

	/**
	 * @brief
	 * Stores image/volume spatial and temporal dimensions in class field.
	 */
	void calcDataDims() {
	    Data::calcDataDims();
	}

	void loadRawHostData(const std::string &fileNamePrefix, std::vector<std::vector< dimIndexType >*>*& pArraysDims,
			     std::vector <dimIndexType>*& pDynDims, const std::string &framesFileNameSuffix = "_frame",
			     const std::string &fileNameExtension = ".raw");
	void saveCFLHeader(const std::string &fileName);
	/// Pointer to vector of Data objects (every object contains information of a sampling mask for one time frame)
	std::unique_ptr<Data> pMasks = nullptr;
	/// Vector with the number of every line that has to be blanked
	std::vector<dimIndexType>* pRowNumbersToBeBlankedVector = {};
	/// Vector of booleans for lines that have to be read (true if line whose index is equal to the vector element index must be read)
	std::vector<bool> mask;
	/// Number of lines of a image (the same for all captured images). Needed to convert from internal format (list of indexes of image lines to be blanked) back
	/// to external format (a list of integers with value 0 if line has to be blanked, 1 otherwise, list size is the number of image lines)
	dimIndexType numberOfImageLines = 0;
	/// number of columns of KData images
	dimIndexType kDataNumCols = 0;
	/// Format of stored sampling masks
	MasksFormat masksFormat = ROWMASK;
};
}/* namespace OpenCLIPER */
#endif
