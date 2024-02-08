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

#ifndef XDATA_HPP
#define XDATA_HPP

#include <OpenCLIPER/defs.hpp>
#include <OpenCLIPER/Data.hpp>
#include <OpenCLIPER/KData.hpp>
#include <OpenCLIPER/CLapp.hpp>

#include <assert.h>

namespace OpenCLIPER {
/**
 * @brief Class for storing data of images in the x-space.
 *
 * This class can be used to store a group of related images in the x-space, every image captured in a different time frame.
 * Load and save formats include standard image formats supported by DevIL library (png, jpeg, gif, tiff, etc., only one image per file),
 * matlab .mat format (one or several images per file) and OpenCLIPER raw format.
 *
 * OpenCLIPER raw format is a binary format where complex numbers of the image data are stored as a pair of real type numbers (first real
 * part, then imaginary part) using C language convention. The order for storing elements of the image data array is by columns (first columns of
 * the same row, then rows, slices and temporal dimensions by definition order).
 */
class XData: public Data {
    public:

	/// @brief Name for variable inside matlab files (specific of XData class)
	static constexpr const char* matVarNameXData = "XData";

	/// Enumeration with the options for generating automatically data values for an XData object
	enum TypeOfGenData {
	    /// Constant values
	    CONSTANT,
	    /// Consecutive values
	    SEQUENTIAL
	};

	//*********************************
	// Constructors and destructor
	//*********************************

	// Empty data
	XData(const std::shared_ptr<CLapp>& pCLapp,
	      ElementDataType elementDataType = TYPEID_COMPLEX);


	// From a bunch of NDArrays
	XData(const std::shared_ptr<CLapp>& pCLapp, NDArray*& pData,
	      ElementDataType elementDataType = TYPEID_COMPLEX);

	XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<NDArray*>*& pData, std::vector<dimIndexType>*& pTempDims,
	      ElementDataType elementDataType = TYPEID_COMPLEX);


	// From given dimensions, uninitialized: single NDArray
	XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width,
	      ElementDataType elementDataType = TYPEID_COMPLEX);

	XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height,
	      ElementDataType elementDataType = TYPEID_COMPLEX);

	XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, dimIndexType depth,
	      ElementDataType elementDataType = TYPEID_COMPLEX);

	XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<dimIndexType>*& pSpatialDims,
	      ElementDataType elementDataType = TYPEID_COMPLEX);


	// From given dimensions, uninitialized: several NDArrays with no temporal dimensions
	XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType numNDArrays, std::vector<dimIndexType>*& pSpatialDims,
	      ElementDataType elementDataType = TYPEID_COMPLEX);

	XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<std::vector<dimIndexType>*>*& pSpatialDims,
	      ElementDataType elementDataType = TYPEID_COMPLEX);


	// From given dimensions, uninitialized: several NDArrays with spatial and temporal dimensions
	XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType numNDArrays, std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims,
	      ElementDataType elementDataType = TYPEID_COMPLEX);

	XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims,
	      ElementDataType elementDataType = TYPEID_COMPLEX, bool initZero = false);


        // From given dimensions and an std::vector: single NDArray
        template<typename T>
        XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, std::vector<T>*& pData);

        template<typename T>
	XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, std::vector<T>*& pData);

        template<typename T>
	XData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, dimIndexType depth, std::vector<T>*& pData);

        template<typename T>
	XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<dimIndexType>*& pSpatialDims, std::vector<T>*& pData);

	// From given dimensions and a vector of std::vectors: severall NDArrays with spatial and temporal dimensions
        template<typename T>
	XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<T>*>*& pData);

        template<typename T>
	XData(const std::shared_ptr<CLapp>& pCLapp, std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<T>*>*& pData);

        // From another Data or Data-derived object (copy contructors)
	XData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data>& sourceData, bool copyData, bool copyDataToDevice = true);
	XData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data>& sourceData, ElementDataType newElementDataType);
	XData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<XData>& sourceData, bool copyData, bool copyDataToDevice = true);
	XData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<XData>& sourceData, ElementDataType newElementDataType);
	XData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<KData>& sourceData);

	// From the filesystem
	//XData(const std::shared_ptr<CLapp>& pCLapp, const std::string matlabFileName);
	XData(const std::shared_ptr<CLapp>& pCLapp, const std::string& fileName, ElementDataType elementDataType = TYPEID_COMPLEX);
	XData(const std::shared_ptr<CLapp>& pCLapp, const std::vector<std::string> &fileNames, ElementDataType elementDataType = TYPEID_COMPLEX);
	XData(const std::shared_ptr<CLapp>& pCLapp, const std::string& dataFileNamePrefix, std::vector<std::vector< dimIndexType >*>*& pArraysDims, std::vector <dimIndexType>*& pDynDims,
	      const std::string& framesFileNameSuffix = "_frame", const std::string& fileNameExtension = ".raw", ElementDataType elementDataType = TYPEID_COMPLEX);

	virtual ~XData() {}

	// Virtual "constructors"
	virtual std::shared_ptr<Data> clone(bool deepCopy) const override;
	virtual XData* cloneHostOnly() const;
	virtual std::shared_ptr<Data> clone(ElementDataType newElementDataType) const override;

	// Getters

	// Setters


	// Other methods
	void saveRawHostData(const SyncSource syncSource, const std::string& fileNamePrefix,
			     const std::string& framesFileNameSuffix = "_frame",
			     const std::string& fileNameExtension = ".raw");
	void save(const std::string& fileName);
	void save(const std::vector<std::string> &fileNames);
	void save(const std::string& prefixName, const std::string& extension);
	void saveCFLData(const std::string &fileName, bool asyncSave = false);
	void saveCFLHeader(const std::string &fileName);

	void calcDataDims();
	// Save in matlab file
	void matlabSave(const std::string& fileName);
	static XData* genTestXData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, dimIndexType numFrames, ElementDataType elementDataType,
				   TypeOfGenData typeOfGenData = CONSTANT);

    protected:
	// Inherit shared_ptr counter from base class
	std::shared_ptr<XData> shared_from_this() const {
	    return std::dynamic_pointer_cast<XData> (std::const_pointer_cast<Data>(Data::shared_from_this()));
	}

    private:
	void load(const std::vector<std::string> &fileNames);
	void loadRawHostData(const std::string& fileNamePrefix, std::vector<std::vector< dimIndexType >*>*& pArraysDims,
			     std::vector <dimIndexType>*& pDynDims,
			     const std::string& framesFileNameSuffix = "_frame", const std::string& fileNameExtension = ".raw");

	// Attributes
	static constexpr const char* errorPrefix = "OpenCLIPER::XData::";

	/** @brief Pointer to object containing dimensions get from matlab Dims variable */
	std::unique_ptr<MatVarDimsData> pMatVarDimsData = nullptr;
};
}
#endif // XDATA_HPP
