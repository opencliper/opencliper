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
#ifndef KDATA_HPP
#define KDATA_HPP

#include <OpenCLIPER/defs.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <OpenCLIPER/TrajType.hpp>
#include <OpenCLIPER/SensitivityMapsData.hpp>
#include <OpenCLIPER/SensitivityMapsRMS.hpp>
#include <OpenCLIPER/SamplingMasksData.hpp>
#include <OpenCLIPER/Trajectories.hpp>
#include <OpenCLIPER/Data.hpp>
#include <type_traits>
#include <OpenCLIPER/MatVarDimsData.hpp>
#include <OpenCLIPER/processes/FFT.hpp>
#include <OpenCLIPER/processes/ComplexElementProd.hpp>
#include <OpenCLIPER/processes/XImageSum.hpp>

namespace OpenCLIPER {
// Forward reference (XData is used in xImageSum method)
class XData;

/**
 * @brief Class for storing data of images in the k-space.
 *
 * This class can be used to store a group of related images in the k-space, every image captured in a different time frame
 * and by a different coil. Load and save formats include matlab .mat format (one or several images per file) and
 * OpenCLIPER raw format.
 *
 * OpenCLIPER raw format is a binary format where complex numbers of the image data are stored as a pair of real type numbers (first real
 * part, then imaginary part) using C language convention. The order for storing elements of the image data array is by columns (first columns of
 * the same row, then rows, slices, coils and temporal dimensions by definition order).
 */
class KData: public Data {

    public:
	/// @brief Name for variable inside matlab files (specific of KData class)
	static constexpr const char* matVarNameKData = "KData";

	/// @brief Enumerated type for selecting extra data to be loaded with KData (sensitivity maps and/or sampling masks)
	enum DataToLoad {
	    /// Load no extra data
	    LOADNONE = 0,
	    /// Load sensitivity maps
	    LOADSENSITIVITYMAPS = (1u << 0),
	    /// Load sampling masks
	    LOADSAMPLINGMASKS = (1u << 1)
	};

	/**
	 * @brief Position of prefix string in vector with string prefixes used for loading senstivity maps and sampling masks from
	 * files
	 */
	enum NamePrefixesPos {
	    /// Position of the sensitivity maps name prefix in the vector of prefix names
	    SENSITIVITYMAPSPREFIX = 0,
	    /// Position of the sampling masks name prefix in the vector of prefix names
	    SAMPLINGMASKSPREFIX = 1
	};

	//*********************************
	// Constructors and destructor
	//*********************************

	// Empty KData
	explicit KData(const std::shared_ptr<CLapp>& pCLapp);

	// From given dimensions (uninitialized)
	KData(const std::shared_ptr< CLapp >& pCLapp, std::vector<std::vector< dimIndexType >* >*& pArraysDims, numCoilsType nCoils, std::vector< dimIndexType >*& pDynDims);

	// From a bunch of NDArrays, sensitivity maps trajectories, and so on
	KData(const std::shared_ptr<CLapp>& pCLapp, SensitivityMapsData*& pSensitivityMapsData, std::vector<NDArray*>*& pNDArrays,
	      std::vector<realType>*& pCoord, numCoilsType nCoils, usedCoilsType usedCoils, std::vector<dimIndexType>*& pDynDims,
	      enum TrajType trajectory, std::vector<realType>*& pDcf, std::vector<realType>*& pDeltaK);
	KData(const std::shared_ptr< CLapp >& pCLapp, SensitivityMapsData*& pSensitivityMapsData,
	      std::vector<OpenCLIPER::NDArray* >*& pNDArrays, std::vector< float >*& pCoord, numCoilsType nCoils, usedCoilsType usedCoils, std::vector <
	      dimIndexType > *& pDynDims, TrajType trajectory, std::vector< float >*& pDcf, std::vector< float >*& pDeltaK, SyncSource
	      host2DeviceSync);

	// From another KData object
	KData(const std::shared_ptr< CLapp >& pCLapp, const std::shared_ptr<KData>& sourceData, bool copyData = false, bool copySensitivityMaps = false, bool copySamplingMasks = false);
	KData(const std::shared_ptr< CLapp >& pCLapp, const std::shared_ptr<KData>& sourceData, ElementDataType newElementDataType);

	// From the filesystem
	KData(const std::shared_ptr< CLapp >& pCLapp, const std::string& fileName, bool asyncLoad = false);
	KData(const std::shared_ptr< CLapp >& pCLapp, const std::string& dataFileNamePrefix,
	      std::vector<std::vector< dimIndexType >*>*& pArraysDims, numCoilsType numCoils,
	      std::vector <dimIndexType>*& pDynDims,
	      uint dataToLoad = OpenCLIPER::KData::LOADNONE,
	      const std::vector<std::string>& otherFieldsFileNamePrefixes = {"SensitivityMap_", "SamplingMask_"},
	      const std::string& coilsFileNameSuffix = "_coil", const std::string& framesFileNameSuffix = "_frame",
	      const std::string& fileNameExtension = ".raw");

	virtual ~KData() {}

	// Virtual "constructors"
	virtual std::shared_ptr<Data> clone(bool deepCopy=true) const override;
	virtual std::shared_ptr<Data> clone(ElementDataType newElementDataType) const override;

	// Getters
	const NDArray* getDataAtDynPosAndCoilId(const std::vector<dimIndexType> &dynIndexes, numCoilsType coilId) const;

	/**
	 * @brief Gets coord class field.
	 * @return coord vector
	 */
	const std::vector<realType>* getCoord() const {
	    return pCoord.get();
	}

	/**
	 * @brief Gets dcf class field.
	 * @return dcf vector
	 */
	const std::vector<realType>* getDcf() const {
	    return pDcf.get();
	}

	/**
	 * @brief Gets deltaK class field.
	 * @return deltaK vector
	 */
	const std::vector<realType>* getDeltaK() const {
	    return pDeltaK.get();
	}

	/**
	 * @brief Gets nCoils class field.
	 * @return the number of coils
	 */
	const numCoilsType getNCoils() const {
	    return nCoils;
	}

	std::shared_ptr<SensitivityMapsData> getSensitivityMapsData() const ;

	std::shared_ptr<SamplingMasksData> getSamplingMasksData() const ;

	const Trajectories* getTrajectories() const ;

	/**
	 * @brief Gets trajectory class field.
	 * @return trajectory (value of enumeration TrajType)
	 */
	const enum TrajType getTrajectory() const {
	    return trajectory;
	}

	/**
	 * @brief Gets usedCoils class field.
	 * @return the number of used coils
	 */
	const usedCoilsType getUsedCoils() const {
	    return usedCoils;
	}

	// Setters
	/**
	 * @brief Sets pCoord class field.
	 *
	 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to
	 * object, and parameter value is set to nullptr after method completion.
	 * @param[in] pCoord pointer to new value of coord vector
	 */
	void setCoord(std::vector<realType>*& pCoord) {
	    this->pCoord.reset(pCoord);
	    pCoord = nullptr;
	}

	/**
	 * @brief Sets pDcf class field.
	 *
	 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to
	 * object, and parameter value is set to nullptr after method completion.
	 * @param[in] pDcf pointer to new value of dcf vector
	 */
	void setDcf(std::vector<realType>*& pDcf) {
	    this->pDcf.reset(pDcf);
	    pDcf = nullptr;
	}

	/**
	 * @brief Sets pDeltaK class field.
	 *
	 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to
	 * object, and parameter value is set to nullptr after method completion.
	 * @param[in] pDeltaK pointer to new value of DeltaK vector
	 */
	void setDeltaK(std::vector<realType>*& pDeltaK) {
	    this->pDeltaK.reset(pDeltaK);
	    pDeltaK = nullptr;
	}

	/**
	 * @brief Sets nCoils class field.
	 * @param[in] nCoils pointer to new for number of available coils
	 */
	void setNCoils(numCoilsType nCoils) {
	    this->nCoils = nCoils;
	}

	/**
	 * @brief Sets pSensitivityMapsData class field.
	 *
	 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to
	 * object, and parameter value is set to nullptr after method completion.
	 * @param[in] pSensitivityMapsData pointer to new value of SensitivityMaps class object
	 */
	void setSensitivityMapsData(SensitivityMapsData*& pSensitivityMapsData) {
	    this->pSensitivityMapsData.reset(pSensitivityMapsData);
	    // delete(this->pSensitivityMapsData);
	    // this->pSensitivityMapsData = pSensitivityMapsData;
	    pSensitivityMapsData = nullptr;
	}

	/**
	 * @brief Sets pSamplingMasksData class field.
	 *
	 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to
	 * object, and parameter value is set to nullptr after method completion.
	 * @param[in] pSamplingMasksData pointer to new value of SensitivityMaps class object
	 */
	void setSamplingMasksData(SamplingMasksData*& pSamplingMasksData) {
	    this->pSamplingMasksData.reset(pSamplingMasksData);
	    pSamplingMasksData = nullptr;
	}

	/**
	 * @brief Sets pTrajectories class field.
	 *
	 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to
	 * object, and parameter value is set to nullptr after method completion.
	 * @param pTrajectories pointer to new value of Trajectories class object
	 */
	void setTrajectories(Trajectories*& pTrajectories) {
	    this->pTrajectories.reset(pTrajectories);
	    pTrajectories = nullptr;
	}

	/**
	 * @brief Sets trajectory class field.
	 * @param[in] trajectory new value of trajectory
	 */
	void setTrajectory(enum TrajType trajectory) {
	    this->trajectory = trajectory;
	}

	/**
	 * @brief Sets usedCoils class field.
	 * @param[in] usedCoils new value for the number of used coils
	 */
	void setUsedCoils(usedCoilsType usedCoils) {
	    this->usedCoils = usedCoils;
	}

	// Other methods
	void saveRawHostData(const std::string &fileNamePrefix,
			     const std::string &coilsFileNameSuffix = "_coil", const std::string &framesFileNameSuffix = "_frame",
			     const std::string &fileNameExtension = ".raw");
	void calcDataDims() override;
	static KData* genTestKData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, dimIndexType numFrames,
				   numCoilsType numCoilsSyncSou);
	// Save to matlab file
	void matlabSave(const std::string &fileName);
	void saveCFLData(const std::string &baseFileName);
	void saveCFLHeader(const std::string &fileName);

	// Sum all images from different coils and same time (calls process XImageSum)
	//void iFFT();
	void xImageSum(std::shared_ptr<XData> dest);
	void multSensMaps(ComplexElementProd::ConjugateSensMap_t conjugate);
	void loadCFLHeader(const std::string &fileName, std::vector< dimIndexType >*& pArraySpatialDims);
	void loadCFLData(const std::string &fileName, std::vector< dimIndexType >*& pArraySpatialDims);
	void waitLoadEnd();
    protected:
	// Inherit shared_ptr counter from base class
	std::shared_ptr<KData> shared_from_this() const {
	    return std::dynamic_pointer_cast<KData> (std::const_pointer_cast<Data>(Data::shared_from_this()));
	}

    private:
	static void create(KData* thisObj, const std::shared_ptr< CLapp >& pCLapp, const std::string& fileName);
	static constexpr const char* errorPrefix = "OpenCLIPER::KData::";
	void commonCopyConstructor(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<KData>& sourceData, bool copyData, bool copySensitivityMaps, bool copySamplingMasks);
	void loadRawHostData(const std::string& fileNamePrefix, const std::vector<std::string>& otherFieldsFileNamePrefixes,
			     uint dataToLoad, std::vector<std::vector< dimIndexType >*>*& pArraysDims, numCoilsType numCoils,
			     std::vector <dimIndexType>*& pDynDims, const std::string &coilsFileNameSuffix = "_coil",
			     const std::string &framesFileNameSuffix = "_frame", const std::string &fileNameExtension = ".raw");
	void initInternalProcesses();
	void checkMatVarDims(matvar_t* matvar);

	// Associations
	/// Pointer to object containing coils sensitivity maps
	std::shared_ptr<SensitivityMapsData> pSensitivityMapsData = nullptr;
	// Data* pSensitivityMapsData = nullptr;
	/// Pointer to object containing coils sensitivity maps rms values
	std::shared_ptr<SensitivityMapsRMS> pSensitivityMapsRMS = nullptr;
	/// Pointer to object containing sampling masks (one per time frame)
	std::shared_ptr<SamplingMasksData> pSamplingMasksData = nullptr;
	/// Pointer to object containing trajectories (one per time frame)
	std::shared_ptr<Trajectories> pTrajectories = nullptr;
	// Attributes
	/// To be documented (vector size equal to spatial dimensions vector size)
	std::unique_ptr<std::vector<realType>> pCoord = nullptr;
	/// Total number of available image coils
	numCoilsType nCoils = 1;
	/// Number of used image coils
	usedCoilsType usedCoils = {true};
	/// Type of trajectory used during images capture
	enum TrajType trajectory;
	/// To be documented (vector size equal to spatial dimensions vector size)
	std::unique_ptr<std::vector<realType>> pDcf = nullptr;
	/// To be documented (vector size equal to spatial dimensions vector size)
	std::unique_ptr<std::vector<realType>> pDeltaK = nullptr;
	/** @brief Pointer to vector of spatial dimensions of the stored group of images if fully sampled */ // default is empty
	std::unique_ptr<std::vector<dimIndexType>> pSpatialDimsFullySampled = std::unique_ptr<std::vector<dimIndexType>>(new std::vector<dimIndexType>());
	/** @brief Pointer to object containing dimensions get from matlab Dims variable */
	std::unique_ptr<MatVarDimsData> pMatVarDimsData = nullptr;
};
}
/* namespace OpenCLIPER */
#endif // KDATA_HPP
