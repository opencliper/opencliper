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

#include <OpenCLIPER/SensitivityMapsData.hpp>
#include <LPISupport/InfoItems.hpp>
#include <OpenCLIPER/InvalidDimension.hpp>

// Uncomment to show class-specific debug messages
#define SENSITIVITYMAPSDATA_DEBUG

#if !defined NDEBUG && defined SENSITIVITYMAPSDATA_DEBUG
    #define SENSITIVITYMAPSDATA_CERR(x) CERR(x)
#else
    #define SENSITIVITYMAPSDATA_CERR(x)
    #undef SENSITIVITYMAPSDATA_DEBUG
#endif

namespace OpenCLIPER {
/**
 * @brief Constructor that creates an empty SensitivityMapsData object.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 */
SensitivityMapsData::SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp): Data() {
    setApp(pCLapp, false);
}

/**
 * @brief Constructor that creates an empty SensitivityMapsData object but with spatial and temporal dimensions set.
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in,out] pArraysDims pointer to vector of spatial dimensions
 * @param[in] nCoils number of coils
 * @param[in] SYNCSOURCEDEFAULT format used for storing data in device memory (buffers, images or both)
 */
SensitivityMapsData::SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp, std::vector<std::vector<dimIndexType>*>*& pArraysDims,
	numCoilsType nCoils) :
    Data(pArraysDims) {
    this->nCoils = nCoils;
    setApp(pCLapp, false);
}

/**
 * @brief Constructor that creates an SensitivityMapsData object with data obtained from a vector of NDArray objects.
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in,out] pData pointer to vector of NDArray objects (every one contains data of the sensitivity map of a coil)
 * @param[in] nCoils number of coils
 */
SensitivityMapsData::SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp, std::vector<NDArray*>*& pData, numCoilsType nCoils): Data(pData) {
    this->nCoils = nCoils;
    setApp(pCLapp, true);
}

/**
 * @brief Constructor that creates a SensitivityMapsData object from another SenstivityMapsData object (at least number of coils
 * parameter is copied, data is copied if copyData parameter is true).
 * @param[in] pCLapp shared_ptr to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData source SensitivityMapsData object
 * @param[in] copyData if true also data (not only dimensions) are copied from sourceData object to this object
 */
SensitivityMapsData::SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<SensitivityMapsData>& sourceData, bool copyData): Data(sourceData, copyData) {
    nCoils = sourceData->nCoils;
    setApp(pCLapp, copyData);
}

/**
 * @brief Constructor that creates an uninitialized SensitivityMapsData object from a given Data object.
 * Element data type for the new object is specified by the caller
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData Object to copy data structure from
 * @param[in] newElementDataType Data type of the newly created object
 */
SensitivityMapsData::SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<SensitivityMapsData>& sourceData, ElementDataType newElementDataType):
		     Data(sourceData, newElementDataType) {
    nCoils = sourceData->nCoils;
    setApp(pCLapp, false);
}

/**
 * @brief Constructor that creates a SensitivityMapsData object from a group of files in raw format.

 * Files of this group have names with the format
 * \<fileNamePrefix\>\<dims\>\<coilsFileNameSuffix\>\<coilNumber\>\<fileNameExtension\>,
 * where \<dims\> is a string with the format \<width\>x\<height\>x\<depth\> and \<coilNumber\> is the identifier of the coil
 * used for image adquisition.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] dataFileNamePrefix name of the file in raw format
 * @param[in,out] pArraysDims reference to vectors with spatial dimensions
 * @param[in] numCoils number of coils used (number of sensitivity maps)
 * @param[in] coilsFileNameSuffix name suffix for name part depending on coil index
 * @param[in] fileNameExtension extension for the name of the file
 */
SensitivityMapsData::SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp, const std::string &dataFileNamePrefix,
	std::vector<std::vector< dimIndexType >*>*& pArraysDims,
	numCoilsType numCoils, const std::string &coilsFileNameSuffix,
	const std::string &fileNameExtension) : Data(TYPEID_COMPLEX) {
    loadRawHostData(dataFileNamePrefix, pArraysDims, numCoils, coilsFileNameSuffix, fileNameExtension);
    setApp(pCLapp, true);
}

SensitivityMapsData::SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp, const std::string &fileName, const std::vector<dimIndexType>* pArraySpatialDims,
		dimIndexType numCoils) : Data(TYPEID_COMPLEX) {
	setNCoils(numCoils);
	loadRawData(fileName, pArraySpatialDims, numCoils);
	setApp(pCLapp, true);
}

/**
 * @brief Constructor that creates a SensitivityMapsData object from a matlab format file.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] pMatlabVar pointer to matlab var containing data of the sensitivity maps
 */
SensitivityMapsData::SensitivityMapsData(const std::shared_ptr<CLapp>& pCLapp, matvar_t* pMatlabVar,
		const std::vector<dimIndexType>* pSpatialDimsFullySampled, dimIndexType nCoils): Data() {
    if(pMatlabVar == nullptr) {
	BTTHROW(std::invalid_argument(std::string(errorPrefix) + "pointer to matlab variable for sensitivity maps is nullptr"), "SensitivityMapsData::SensitivityMapsData");
    }

    checkMatVarDims(pMatlabVar, pSpatialDimsFullySampled, nCoils);

    // Dimensions of SensitivityMapdData (rank) is number of spatial dimensions + 1 (coil dimension)
    dimIndexType numOfSpatialDimensions, numOfNDArraysToBeRead;
    numOfSpatialDimensions = pMatlabVar->rank - 1;
    // Number of spatial data (NDArrays) to be read is the number of coils, last dimension of matlab variable
    setNCoils(pMatlabVar->dims[pMatlabVar->rank - 1]);
    numOfNDArraysToBeRead = getNCoils();
    SENSITIVITYMAPSDATA_CERR("SensitivityMaps reading data...\n");
    ((Data*)(this))->loadMatlabHostData(pMatlabVar, numOfSpatialDimensions, numOfNDArraysToBeRead);
    SENSITIVITYMAPSDATA_CERR("Done.\n");
    setApp(pCLapp, true);
}

void SensitivityMapsData::checkMatVarDims(matvar_t* matvar, const std::vector<dimIndexType>* pSpatialDimsFullySampled,
		dimIndexType nCoils) {
	std::string matvarName = "SensitivityMapsData";
	dimIndexType nSpatialDims = 0, nAllDims = 0;
    nSpatialDims = pSpatialDimsFullySampled->size();
    nAllDims = nSpatialDims;
	if(nCoils > 1) {
	// No coils dimensions, only 1 coil used
	//firstTemporalDimIndex = nSpatialDims; // Spatial dimensions indexes from 0 to nSpatialDims - 1
        nAllDims += 1;
    }

    if(static_cast<dimIndexType>(matvar->rank) != nAllDims) {   // Incorrect number of dimensions for SensitivityMapsData
    	SENSITIVITYMAPSDATA_CERR("Error: incorrect dimensions for SensitivityMapsData\n");
    	std::ostringstream oss;
    	oss << "Invalid " << matvarName << " dimensions (" << matvar->rank << "), should be " << (nAllDims) << std::endl;
    	BTTHROW(std::out_of_range(oss.str()), "SensitivityMapsData::SensitivityMapsData");
    }

    std::vector<dimIndexType> allDims = *(pSpatialDimsFullySampled);
    if (nCoils > 1) {
    	allDims.push_back(nCoils);
    }
    try {
    	this->checkMatvarDims(matvar, &allDims);
    } catch (InvalidDimension &e) {
    	SENSITIVITYMAPSDATA_CERR("Error: incorrect dimensions for SensitivityMapsData\n");
    	std::ostringstream oss;
    	oss << "Invalid SensitivityMapsData ";
    	if (e.getInvalidDimId() < nSpatialDims) {// Error in spatial dim
    		oss << "spatial dimension " << e.getInvalidDimId() << ": is ";
    	} else if ((nCoils > 1) && e.getInvalidDimId() == nSpatialDims) { // Error in number of coils
    		oss << "number of coils: is ";
    	}
    	oss << e.getWrongValue() << ", should be " << e.getRightValue() << std::endl;
    	BTTHROW(std::out_of_range(oss.str()), "SensitivityMapsData::SensitivityMapsData");
    }
}

/**
 * Destructor.
 */
SensitivityMapsData::~SensitivityMapsData() {
#ifdef SensitivityMapsData_DEBUG
    SENSITIVITYMAPSDATA_CERR("~SensitivityMapsData() begins..." << std::endl);
    SENSITIVITYMAPSDATA_CERR("~SensitivityMapsData() ends..." << std::endl);
#endif
}

/**
 * @brief Virtual copy "constructor"
 * @param[in] deepCopy copy stored data if true; only data structure if false (shallow copy)
 * @return A copy of this object
 */
std::shared_ptr<Data> SensitivityMapsData::clone(bool deepCopy) const {
    return std::make_shared<SensitivityMapsData>(pCLapp, shared_from_this(), deepCopy);
}

/**
 * @brief Virtual copy "constructor"
 * @param[in] newElementDataType data type of the newly created object. Shallow copy is implied.
 * @return An empty copy of this object that contains elements of type "newElementDataType"
 */
std::shared_ptr<Data> SensitivityMapsData::clone(ElementDataType newElementDataType) const {
    return std::make_shared<SensitivityMapsData>(pCLapp, shared_from_this(), newElementDataType);
}


/**
 * Load data of a group of files to NDArray objects (every file contains one image and is stored into a NDArray object).
 * Files of this group have names with the format
 * \<dataFileNamePrefix\>\<dims\>\<coilsFileNameSuffix\>\<coilNumber\>\<fileNameExtension\>,
 * where \<dims\> is a string with the format \<width\>x\<height\>x\<depth\> and \<coilNumber\> is the identifier
 * of the coil used for image adquisition.
 * @param[in] dataFileNamePrefix fixed part of the file name before the variable part
 * @param[in,out] pArraysDims pointer to a vector of pointers to vectors with the dimensions of every image
 * @param[in] numCoils number of images per time frame (number of coils)
 * @param[in] coilsFileNameSuffix part of the file name before the coil number
 * @param[in] fileNameExtension extension for file names
 *
 */
void SensitivityMapsData::loadRawHostData(const std::string &dataFileNamePrefix,
	std::vector<std::vector< dimIndexType >*>*& pArraysDims, numCoilsType numCoils,
	const std::string &coilsFileNameSuffix,
	const std::string &fileNameExtension) {
    std::stringstream outputstream;
    if(pArraysDims == nullptr) {
	outputstream << errorPrefix << "pArrayDims must not be null";
	BTTHROW(std::invalid_argument(outputstream.str()), "SensitivityMapsData::loadRawHostData");
    }
    this->setNCoils(numCoils);
    //vector<dimIndexType>* pDims = new vector<dimIndexType>(*(pArraysDims->at(0)));
    // Filename prefix for SensitivityMaps
    std::string sensitivityMapsfileNamePrefixWitDims =
	OpenCLIPER::Data::buildFileNamePrefix(dataFileNamePrefix, pArraysDims->at(0));
    std::vector<std::vector<dimIndexType>*>* pSensitivityMapsArraysDims = new std::vector<std::vector<dimIndexType>*>();
    // Filename suffixes for SensitivityMaps
    std::vector<std::string> sensitivityMapsFileNameSuffixes;
    std::stringstream variableSuffixStreamSensitivityMaps;
    for(numCoilsType coil = 0; coil < numCoils; coil++) {
	// add dimensions vector (same dimensions for every map)
	pSensitivityMapsArraysDims->push_back(new std::vector<dimIndexType>(*(pArraysDims->at(coil))));
	// 2 digits for coil number
	variableSuffixStreamSensitivityMaps << coilsFileNameSuffix << std::setfill('0') << std::setw(2) << coil;
	sensitivityMapsFileNameSuffixes.push_back(variableSuffixStreamSensitivityMaps.str()); // add file name suffix
	variableSuffixStreamSensitivityMaps.str(""); // emtpy string, we don't want to accumulate strings among iterations
    }
    std::vector <dimIndexType>* pDynDimsSensitivityMaps = new std::vector <dimIndexType>(*(pDynDims.get()));
    (reinterpret_cast<OpenCLIPER::Data*>(this))->loadRawData(sensitivityMapsfileNamePrefixWitDims, pSensitivityMapsArraysDims,
	    pDynDimsSensitivityMaps, sensitivityMapsFileNameSuffixes);
}

void SensitivityMapsData::saveCFLHeader(const std::string &fileName) {
	std::fstream f;
	LPISupport::Utils::openFile(fileName, f, std::ofstream::out|std::ofstream::trunc, "SensitivityMapsData::saveCFLHeader");
	const std::vector<dimIndexType>* pSpatialDims = getNDArray(0)->getDims(); // All NDArrays must have the same sptial dimensions
	for (uint i=0; i < pSpatialDims->size(); i++) {
		f << pSpatialDims->at(i) << " ";
	}
	f << this->nCoils;
	f << " 1"; // No temporal dims => 1 temporal dimension supported of value 1
	f << std::endl;
	f.close();
}

/**
 * Stores image/volume spatial and temporal dimensions in class field.
 */
void SensitivityMapsData::calcDataDims() {
    Data::calcDataDims();
    pDataDimsAndStridesVector->at(NumCoilsPos) = nCoils;
}

} /* namespace OpenCLIPER */
#undef SENSITIVITYMAPSDATA_DEBUG
