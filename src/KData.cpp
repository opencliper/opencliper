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
/*
 * KData.cpp
 *
 *  Created on: 28 de oct. de 2016
 *      Author: manrod
 */

#include <OpenCLIPER/KData.hpp>
#include <OpenCLIPER/XData.hpp> // For XImagesSum method
#include <OpenCLIPER/InvalidDimension.hpp>

// Uncomment to show class-specific debug messages
#define KDATA_DEBUG

#if !defined NDEBUG && defined KDATA_DEBUG
    #define KDATA_CERR(x) CERR(x)
#else
    #define KDATA_CERR(x)
    #undef KDATA_DEBUG
#endif

namespace OpenCLIPER {

/**
 * @brief Constructor that creates an empty KData object.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 */
KData::KData(const std::shared_ptr<CLapp>& pCLapp): Data() {
    // AddData, called by setApp, fails if pNDArrays is nullptr or empty, do not call setApp.
    //setApp(pCLapp);
    // But pCLapp must be set.
    this->pCLapp = pCLapp;
}

/**
 * @brief Constructor that creates a KData object from a vector of NDArrays.
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in,out] pSensitivityMapsData pointer to a SensitivityMaps object (group of sensitivity maps for coils)
 * @param[in,out] pNDArrays pointer to vector of pointers to NDArray objects
 * @param[in,out] pCoord
 * @param[in] nCoils total number of coils available
 * @param[in] usedCoils number of coils used
 * @param[in,out] pDynDims pointer to vector of temporal dimensions
 * @param[in] trajectory trajectory used
 * @param[in,out] pDcf
 * @param[in,out] pDeltaK
 * @throw std::invalid_argument if incorrect dimensions of KData according to spatial and temporal dimensions parameters
 */
KData::KData(const std::shared_ptr<CLapp>& pCLapp, SensitivityMapsData*& pSensitivityMapsData, std::vector<NDArray*>*& pNDArrays,
	     std::vector<realType>*& pCoord, numCoilsType nCoils, usedCoilsType usedCoils, std::vector<dimIndexType>*& pDynDims,
	     enum TrajType trajectory, std::vector<realType>*& pDcf, std::vector<realType>*& pDeltaK): Data(pNDArrays) {
    // TODO Auto-generated constructor stub
    /* To-do:
     * - check size(coord) == nDims
     * - check size(dcf) == nDims
     * - check size(deltaK) == nDims
     * - check size(usedCoils) == nCoils
     * Problema: nDims hay que leerlo de un objeto de la clase NDArray,
     * pero KData tiene referencias a varios objetos de tipo NDArray
     * (vector data que pueden tener valores distintos de nDims)
     */
    this->pSensitivityMapsData.reset(pSensitivityMapsData);
    pSensitivityMapsData = nullptr;
    this->pCoord.reset(pCoord);
    pCoord = nullptr;
    this->nCoils = nCoils;
    this->usedCoils = usedCoils;
    this->trajectory = trajectory;
    this->pDcf.reset(pDcf);
    pDcf = nullptr;
    this->pDeltaK.reset(pDeltaK);
    pDeltaK = nullptr;
    this->pDynDims.reset(pDynDims);
    pDynDims = nullptr;
    dimIndexType totalNumberOfDynDims = getDynDimsTotalSize();
    if((totalNumberOfDynDims * nCoils) != pNDArrays->size()) {
	BTTHROW(std::invalid_argument("sum of dynamic dimensions (DynDims) * number of coils must be equal to kData vector size"), "KData::KData");
    }
    internalSetData(pNDArrays);
    setApp(pCLapp, true);
}

/**
 * @brief Constructor that creates an empty KData object but with spatial and temporal dimensions set.
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in,out] pArraysDims pointer to a vector of pointers to vectors with the spatial dimensions of every image
 * @param[in] nCoils total number of coils available
 * @param[in,out] pDynDims pointer to vector of temporal dimensions
 */
KData::KData(const std::shared_ptr<CLapp>& pCLapp, std::vector<std::vector<dimIndexType>*>*& pArraysDims, numCoilsType nCoils,
	     std::vector<dimIndexType>*& pDynDims) : Data(pArraysDims, pDynDims) {
    // All dimensions for every NDArray must be valid
    //checkValidSpatialDimensions();
    this->nCoils = nCoils;
    setApp(pCLapp, false);
}

/**
 * @brief Constructor that creates a KData object with data (optional), spatial and temporal dimensions got from
 * another KData object.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData KData object source of spatial and temporal dimensions
 * @param[in] copyData if true also data (not only dimensions) are copied from sourceData object to this object
 * @param[in] copySensitivityMaps if true also sensitivity maps (not only dimensions) are copied from sourceData object to this object
 * @param[in] copySamplingMasks if true also sampling masks (not only dimensions) are copied from sourceData object to this object
 */
KData::KData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<KData>& sourceData, bool copyData, bool copySensitivityMaps, bool copySamplingMasks):
       Data(sourceData, copyData) {

    commonCopyConstructor(pCLapp, sourceData, copyData, copySensitivityMaps, copySamplingMasks);
    setApp(pCLapp, copyData);
}

/**
 * @brief Constructor that creates an uninitialized KData object from another KData object.
 * Element data type for the new object is specified by the caller
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData Object to copy data structure from
 * @param[in] newElementDataType Data type of the newly created object
 */
KData::KData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<KData>& sourceData, ElementDataType newElementDataType):
       Data(sourceData, newElementDataType) {

    commonCopyConstructor(pCLapp, sourceData, false, false, false);
    setApp(pCLapp, false);
}

/**
 * @brief Common tasks for copy constructors (creates a KData object with data (optional), spatial and temporal dimensions got from
 * another KData object).
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData KData object source of spatial and temporal dimensions
 * @param[in] copyData if true also data (not only dimensions) are copied from sourceData object to this object
 * @param[in] copySensitivityMaps if true also sensitivity maps (not only dimensions) are copied from sourceData object to this object
 * @param[in] copySamplingMasks if true also sampling masks (not only dimensions) are copied from sourceData object to this object
 */
void KData::commonCopyConstructor(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<KData>& sourceData, bool copyData, bool copySensitivityMaps,  bool copySamplingMasks)  {
    std::vector<realType>* pLocalCoord;
    if(sourceData->getCoord() == nullptr) {
	pLocalCoord = nullptr;
    }
    else {
	pLocalCoord = new std::vector<realType>(*(sourceData->getCoord()));
    }
    setCoord(pLocalCoord);
    setNCoils(sourceData->getNCoils());
    setUsedCoils(sourceData->getUsedCoils());
    setTrajectory(sourceData->getTrajectory());
    std::vector<realType>* pLocalDcf;
    if(sourceData->getDcf() == nullptr) {
	pLocalDcf = nullptr;
    }
    else {
	pLocalDcf = new std::vector<realType>(*(sourceData->getDcf()));
    }
    setDcf(pLocalDcf);
    std::vector<realType>* pLocalDeltaK;
    if(sourceData->getDeltaK() == nullptr) {
	pLocalDeltaK = nullptr;
    }
    else {
	pLocalDeltaK = new std::vector<realType>(*(sourceData->getDeltaK()));
    }
    setDeltaK(pLocalDeltaK);
    if(copySensitivityMaps == true) {
	SensitivityMapsData* sensitivityMapsData =
	    new SensitivityMapsData(pCLapp, sourceData->getSensitivityMapsData(), true);
	setSensitivityMapsData(sensitivityMapsData);
    }
    if(copySamplingMasks == true) {
	SamplingMasksData* samplingMasksData =
	    new SamplingMasksData(pCLapp, sourceData->getSamplingMasksData(), true);
	setSamplingMasksData(samplingMasksData);
    }
}

/**
 * @brief Constructor tha loads data of a group of files in raw format (see @ref loadRawHostData) to NDArray objects
 * (every file contains one image and is stored into a NDArray object).
 *
 * Files of this group have names with the format
 * \<fileNamePrefix\>\<dims\>\<coilsFileNameSuffix\>\<coilNumber\>\<framesFileNameSuffix\>\<frameNumber\>\<fileNameExtension\>,
 * where \<dims\> is a string with the format \<width\>x\<height\>x\<depth\>, \<coilNumber\> is the identifier of the coil
 * used for image adquisition and \<frameNumber\> is the identifier of the time of image adquisition.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] dataFileNamePrefix fixed part of the file name before the variable part
 * @param[in,out] pArraysDims pointer to a vector of pointers to vectors with the dimensions of every image
 * @param[in] numCoils number of images per time frame (number of coils)
 * @param[in,out] pDynDims pointer to a vector with the temporal dimensions of every image of the sequence
 * @param[in] dataToLoad bit mask storing extra KData fields to be loaded (LOADSENSITIVITYMAPS for loading sensitivity maps and
 * LOADSAMPLINGMASKS for loading sampling masks)
 * @param[in] otherFieldsFileNamePrefixes name prefixes (after dataFileNamePrefix) for sensitivity maps and sampling masks
 * @param[in] coilsFileNameSuffix part of the file name before the coil number
 * @param[in] framesFileNameSuffix part of the file name before the frame number
 * @param[in] fileNameExtension extension for file names
 */
KData::KData(const std::shared_ptr<CLapp>& pCLapp, const std::string& dataFileNamePrefix,
	     std::vector<std::vector< dimIndexType >*>*& pArraysDims, numCoilsType numCoils,
	     std::vector <dimIndexType>*& pDynDims, uint dataToLoad,
	     const std::vector<std::string>& otherFieldsFileNamePrefixes, const std::string& coilsFileNameSuffix,
	     const std::string& framesFileNameSuffix, const std::string& fileNameExtension):
    Data() {
    loadRawHostData(dataFileNamePrefix, otherFieldsFileNamePrefixes, dataToLoad, pArraysDims, numCoils, pDynDims,
		    coilsFileNameSuffix, framesFileNameSuffix, fileNameExtension);
    setApp(pCLapp, true);
}

void KData::loadCFLHeader(const std::string &fileName, std::vector< dimIndexType >*& pArraySpatialDims) {
	std::fstream f;
	LPISupport::Utils::openFile(fileName, f, std::ios::in, "KData::loadCFLFormatHeader");
	dimIndexType width, height, temporalDim0;
	std::string line;
	while (std::getline(f, line))
	{
		// Discard comments (lines beginning with a # symbol)
		if (line.at(0) != '#') {
			std::istringstream iss(line);
			if (!(iss >> width >> height >> nCoils >> temporalDim0)) {
				BTTHROW(std::invalid_argument("Invalid line format: " + line + " (should be <width> <depth> <numCoils> <numFrames>)"), "KData::loadCFLHeader");
			}
			break; // First valid format line processed, discard remaining lines
		}
	}
	pArraySpatialDims->push_back(width);
	pArraySpatialDims->push_back(height);
	std::vector<dimIndexType>* pTemporalDimsVector = new std::vector<dimIndexType>;
	pTemporalDimsVector->push_back(temporalDim0);
	this->setDynDims(pTemporalDimsVector);
}

void KData::loadCFLData(const std::string &fileName, std::vector< dimIndexType >*& pArraySpatialDims) {
	dimIndexType numNDArrays = nCoils * this->getDynDimsTotalSize();
	Data::loadRawData(fileName, pArraySpatialDims, numNDArrays);
}

/**
 * @brief Constructor that creates an KData object from a file in matlab format (containing one ore more variables).
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] fileName name of the data file
 * @throw std::invalid_argument if dimensions and KData mandatory variables are missing from matlab file or KData dimensions are incorrect
 * according to dimensions variable
 */
KData::KData(const std::shared_ptr<CLapp>& pCLapp, const std::string &fileName, bool asyncLoad) : Data() {
    if(asyncLoad) {
	this->pFileLoaderThread = std::unique_ptr<std::thread>(new std::thread(KData::create, this, pCLapp, fileName));
    }
    else {
	create(this, pCLapp, fileName);
	pFileLoaderThread = nullptr;
    }
}


void KData::create(KData* thisObj, const std::shared_ptr<CLapp>& pCLapp, const std::string &fileName) {
    std::map<std::string, matvar_t*>* pMatlabVariablesMap;
    SensitivityMapsData* pSensitivityMapsData;
    SamplingMasksData* pSamplingMasksData;
    #ifdef KDATA_DEBUG
    BEGIN_TIME(bTReadingMatlabFile);
    KDATA_CERR("Reading matlab file...\n");
#endif
    try {
    	pMatlabVariablesMap = Data::readMatlabVariablesFromFile(fileName);
    } catch(std::invalid_argument& e) { // Not Matlab file, try CFL format file
    	std::vector<dimIndexType>* pArraySpatialDims = new std::vector<dimIndexType>;
    	std::string headerFileName;
    	headerFileName = LPISupport::Utils::basename(fileName) + CFLHeaderExtension;
    	// loadCFLHeader required to set pArraySpatialDims, which is required by
    	// sensitivityMaps and samplingMasks constructors
		thisObj->loadCFLHeader(headerFileName, pArraySpatialDims);
		thisObj->loadCFLData(fileName, pArraySpatialDims);

		std::string sensitivityMapsDataName = LPISupport::Utils::basename(fileName) + std::string(CFLSensMapsSuffix);

		try {
			pSensitivityMapsData = new SensitivityMapsData(pCLapp, sensitivityMapsDataName, pArraySpatialDims, thisObj->nCoils);
			thisObj->setSensitivityMapsData(pSensitivityMapsData);
		} catch (std::exception &e) {
			KDATA_CERR("Warning: no sensitivity maps file " + sensitivityMapsDataName + "\n");
		}

		std::string samplingMasksDataName = LPISupport::Utils::basename(fileName) + std::string(CFLSampMasksSuffixRowMask);
		try {
			KDATA_CERR("Trying to open " + samplingMasksDataName + "... ");
			std::ifstream infile(samplingMasksDataName);
			dimIndexType numColumns = pArraySpatialDims->at(0);
			if (infile.good()) { // rowmask format, dimensions height x number of frames
				KDATA_CERR("Found.");
				dimIndexType numColumns = pArraySpatialDims->at(0);
				pArraySpatialDims->erase(pArraySpatialDims->begin()); // Remove width from spatial dimensions (row mask format)
				pSamplingMasksData = new SamplingMasksData(pCLapp, samplingMasksDataName, pArraySpatialDims, thisObj->getDynDims(), numColumns, TYPEID_INDEX);
				thisObj->setSamplingMasksData(pSamplingMasksData);
			} else { // pixelmask format, dimensions: width x height x number of frames
				samplingMasksDataName = LPISupport::Utils::basename(fileName) + std::string(CFLSampMasksSuffixPixelMask);
				KDATA_CERR("Not found. Trying to open " + samplingMasksDataName + "... ");
				pSamplingMasksData = new SamplingMasksData(pCLapp, samplingMasksDataName, pArraySpatialDims,thisObj->getDynDims(), numColumns, TYPEID_CL_UCHAR);
				thisObj->setSamplingMasksData(pSamplingMasksData);
			}
		} catch (std::exception &e) {
			KDATA_CERR("Warning: no sampling mask file " + samplingMasksDataName + "\n");
		}

	    thisObj->setApp(pCLapp, true);
	    // Delete pArraySpatialDims after being used for creating KData, Sensitivity maps and sampling masks
	    delete(pArraySpatialDims);
	    pArraySpatialDims = nullptr;
		return;
	}
#ifdef KDATA_DEBUG
    KDATA_CERR("Done\n");
    END_TIME(eTReadingMatlabFile);
    TIME_DIFF_TYPE diffTReadingMatlabFile;
    TIME_DIFF(diffTReadingMatlabFile, bTReadingMatlabFile, eTReadingMatlabFile);
    KDATA_CERR("Elapsed time: " << diffTReadingMatlabFile << " s" << std::endl);
#endif
    matvar_t* pDimsMatVar = nullptr, *pKDataMatVar = nullptr, *pSensitivityMapsDataMatVar = nullptr;
    matvar_t* pSamplingMasksDataMatVar = nullptr, *pTrajectoriesMatVar = nullptr;

    // Read of matlab variable storing number of spatial and temporal dimensions of data (if not exists it is an unrecoverable error and method aborts)
    try {
	pDimsMatVar = pMatlabVariablesMap->at(MatVarDimsData::matVarNameDims);
    }
    catch(std::out_of_range& e) {
	BTTHROW(std::invalid_argument("Error: no Dims variable in matlab file"), "KData::KData");
    }

    dimIndexType nSpatialDims = 0, numOfNDArraysToBeRead = 0; //, firstTemporalDimIndex = 0;

    // KData number of dimensions = number of spatial dimensions + 1 (coil dimension) + number of temporal dimensions or
    //= number of spatial dimensions +  number of temporal dimensions (if only 1 coil is used)
    // SensitivityMaps number of dimensions = number of spatial dimensions + 1 (coil dimension)
    // SamplingMasks number of dimensions = number of spatial dimensions - 1 + number of temporal dimensions

    // Read of matlab variable with dimensions and extraction of sizes of spatial dimensions (pSpatialDimsFullySampled parameter),
    // number of coils used (nCoils parameter) and number of temporal dimensions (nTemporalDims parameter)
    MatVarDimsData* pMatVarDimsData = new MatVarDimsData(pCLapp, pDimsMatVar);
    pMatVarDimsData->getAllDims(thisObj->pSpatialDimsFullySampled.get(), thisObj->nCoils, thisObj->pDynDims.get());
    //readDimsMatlabVar(pDimsMatVar, pSpatialDimsFullySampled.get(), nCoils, pDynDims.get());
    nSpatialDims = thisObj->pSpatialDimsFullySampled->size();
    thisObj->setNCoils(thisObj->nCoils);

    // Read of matlab variable storing KData data (if not exists it is an unrecoverable error and method is aborted)
    try {
	pKDataMatVar = pMatlabVariablesMap->at(matVarNameKData);
	// out_of_range exception if there is no value with the key "KData" in the map
    }
    catch(std::out_of_range& e) {
	BTTHROW(std::invalid_argument("Error: no KData variable in matlab file"), "KData::KData");
    }

    thisObj->checkMatVarDims(pKDataMatVar);

    // Number of KData NDArrays to be read is number of coils * product of all temporal dimensions
    numOfNDArraysToBeRead = thisObj->getNCoils() * thisObj->getDynDimsTotalSize();
    if(numOfNDArraysToBeRead == 0) {
	numOfNDArraysToBeRead = 1; // At least 1 NDArray exist when there is only 1 coil and 1 time frame
    }
    KDATA_CERR("KData reading data...\n" << std::endl);
    ((Data*)(thisObj))->loadMatlabHostData(pKDataMatVar, nSpatialDims, numOfNDArraysToBeRead);
    KDATA_CERR("Done.\n");

    // Read of matlab variable storing sensitivity maps data (if not exists, a warning error is shown)
    try {
	pSensitivityMapsDataMatVar = pMatlabVariablesMap->at(SensitivityMapsData::matVarNameSensitivityMaps);
	// Create sensitivity maps and store them in class field
	pSensitivityMapsData = new SensitivityMapsData(pCLapp, pSensitivityMapsDataMatVar, thisObj->pSpatialDimsFullySampled.get(), thisObj->nCoils);
	thisObj->setSensitivityMapsData(pSensitivityMapsData);
    }
    catch(std::out_of_range& e) {
	KDATA_CERR("Warning: no sensitivity maps variable in matlab file (variable name not found: " + std::string(SensitivityMapsData::matVarNameSensitivityMaps) + ")\n");
    }


    // Read of matlab variable storing sampling masks data (if not exists, a warning error is shown)
    try {
	pSamplingMasksDataMatVar = pMatlabVariablesMap->at(SamplingMasksData::matVarNameSamplingMasks);
	// Create sampling masks and store them in class field
	pSamplingMasksData = new SamplingMasksData(pCLapp, pSamplingMasksDataMatVar, thisObj->getNDArray(0)->getDims());
	thisObj->setSamplingMasksData(pSamplingMasksData);
    }
    catch(std::out_of_range& e) {
	KDATA_CERR("Warning: no sampling masks variable in matlab file (variable name not found: " + std::string(SamplingMasksData::matVarNameSamplingMasks) + ")\n");
    }

    // Read of matlab variable storing trajectories data (if not exists, a warning error is shown)
    try {
	pTrajectoriesMatVar = pMatlabVariablesMap->at(Trajectories::matVarNameTrajectories);
	// Create sampling masks and store them in class field (number of spatial dimensions of sampling mask data is
	// number of spatial dimensions of KData - 1
	Trajectories* pTrajectoriesData = new Trajectories(pCLapp, pTrajectoriesMatVar, nSpatialDims);
	thisObj->setTrajectories(pTrajectoriesData);
    }
    catch(std::out_of_range& e) {
	KDATA_CERR("Warning: no trajectories variable in matlab file (variable name not found: " + std::string(Trajectories::matVarNameTrajectories) + ")\n");
    }
    thisObj->setApp(pCLapp, true);
}

void KData::checkMatVarDims(matvar_t* matvar) {
	std::string matvarName = "KData";
    dimIndexType nSpatialDims = 0, nTemporalDims = 0, nSpatialAndTempDims = 0; //, firstTemporalDimIndex = 0;
    nSpatialDims = pSpatialDimsFullySampled->size();
    nTemporalDims = pDynDims->size();
    int firstTemporalDimMatVarId = nSpatialDims; // First temporal dimension index after last spatial dimension
    nSpatialAndTempDims = nSpatialDims + nTemporalDims;
	if(nCoils > 1) {
	// No coils dimensions, only 1 coil used
	//firstTemporalDimIndex = nSpatialDims; // Spatial dimensions indexes from 0 to nSpatialDims - 1
    	firstTemporalDimMatVarId += 1; // First temporal dimension index after last spatial dimension
        nSpatialAndTempDims += 1;
    }

    if(static_cast<dimIndexType>(matvar->rank) != nSpatialAndTempDims) {   // Incorrect number of dimensions for KData
    	KDATA_CERR("Error: incorrect dimensions for KData\n");
    	std::ostringstream oss;
    	oss << "Invalid " << matvarName << " dimensions (" << matvar->rank << "), should be " << (nSpatialAndTempDims) << std::endl;
    	BTTHROW(std::out_of_range(oss.str()), "KData::KData");
    }

    std::vector<dimIndexType> allDims = *(pSpatialDimsFullySampled.get());
    if (nCoils > 1) {
    	allDims.push_back(nCoils);
    }
    allDims.insert(std::end(allDims), std::begin(*(pDynDims.get())), std::end(*(pDynDims.get())));
    try {
    	this->checkMatvarDims(matvar, &allDims);
    } catch (InvalidDimension &e) {
    	KDATA_CERR("Error: incorrect dimensions for KData\n");
    	std::ostringstream oss;
    	oss << "Invalid KData ";
    	if (e.getInvalidDimId() < nSpatialDims) {// Error in spatial dim
    		oss << "spatial dimension " << e.getInvalidDimId() << ": is ";
    	} else if ((nCoils > 1) && e.getInvalidDimId() == nSpatialDims) { // Error in number of coils
    		oss << "number of coils: is ";
    	} else {
    		 // First temporal dim after last spatial dim
    		dimIndexType temporalDimId = e.getInvalidDimId() - nSpatialDims;
    		if (nCoils > 1) {
    			 temporalDimId -= 1; // First temporal dim after number of coils
    		}
    		oss << "temporal dimension " << temporalDimId << ": is ";
    	}
    	oss << e.getWrongValue() << ", should be " << e.getRightValue() << std::endl;
    	BTTHROW(std::out_of_range(oss.str()), "KData::KData");
    }
}

/**
 * @brief Virtual copy "constructor"
 * @param[in] deepCopy copy stored data if true; only data structure if false (shallow copy)
 * @return A copy of this object
 */
std::shared_ptr<Data> KData::clone(bool deepCopy) const {
    return std::make_shared<KData>(pCLapp, shared_from_this(), deepCopy, deepCopy, deepCopy);
}

/**
 * @brief Virtual copy "constructor"
 * @param[in] newElementDataType data type of the newly created object. Shallow copy is implied.
 * @return An empty copy of this object that contains elements of type "newElementDataType"
 */
std::shared_ptr<Data> KData::clone(ElementDataType newElementDataType) const {
    return std::make_shared<KData>(pCLapp, shared_from_this(), newElementDataType);
}

/**
 * @brief Gets pSensitivityMapsData field.
 * @return pointer to object of class SensitivityMapsData (group of sensitivity maps for coils)
 */
std::shared_ptr<SensitivityMapsData> KData::getSensitivityMapsData() const {
    if(pSensitivityMapsData == nullptr)
	BTTHROW(std::invalid_argument("Cannot get SensitivityMaps pointer, KData does not include sensitivity maps"), "KData::getSensitivityMapsData");
    return pSensitivityMapsData;
}

/**
 * @brief Gets pSamplingMasksData field.
 * @return pointer to object of class SamplingMasksData
 */
std::shared_ptr<SamplingMasksData> KData::getSamplingMasksData() const {
    if(pSamplingMasksData == nullptr)
	BTTHROW(std::invalid_argument("Cannot get SamplingMasks pointer, KData does not include sampling masks"), "KData::getSamplingMasksData");
    return pSamplingMasksData;
}

/**
 * @brief Gets pTrajectories field.
 * @return pointer to object of class Trajectories
 */
const Trajectories* KData::getTrajectories() const {
    if(pTrajectories == nullptr)
	BTTHROW(std::invalid_argument("Cannot get Trajectories pointer, KData does not include trajectories"), "KData::getTrajectories");
    return pTrajectories.get();
}

/**
 * @brief Gets a pointer to one NDArray object (image data) from the vector of NDArray objects.
 * @param[in] dynIndexes vector with values for the temporal indexes used for indexing the NDArray objects vector
 * @param[in] coilId number of coil from indexing the NDArray objects vector
 * @return pointer to NDArray object at the specified position
 */
const NDArray* KData::getDataAtDynPosAndCoilId(const std::vector<dimIndexType> &dynIndexes, numCoilsType coilId) const {
    index1DType index1D;
    index1D = this->get1DIndexFromDynPos(dynIndexes);
    index1D = index1D + coilId * this->getDynDimsTotalSize();
    KDATA_CERR("KData index1D: " << index1D);
    return pNDArrays->at(index1D).get();
}

/**
 * @brief Load data of a group of files to NDArray objects (every file contains one image and is stored into a NDArray object).
 *
 * Files of this group have names with the format
 * \<fileNamePrefix\>\<dims\>\<coilsFileNameSuffix\>\<coilNumber\>\<framesFileNameSuffix\>\<frameNumber\>\<fileNameExtension\>,
 * where \<dims\> is a string with the format \<width\>x\<height\>x\<depth\>, \<coilNumber\> is the identifier of the coil used
 * for image adquisition and \<frameNumber\> is the identifier of the time of image adquisition.
 * @param[in] dataFileNamePrefix fixed part of the file name before the variable part
 * @param[in] otherFieldsFileNamePrefixes name prefixes (after dataFileNamePrefix) for sensitivity maps and sampling masks
 * @param[in] dataToLoad bit mask storing extra KData fields to be loaded (LOADSENSITIVITYMAPS for loading sensitivity maps and
 * LOADSAMPLINGMASKS for loading sampling masks)
 * @param[in,out] pArraysDims pointer to a vector of pointers to vectors with the dimensions of every image
 * @param[in] numCoils number of images per time frame (number of coils)
 * @param[in,out] pDynDims pointer to a vector with the temporal dimensions of every image of the sequence
 * @param[in] coilsFileNameSuffix part of the file name before the coil number
 * @param[in] framesFileNameSuffix part of the file name before the frame number
 * @param[in] fileNameExtension extension for file names
 */
void KData::loadRawHostData(const std::string &dataFileNamePrefix, const std::vector<std::string> &otherFieldsFileNamePrefixes,
			    uint dataToLoad, std::vector<std::vector< dimIndexType >*>*& pArraysDims, numCoilsType numCoils,
			    std::vector <dimIndexType>*& pDynDims, const std::string &coilsFileNameSuffix,
			    const std::string &framesFileNameSuffix, const std::string &fileNameExtension) {
    index1DType numFrames;
    if(pDynDims->size() == 0) {
	numFrames = 0;
    }
    else {
	numFrames = 1;
    }
    for(index1DType i = 0; i < pDynDims->size(); i++) {
	numFrames = numFrames * pDynDims->at(i);
    }

    // Sensitivity maps loading
    if(dataToLoad & LOADSENSITIVITYMAPS) {
	// Spatial dimensions vector for sensitivity maps (all images of same size)
	std::vector<std::vector< dimIndexType >*>* pArraysDimsSensitivityMaps = new std::vector<std::vector< dimIndexType >*>();
	//for (dimIndexType i = 0; i < pArraysDims->size(); i++) {
	// Size of pArraysDims == number of NDArrays == numCoils * getDynDimsTotalSize
	// spatial dimensions array for sampling masks only have
	// numCoil positions
	for(dimIndexType i = 0; i < numCoils; i++) {
	    std::vector< dimIndexType >* pDims;
	    pDims = new std::vector<dimIndexType>(*(pArraysDims->at(i)));
	    pArraysDimsSensitivityMaps->push_back(pDims);
	}
	// New Data object for storing SensitivityMaps data of input KData
	// Load SensitivityMaps data from raw files created from matlab

	SensitivityMapsData* pSensitivityMapsData =
	    new OpenCLIPER::SensitivityMapsData(pCLapp,
						dataFileNamePrefix + otherFieldsFileNamePrefixes.at(SENSITIVITYMAPSPREFIX),
						pArraysDimsSensitivityMaps, numCoils, coilsFileNameSuffix, fileNameExtension);
	setSensitivityMapsData(pSensitivityMapsData);
    }

    // Sampling masks loading
    if(dataToLoad & LOADSAMPLINGMASKS) {
	// Spatial dimensions vector for sensitivity maps (all images of same size)
	std::vector<std::vector< dimIndexType >*>* pArraysDimsSamplingMasks = new std::vector<std::vector< dimIndexType >*>();
	// for (dimIndexType i = 0; i < pArraysDims->size(); i++) {
	// Size of pArraysDims == number of NDArrays == numCoils * getDynDimsTotalSize
	// spatial dimensions array for sampling masks only have
	// getDynDimsTotalSize positions
	// Sampling masks has 1 spatial dimension (number of columns) for 2D-images
	for(dimIndexType i = 0; i < numFrames; i++) {
	    std::vector< dimIndexType >* pDims;
	    pDims = new std::vector<dimIndexType>();
	    pDims->push_back(pArraysDims->at(i)->at(0));
	    pArraysDimsSamplingMasks->push_back(pDims);
	}
	// New Data object for storing SensitivityMaps data of input KData
	// Load SensitivityMaps data from raw files created from matlab

	std::vector<dimIndexType>* pDynDimsSamplingMasks = new std::vector<dimIndexType>(*(pDynDims));
	SamplingMasksData* pSamplingMasksData =
	    new OpenCLIPER::SamplingMasksData(pCLapp, dataFileNamePrefix + otherFieldsFileNamePrefixes.at(SAMPLINGMASKSPREFIX),
					      pArraysDimsSamplingMasks, pDynDimsSamplingMasks, pArraysDims->at(0)->at(WIDTHPOS) , framesFileNameSuffix,
					      fileNameExtension);
	setSamplingMasksData(pSamplingMasksData);
    }

    // KData data loading
    this->setNCoils(numCoils);
    std::stringstream variableSuffixStream;
    std::vector<std::string> fileNameSuffixes;
    for(dimIndexType frame = 0; frame < numFrames; frame++) {
	for(numCoilsType coil = 0; coil < numCoils; coil++) {
	    variableSuffixStream << coilsFileNameSuffix << std::setfill('0') << std::setw(2) << coil; // _coilCC
	    variableSuffixStream << framesFileNameSuffix << std::setfill('0') << std::setw(2) << frame ; // _frameFF
	    fileNameSuffixes.push_back(variableSuffixStream.str());
	    variableSuffixStream.str(""); // emtpy string, we don't want to accumulate strings among iterations
	}
    }
    std::string dataFileNamePrefixWithDims = OpenCLIPER::Data::buildFileNamePrefix(dataFileNamePrefix, pArraysDims->at(0));
    (reinterpret_cast<Data*>(this))->loadRawData(dataFileNamePrefixWithDims, pArraysDims, pDynDims, fileNameSuffixes,
	    fileNameExtension);
#ifdef KData_DEBUG
    KDATA_CERR("KData size: " << getData()->size() << std::endl);
#endif
}

/**
 * @brief Save data of NDArray objects to a group of files (every NDArray contains one image and is stored into a file).
 *
 * Files of this group have names with the format
 * \<fileNamePrefix\>\<dims\>\<coilsFileNameSuffix\>\<coilNumber\>\<framesFileNameSuffix\>\<frameNumber\>\<fileNameExtension\>,
 * where \<dims\> is a string with the format \<width\>x\<height\>x\<depth\>, \<coilNumber\> is the identifier of the coil
 * used for image adquisition and \<frameNumber\> is the identifier of the time of image adquisition.
 * @param[in] syncSource set the data source used for saving (IMAGE_ONLY or BUFFER_ONLY)
 * @param[in] fileNamePrefix fixed part of the file name before the variable part
 * @param[in] coilsFileNameSuffix part of the file name before the coil number
 * @param[in] framesFileNameSuffix part of the file name before the frame number
 * @param[in] fileNameExtension extension for file names
 */
void KData::saveRawHostData(const std::string &fileNamePrefix,
			    const std::string &coilsFileNameSuffix, const std::string &framesFileNameSuffix,
			    const std::string &fileNameExtension) {
    dimIndexType numFrames = getDynDimsTotalSize();
    numCoilsType numCoils = pNDArrays->size() / numFrames;
    std::stringstream variableSuffixStream;
    std::vector<std::string> fileNameSuffixes;
    for(dimIndexType frame = 0; frame < numFrames; frame++) {
	for(numCoilsType coil = 0; coil < numCoils; coil++) {
	    variableSuffixStream << coilsFileNameSuffix << std::setfill('0') << std::setw(2) << coil; // _coilCC
	    variableSuffixStream << framesFileNameSuffix << std::setfill('0') << std::setw(2) << frame ; // _frameFF
	    fileNameSuffixes.push_back(variableSuffixStream.str());
	    variableSuffixStream.str(""); // emtpy string, we don't want to accumulate strings among iterations
	}
    }

    this->device2Host();

    (reinterpret_cast<Data*>(this))->saveRawData(fileNamePrefix, fileNameSuffixes, fileNameExtension);
#ifdef KData_DEBUG
    KDATA_CERR("KData size: " << getData()->size() << std::endl);
#endif
}

void KData::saveCFLData(const std::string &baseFileName) {
	device2Host();
	saveRawData(baseFileName + ".cfl");
	if (pSensitivityMapsData != nullptr)
		pSensitivityMapsData->saveRawData(baseFileName + CFLSensMapsSuffix);
	if (pSamplingMasksData != nullptr)
		pSamplingMasksData->saveRawData(baseFileName + CFLSampMasksSuffixPixelMask); // pixelmask format
	saveCFLHeader(baseFileName + CFLHeaderExtension);
}

void KData::saveCFLHeader(const std::string &fileName) {
	std::fstream f;
	LPISupport::Utils::openFile(fileName, f, std::ofstream::out|std::ofstream::trunc, "KData::saveCFLHeader");
	const std::vector<dimIndexType>* pSpatialDims = getNDArray(0)->getDims(); // All NDArrays must have the same sptial dimensions
	for (uint i=0; i < pSpatialDims->size(); i++) {
		f << pSpatialDims->at(i) << " ";
	}
	f << getNCoils() << " ";
	f << getDynDims()->at(0) << " "; // Only 1 temporal dimension supported
	f << "1";
	f << std::endl;
	f.close();
}
/**
 * @brief Stores image/volume spatial and temporal dimensions in DataDimsAndStrides field.
 */
void KData::calcDataDims() {
    Data::calcDataDims();
    pDataDimsAndStridesVector->at(NumCoilsPos) = nCoils;
}

/**
 * @brief Creates a new KData object with data following a pattern (odd negative numbers for real part, even negative numbers for imaginary part).
 *
 * @param[in] width width of the data (number of columns)
 * @param[in] height height of the data (number of rows)
 * @param[in] numCoils number of coils
 * @param[in] numFrames number of frames
 * @return a pointer to the new KData object
 */
KData* KData::genTestKData(const std::shared_ptr<CLapp>& pCLapp, dimIndexType width, dimIndexType height, dimIndexType numFrames,
			   numCoilsType numCoils) {
    KDATA_CERR("width: " << width << std::endl << "height: " << height << std::endl);
    std::vector<dimIndexType>* pDimsInputImage; //, *pDimsOutputImage;
    std::vector <complexType>* pImageData;
    complexType elementImage, elementSensitivityMap;
    dimIndexType elementSamplingMask;
    std::vector<NDArray*>* pObjNDArraysImage = new std::vector<NDArray*>();
    std::vector<NDArray*>* pObjNDArraysSensitivityMap = new std::vector<NDArray*>();
    std::vector<NDArray*>* pObjNDArraysSamplingMasks = new std::vector<NDArray*>();
    std::vector<dimIndexType>* pInputDynDims = new std::vector<dimIndexType>({numFrames});
    std::vector<dimIndexType>* pSamplingMasksDynDims = new std::vector<dimIndexType>({numFrames});
    KDATA_CERR("Creating sensitivity maps...\n");
    realType realElement = 1.0;
    for(numCoilsType coilId = 0; coilId < numCoils; coilId++) {
	std::vector <complexType>* pSensitivityMap;
	pSensitivityMap = new std::vector<complexType>();
	for(index1DType i = 0; i < (width * height) * 2; i += 2) {
	    elementSensitivityMap.real((-1.0) * (realElement)); // (-1, -3, -5, -7, ...)
	    elementSensitivityMap.imag((-1.0) * (realElement + 1)); // (-2, -4, -6, -8, ...)
	    pSensitivityMap->push_back(elementSensitivityMap);
	    realElement += 2.0;
	}
	std::vector<dimIndexType>* pDimsSensitivityMap;
	pDimsSensitivityMap = new std::vector<dimIndexType>({width, height});
	NDArray* pObjNDArraySensitivityMap = NDArray::createNDArray<complexType>(pDimsSensitivityMap, pSensitivityMap);
	//new ConcreteNDArray<complexType>(pDimsSensitivityMap, pSensitivityMap);
	KDATA_CERR("Sensitivity map coilId: " << coilId << std::endl);
	KDATA_CERR(pObjNDArraySensitivityMap->hostDataToString("Sensitivity map") << std::endl);
	pObjNDArraysSensitivityMap->push_back(pObjNDArraySensitivityMap);
    }
    SensitivityMapsData* pSensitivityMapsData = new SensitivityMapsData(pCLapp, pObjNDArraysSensitivityMap, numCoils);
    KDATA_CERR("Creating sensitivity maps done.\n");
    KDATA_CERR("Creating KData...\n");
    realElement = 1.0;
    for(dimIndexType dynId = 0; dynId < pInputDynDims->at(0); dynId++) {
	for(numCoilsType coilId = 0; coilId < numCoils; coilId++) {
	    pImageData = new std::vector<complexType>();
	    for(index1DType i = 0; i < (width * height) * 2; i += 2) {
		elementImage.real(realElement); // (1, 3, 5, 7, ...)
		elementImage.imag(realElement + 1); // (2, 4, 6, 8, ...)
		pImageData->push_back(elementImage);
		realElement += 2.0;
	    }
	    pDimsInputImage = new std::vector<dimIndexType>({width, height});
	    //pDimsOutputImage = new std::vector<dimIndexType>(*pDimsInputImage);

	    NDArray* pObjNDArrayImage = NDArray::createNDArray<complexType>(pDimsInputImage, pImageData);
	    KDATA_CERR("Initial data" << std::endl);
	    KDATA_CERR("coilId: " << coilId << "\tdynId: " << dynId << std::endl);
	    KDATA_CERR(pObjNDArrayImage->hostDataToString("NDArray:") << std::endl);

	    KDATA_CERR("Objeto creado, ndims: " << (numberOfDimensionsType) pObjNDArrayImage->getNDims() << std::endl);
	    KDATA_CERR("Creando objeto de la clase KData" << std::endl);

	    pObjNDArraysImage->push_back(pObjNDArrayImage);
	}
    }
    KDATA_CERR("Creating KData done.\n");
    KDATA_CERR("Creating sampling masks...\n");
    elementSamplingMask = 1;
    for(dimIndexType dynId = 0; dynId < pInputDynDims->at(0); dynId++) {
	std::vector <dimIndexType>* pSamplingMask;
	pSamplingMask = new std::vector<dimIndexType>();
	for(index1DType i = 0; i < height; i++) {
	    pSamplingMask->push_back(elementSamplingMask);
	    elementSamplingMask = (elementSamplingMask + 1) % 2;
	}
	std::vector<dimIndexType>* pDimsSamplingMasks;
	pDimsSamplingMasks = new std::vector<dimIndexType>({height});

	NDArray* pObjNDArraySamplingMasks = NDArray::createNDArray<dimIndexType>(pDimsSamplingMasks, pSamplingMask);
	KDATA_CERR("Initial data" << std::endl);
	KDATA_CERR("dynId: " << dynId << std::endl);
	KDATA_CERR(pObjNDArraySamplingMasks->hostDataToString("NDArray:") << std::endl);

	KDATA_CERR("Objeto creado, ndims: " << (numberOfDimensionsType) pObjNDArraySamplingMasks->getNDims() << std::endl);
	// pObjNDArrays only needed for Data constructor but now is abstract => can't be instantiated
	// vector<NDArray*>* pObjNDArrays = new vector<NDArray*>;
	// pObjNDArrays->push_back(pObjNDArray);
	pObjNDArraysSamplingMasks->push_back(pObjNDArraySamplingMasks);
    }
    SamplingMasksData* pSamplingMasksData = new SamplingMasksData(pCLapp, pObjNDArraysSamplingMasks, pSamplingMasksDynDims, width);
    KDATA_CERR("Creating sampling masks done.\n");
    for(unsigned int i = 0; i < pSamplingMasksData->getDynDimsTotalSize(); i ++) {
	KDATA_CERR(pSamplingMasksData->getData()->at(i)->hostDataToString(std::string("Sampling mask (") + std::to_string(i) +
		   std::string(")")));
    }

    Data* pKData = new KData(pCLapp);
    pKData->setData(pObjNDArraysImage);
    pKData->setDynDims(pInputDynDims);
    (dynamic_cast<KData*>(pKData))->setSensitivityMapsData(pSensitivityMapsData);
    (dynamic_cast<KData*>(pKData))->setSamplingMasksData(pSamplingMasksData);
    (dynamic_cast<KData*>(pKData))->setNCoils(numCoils);
    pCLapp->addData(pKData, true);
    delete(pDimsInputImage);
    delete(pInputDynDims);
    delete(pImageData);
    return (dynamic_cast<KData*>(pKData));
}

/**
 * @brief Saves data to a file in matlab format
 *
 * @param[in] fileName name of the file tha data will be saved to
 * @param[in] syncSource source of data: (OpenCL buffer or image)
 * @throw std::invalid_argument if matlab file cannot be opened or the number of matlab variable names is less than the number of
 * matlab variables to be saved ot number of spatial dimensions of data is less than 1
 */
void KData::matlabSave(const std::string &fileName) {
    mat_t* matfp;
    //std::vector<dimIndexType> matVarKDataDimsVector, matVarSensMapsDimsVector, matVarSampMasksDimsVector;
    matfp = Mat_CreateVer(fileName.c_str(), NULL, MAT_FT_DEFAULT);
    if(NULL == matfp) {
	BTTHROW(std::invalid_argument(std::string("Error creating MAT file")  + fileName), "KData::matlabSave");
    }

    this->device2Host();

    // Save dims variable
    if(pMatVarDimsData == nullptr) {
	// Create matlab Dims variable
	pMatVarDimsData.reset(new MatVarDimsData(pCLapp, getNDArray(0)->getDims(), getAllSizesEqual(), getNCoils(), getDynDims()));
    }
    pMatVarDimsData->matlabSaveVariable(matfp);

    // Only image sequences with the same spatial dimensions are supported
    dimIndexType numMatVarKDataElements = pNDArrays->at(0)->size() * getNCoils() * getDynDimsTotalSize();
    if(NDARRAYWIDTH(pNDArrays->at(0)) == 0) {
	BTTHROW(std::invalid_argument("Invalid data size (width is 0)"), "KData::matlabSave");
    }

    MatVarInfo* pMatVarInfo = new MatVarInfo(elementDataType, numMatVarKDataElements);
    std::vector<dimIndexType>* joinedDimensions = new std::vector<dimIndexType>(*(pNDArrays->at(0)->getDims()));
    joinedDimensions->push_back(getNCoils());
    joinedDimensions->insert(std::end(*joinedDimensions), std::begin(*getDynDims()), std::end(*getDynDims()));
    // Update matlab dimensions and rank with spatial dimensions
    // Update matlab dimensions and rank with coils dimensions
    // Update matlab dimensions and rank with temporal dimensions
    pMatVarInfo->updateDimsAndRank(joinedDimensions);
    fillMatlabVarInfo(matfp, matVarNameKData, pMatVarInfo);
    delete(pMatVarInfo);
    joinedDimensions->clear();

    if(pSensitivityMapsData != nullptr) {
	dimIndexType numMatVarSensMapsElements = pSensitivityMapsData->getData()->at(0)->size() * getNCoils();
	pMatVarInfo = new MatVarInfo(pSensitivityMapsData->getElementDataType(), numMatVarSensMapsElements);
	// Update matlab dimensions and rank with spatial dimensions (true for swapping width and depth dimensions)
	// Update matlab dimensions and rank with coils dimensions
	joinedDimensions->insert(std::end(*joinedDimensions), std::begin(*(pSensitivityMapsData->pNDArrays->at(0)->getDims())),
				 std::end(*(pSensitivityMapsData->pNDArrays->at(0)->getDims())));
	joinedDimensions->push_back(pSensitivityMapsData->getNCoils());
	pMatVarInfo->updateDimsAndRank(joinedDimensions);
	pSensitivityMapsData->fillMatlabVarInfo(matfp, SensitivityMapsData::matVarNameSensitivityMaps, pMatVarInfo);
	delete(pMatVarInfo);
	joinedDimensions->clear();
    }
    else {
	KDATA_CERR("KData::matlabSave: KData without sensitivity maps\n");
    }
    if(pSamplingMasksData != nullptr) {
	KDATA_CERR("KData::matlabsave, sampling masks, internal format\n");
	/*for(unsigned int i = 0; i < pSamplingMasksData->getDynDimsTotalSize(); i ++) {
	    KDATA_CERR(pSamplingMasksData->getData()->at(i)->hostDataToString(std::string("Sampling mask (") + std::to_string(i) +
		       std::string(")")));
	}*/
	KDATA_CERR("KData::matlabsave, sampling masks, external format\n");
	/*for(unsigned int i = 0; i < pSamplingMasksData->getDynDimsTotalSize(); i ++) {
	    KDATA_CERR(pSamplingMasksData->getData()->at(i)->hostDataToString(std::string("Sampling mask (") + std::to_string(i) +
		       std::string(")")));
	}*/
	/*pCLapp->addData(pSamplingMasksData, true);
	dimIndexType numMatVarSampMasksElements = pSamplingMasksData->getData()->at(0)->size() * pSamplingMasksData->getDynDimsTotalSize();
	pMatVarInfo = new MatVarInfo(pSamplingMasksData->getElementDataType(), numMatVarSampMasksElements);
	// First dimension of sampling masks is the number of image lines (its height), second and following
	// are the temporal dimensions;
	std::vector<dimIndexType>* pSamplingMasksDimensions = new std::vector<dimIndexType>();
	// Add spatial dimension: number of columns (width)
	pSamplingMasksDimensions->insert(pSamplingMasksDimensions->end(),
					 pSamplingMasksData->pNDArrays->at(0)->getDims()->begin(),
					 pSamplingMasksData->pNDArrays->at(0)->getDims()->end());
// 	 Add number of rows (is 1) for 1-spatial dimension mask: matlab needs at least two dimensions (and a sampling mask must
// 	 * be a row vector with a number of columns equal to the image number of lines if mask format is ROWFORMAT)
	*/
	/*if (pSamplingMasksData->pNDArrays->at(0)->getNDims() < 2) {
	    pSamplingMasksDimensions->push_back(1);
	}
	// Add temporal dimensions
	pSamplingMasksDimensions->insert(pSamplingMasksDimensions->end(),
					 pSamplingMasksData->getDynDims()->begin(),
					 pSamplingMasksData->getDynDims()->end());
	// Update matlab dimensions and rank with sampling mask dimensions dimensions
	pMatVarInfo->updateDimsAndRank(pSamplingMasksDimensions);
	pSamplingMasksData->fillMatlabVarInfo(matfp, SamplingMasksData::matVarNameSamplingMasks, pMatVarInfo);
	delete(pMatVarInfo);*/
	// Format must be reverted to internal after saving (sampling masks may be reused without beeing recreated)
    }
    else {
	KDATA_CERR("KData::matlabSave: KData without sampling masks\n");
    }

    if(pTrajectories != nullptr) {

    }
    else {
	KDATA_CERR("KData::matlabSave: KData without trajectories\n");
    }
    Mat_Close(matfp);
}

} /* namespace OpenCLIPER */
#undef KDATA_DEBUG

