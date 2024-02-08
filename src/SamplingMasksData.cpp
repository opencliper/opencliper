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
 * SamplingMasksData.cpp
 *
 *  Created on: 28 de oct. de 2016
 *      Author: manrod
 */

#include <OpenCLIPER/SamplingMasksData.hpp>
#include <OpenCLIPER/ConcreteNDArray.hpp>

// Uncomment to show class-specific debug messages
#define SAMPLINGMASKSDATA_DEBUG

#if !defined NDEBUG && defined SAMPLINGMASKSDATA_DEBUG
#define SAMPLINGMASKSDATA_CERR(x) CERR(x)
#else
#define SAMPLINGMASKSDATA_CERR(x)
#undef SAMPLINGMASKSDATA_DEBUG
#endif

namespace OpenCLIPER {

/**
 * @brief Constructor that create a SamplingMasksData object from a vector of Data (containing NDArrays, each one contains the
 * sampling mask used for images captured at the same time frame).
 *
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] pMasks pointer to Data object containing sampling mask information (the format used in this input data is the
 * external format for sampling masks, see <a href="#details">detailed description of the class</a>)
 */
SamplingMasksData::SamplingMasksData(const std::shared_ptr<CLapp>& pCLapp, Data*& pMasks, dimIndexType kDataNumCols, ElementDataType elementDataType): Data(elementDataType) {
	// ROWMASK format => elementDataType of type dimIndex
	// PIXELMASK format => elementDataType of type cl_uchar
	if ((elementDataType != TYPEID_INDEX) && (elementDataType != TYPEID_CL_UCHAR)) {
		BTTHROW(std::invalid_argument("Invalid mask format (supported: ROWMASK and PIXELMASK"), "SamplingMasksData::SamplingMasksData");
	}
	this->pMasks.reset(pMasks);
	pMasks = nullptr;
	this->kDataNumCols = kDataNumCols;
	// If ROWMASK format, convert mask to PIXELMASK format
	if (elementDataType == TYPEID_INDEX) {
		this->masksFormat = ROWMASK;
		rowmask2pixelmask();
	}
	setApp(pCLapp, true);
}

/**
 * @brief Constructor that create a SamplingMasksData object from a vector of NDArrays (each one contains the sampling mask of one frame)
 *
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] pData pointer to vector of NDArrays containing sampling mask information (the format used in this input data is the
 * external format for sampling masks, see <a href="#details">detailed description of the class</a>)
 * @param[in] pDynDims pointer to vector of tempoeral dimensions
 */
SamplingMasksData::SamplingMasksData(const std::shared_ptr<CLapp>& pCLapp, std::vector<NDArray*>*& pData,
		std::vector<dimIndexType>*& pDynDims, dimIndexType kDataNumCols, ElementDataType elementDataType):
    		Data(pData, pDynDims, elementDataType) {
	this->kDataNumCols = kDataNumCols;
	// ROWMASK format => elementDataType of type dimIndex
	// PIXELMASK format => elementDataType of type cl_uchar
	if ((elementDataType != TYPEID_INDEX) && (elementDataType != TYPEID_CL_UCHAR)) {
		BTTHROW(std::invalid_argument("Invalid mask format (supported: ROWMASK and PIXELMASK"), "SamplingMasksData::SamplingMasksData");
	}
	// If ROWMASK format, convert mask to PIXELMASK format
	if (elementDataType == TYPEID_INDEX) {
		masksFormat = ROWMASK;
		rowmask2pixelmask();
	}
	setApp(pCLapp, true);
}

/**
 * @brief Constructor that creates a SamplingMaskData object from a group if files in raw format (the format used in these files is
 * the external format for sampling masks, see <a href="#details">detailed description of the class</a>)
 *
 * Files of this group have names with the format
 * \<dataFileNamePrefix\>\<dims\>\<framesFileNameSuffix\>\<frameNumber\>\<fileNameExtension\>,
 * where \<dims\> is a string with the format \<width\>x\<height\>x\<depth\> and \<frameNumber\> is the identifier
 * of the temporal frame of image adquisition.

 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] dataFileNamePrefix name of the file in raw format
 * @param[in,out] pArraysDims pointer to vectors with spatial dimensions
 * @param[in,out] pDynDims pointer to vector of tempoeral dimensions
 * @param[in] framesFileNameSuffix name suffix for name part depending on frame index
 * @param[in] fileNameExtension extension for the name of the file
 */
SamplingMasksData::SamplingMasksData(const std::shared_ptr<CLapp>& pCLapp, const std::string &dataFileNamePrefix,
		std::vector<std::vector< dimIndexType >*>*& pArraysDims,
		std::vector <dimIndexType>*& pDynDims, dimIndexType kDataNumCols,
		const std::string& framesFileNameSuffix, const std::string& fileNameExtension,
		ElementDataType elementDataType):
				   Data(elementDataType) {
	this->kDataNumCols = kDataNumCols;
	// ROWMASK format => elementDataType of type dimIndex
	// PIXELMASK format => elementDataType of type cl_uchar
	if ((elementDataType != TYPEID_INDEX) && (elementDataType != TYPEID_CL_UCHAR)) {
		BTTHROW(std::invalid_argument("Invalid mask format (supported: ROWMASK and PIXELMASK"), "SamplingMasksData::SamplingMasksData");
	}
	loadRawHostData(dataFileNamePrefix, pArraysDims, pDynDims, framesFileNameSuffix, fileNameExtension);
	// If ROWMASK format, convert mask to PIXELMASK format
	if (elementDataType == TYPEID_INDEX) {
		masksFormat = ROWMASK;
		rowmask2pixelmask();
	}
	setApp(pCLapp, true);
}

SamplingMasksData::SamplingMasksData(const std::shared_ptr<CLapp>& pCLapp, const std::string &fileName,
		const std::vector<dimIndexType>* pArraySpatialDims,
		const std::vector <dimIndexType>* pTemporalDims, dimIndexType kDataNumCols,
		ElementDataType elementDataType):
				   Data(elementDataType) {
	this->kDataNumCols = kDataNumCols;
	if ((elementDataType != TYPEID_INDEX) && (elementDataType != TYPEID_CL_UCHAR)) {
		BTTHROW(std::invalid_argument("Invalid mask format (supported: ROWMASK and PIXELMASK"), "SamplingMasksData::SamplingMasksData");
	}
	std::vector<dimIndexType>* pAuxTemporalDims = new std::vector<dimIndexType>(*(pTemporalDims));
	setDynDims(pAuxTemporalDims);
	loadRawData(fileName, pArraySpatialDims, getDynDimsTotalSize());
	if (elementDataType == TYPEID_INDEX) {
		masksFormat = ROWMASK;
		rowmask2pixelmask();
	} else {
		masksFormat = PIXELMASK;
	}
	setApp(pCLapp, true);
}

/**
 * @brief Constructor that creates a SamplingMaskData object from a matlab format file (the format used in this file is the
 * external format for sampling masks, see <a href="#details">detailed description of the class</a>)
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] pMatlabVar pointer to matlab var containing data of the sampling masks
 * @param[in] numOfSpatialDimensions number of spatial dimensions of data
 */
SamplingMasksData::SamplingMasksData(const std::shared_ptr<CLapp>& pCLapp, matvar_t* pMatlabVar,
		const std::vector<dimIndexType>* pKDataSpatialDimensions) {
	this->kDataNumCols = pKDataSpatialDimensions->at(WIDTHPOS);
	if(pMatlabVar == nullptr) {
		BTTHROW(std::invalid_argument("pointer to matlab variable for sampling masks is nullptr"), "SamplingMasksData::SamplingMasksData");
	}

	dimIndexType numOfSpatialDimensions;
	// Mask in ROWMASK format
	if (pMatlabVar->class_type == MAT_C_UINT32) {
		elementDataType == TYPEID_INDEX;
		// number of spatial dimensions of sampling mask data is
		// number of spatial dimensions of KData - 1 for a mask with ROWMASK format
		numOfSpatialDimensions = pKDataSpatialDimensions->size() - 1 ;
		this->masksFormat = ROWMASK;
	} else if (pMatlabVar->class_type == MAT_C_UINT8) {
		elementDataType == TYPEID_CL_UCHAR;
		// Mask in PIXELMASK format
		// number of spatial dimensions of sampling mask data is
		// number of spatial dimensions of KData for a mask with PIXELMASK format
		numOfSpatialDimensions = pKDataSpatialDimensions->size() ;
		this->masksFormat = PIXELMASK;
	} else {
		BTTHROW(std::invalid_argument("Invalid mask format (supported: ROWMASK and PIXELMASK"), "SamplingMasksData::SamplingMasksData");
	}
    commonFieldInitialization(elementDataType);
    pNDArraysForGet = new std::vector<const NDArray*>;
    pNDArrays = nullptr;

	// Vector of temporal dimensions vector for Data object
	std::vector<dimIndexType>* pDynDims = new std::vector<dimIndexType>;

	// Loop over temporal dimensions for creating temporal dimensions vector for Data object, first temporal dimension comes after
	// last spatial dimension
	for(int i = numOfSpatialDimensions; i < pMatlabVar->rank; i++) {
		pDynDims->push_back(pMatlabVar->dims[i]);
	}
	setDynDims(pDynDims);

	// Number of spatial dimensions data to be read (NDArrays) is the product of all temporal dimensions values
	dimIndexType numOfNDArraysToBeRead = getDynDimsTotalSize();
	SAMPLINGMASKSDATA_CERR("SamplingMasks reading data...\n");
	((Data*)(this))->loadMatlabHostData(pMatlabVar, numOfSpatialDimensions, numOfNDArraysToBeRead);
	SAMPLINGMASKSDATA_CERR("Done.\n");
	// If ROWMASK format, convert mask to PIXELMASK format
	if (pMatlabVar->class_type == MAT_C_UINT32) {
            rowmask2pixelmask();
	}
	setApp(pCLapp, true);
}

/**
 * @brief Constructor that creates a SamplingMasksData object from another SamplingMasksData object (at least temporal
 * dimensions are copied, data is copied if copyData parameter is true).
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData source SensitivityMapsData object
 * @param[in] copyData if true also data (not only dimensions) are copied from sourceData object to this object
 */
SamplingMasksData::SamplingMasksData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<SamplingMasksData>& sourceData,
		bool copyData): Data(sourceData, copyData) {
	this->kDataNumCols = sourceData->getKDataNumCols();
	this->masksFormat = sourceData->getMasksFormat();
	if(sourceData->getMasksFormat() == ROWMASK) {
		rowmask2pixelmask();
	}
	setApp(pCLapp, copyData);
}

/**
 * @brief Constructor that creates an uninitialized SamplingMasksData object from a given Data object.
 * Element data type for the new object is specified by the caller
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData Object to copy data structure from
 * @param[in] newElementDataType Data type of the newly created object
 */
SamplingMasksData::SamplingMasksData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<SamplingMasksData>& sourceData, ElementDataType newElementDataType):
				   Data(sourceData, newElementDataType) {
	this->masksFormat = sourceData->getMasksFormat();
	this->kDataNumCols = sourceData->getKDataNumCols();
	if(sourceData->getMasksFormat() == ROWMASK) {
		rowmask2pixelmask();
	}
	setApp(pCLapp, false);
}

/**
 * @brief Destructor.
 *
 */
SamplingMasksData::~SamplingMasksData() {
#ifdef SamplingMasksData_DEBUG
	SAMPLINGMASKSDATA_CERR("~SamplingMasksData() begins..." << std::endl);
	SAMPLINGMASKSDATA_CERR("~SamplingMasksData() ends..." << std::endl);
#endif
}

/**
 * @brief Virtual copy "constructor"
 * @param[in] deepCopy copy stored data if true; only data structure if false (shallow copy)
 * @return A copy of this object
 */
std::shared_ptr<Data> SamplingMasksData::clone(bool deepCopy) const {
	return std::make_shared<SamplingMasksData>(pCLapp, shared_from_this(), deepCopy);
}

/**
 * @brief Virtual copy "constructor"
 * @param[in] newElementDataType data type of the newly created object. Shallow copy is implied.
 * @return An empty copy of this object that contains elements of type "newElementDataType"
 */
std::shared_ptr<Data> SamplingMasksData::clone(ElementDataType newElementDataType) const {
	return std::make_shared<SamplingMasksData>(pCLapp, shared_from_this(), newElementDataType);
}

/**
 * @brief Load data of a group of files to NDArray objects (every file contains one sampling mask and is stored into a NDArray object).
 *
 * Files of this group have names with the format
 * \<dataFileNamePrefix\>\<dims\>\<framesFileNameSuffix\>\<frameNumber\>\<fileNameExtension\>,
 * where \<dims\> is a string with the format \<width\>x\<height\>x\<depth\> and \<frameNumber\> is the identifier
 * of the temporal frame of image adquisition.
 * @param[in] fileNamePrefix fixed part of the file name before the variable part
 * @param[in] pArraysDims pointer to a vector of pointers to vectors with the spatial dimensions of every sampling mask
 * @param[in] pDynDims pointer to vector with temporal dimensions (number of frames)
 * @param[in] framesFileNameSuffix part of the file name before the frame number
 * @param[in] fileNameExtension extension for file names
 */
void SamplingMasksData::loadRawHostData(const std::string &fileNamePrefix, std::vector<std::vector< dimIndexType >*>*& pArraysDims,
		std::vector <dimIndexType>*& pDynDims, const std::string &framesFileNameSuffix,
		const std::string &fileNameExtension) {
	index1DType numFrames = 1;
	for(index1DType i = 0; i < pDynDims->size(); i++) {
		numFrames = numFrames * pDynDims->at(i);
	}

	std::stringstream variableSuffixStream;
	std::vector<std::string> fileNameSuffixes;
	for(dimIndexType frame = 0; frame < numFrames; frame++) {
		variableSuffixStream << framesFileNameSuffix << std::setfill('0') << std::setw(2) << frame ; // _frameFF
		fileNameSuffixes.push_back(variableSuffixStream.str());
		variableSuffixStream.str(""); // emtpy string, we don't want to accumulate strings among iterations
	}
	std::string dataFileNamePrefixWithDims = OpenCLIPER::Data::buildFileNamePrefix(fileNamePrefix, pArraysDims->at(0));
	(reinterpret_cast<Data*>(this))->loadRawData(dataFileNamePrefixWithDims, pArraysDims, pDynDims, fileNameSuffixes,
			fileNameExtension);
	SAMPLINGMASKSDATA_CERR("SamplingMasksData size: " << getData()->size() << std::endl);
	if (getElementDataType() == TYPEID_INDEX) {
		masksFormat = ROWMASK;
		rowmask2pixelmask();
	} else if (getElementDataType() == TYPEID_CL_UCHAR) {
		masksFormat = PIXELMASK;
	} else {
		BTTHROW(std::invalid_argument("Invalid mask format (supported: ROWMASK and PIXELMASK"), "SamplingMasksData::SamplingMasksData");
	}
}

void SamplingMasksData::saveCFLHeader(const std::string &fileName) {
	std::fstream f;
	LPISupport::Utils::openFile(fileName, f, std::ofstream::out|std::ofstream::trunc, "XData::saveCFLHeader");
	const std::vector<dimIndexType>* pSpatialDims = getNDArray(0)->getDims(); // All NDArrays must have the same sptial dimensions
	for (uint i=0; i < pSpatialDims->size(); i++) {
		f << pSpatialDims->at(i) << " ";
	}
	f << "1 "; // 1 coil
	f << getDynDims()->at(0); // Only 1 temporal dimension supported
	f << std::endl;
	f.close();
}

/**
 * @brief Converts sampling mask information as a vector of 0 or 1 values (1 if line has been captured, 0 otherwise) to a
 * group of line numbers (only numbers of the lines to be blanked, i.e., marked as not used)
 *
 * External format example for a image of 8 lines: {1, 0, 1, 1, 0, 0, 0, 1}, corresponding internal format {1, 4, 5, 6}.
 */
void SamplingMasksData::rowmask2rowstoblank() {
	if(masksFormat != ROWMASK) {
		return;
	}
	//pRowNumbersToBeBlankedVector->resize(0);
	dimIndexType elementData;
	std::vector<NDArray*>* pNDArraysBlankingInfo = new std::vector<NDArray*>;
	// Number of image lines is the first spatial dimension of the first NDArray (all NDArray with the same number of lines)
	numberOfImageLines = getData()->at(0)->getDims()->at(0);
	for(dimIndexType i = 0; i < getDynDimsTotalSize(); i++) {  // loop for all frames
		std::vector<dimIndexType>* pListOfRawsToBlankInFrame;
		pListOfRawsToBlankInFrame = new std::vector<dimIndexType>;
		// Only 1 spatial dimension per NDArray
		for(dimIndexType elementIndex = 0; elementIndex < getData()->at(i)->getDims()->at(0); elementIndex ++) {
			elementData = static_cast<const ConcreteNDArray<dimIndexType>*>(getData()->at(i))->getHostData()->at(elementIndex);
			switch(elementData) {
			case 0: // if 0 line has not been captured, it must be blanked
				pListOfRawsToBlankInFrame->push_back(elementIndex);
				break;
			case 1: // if 1 line has been captured, it must not be blanked
				break;
			default:
				BTTHROW(std::invalid_argument("sampling maks data element must be 0 or 1 (it is " +
						std::to_string(elementData) + ")"), "SamplingMasksData::rowmask2rowstoblank");
			};
		}
		std::vector<dimIndexType>* NDArrayBlankingInfoDims;
		NDArrayBlankingInfoDims = new std::vector<dimIndexType>;
		NDArrayBlankingInfoDims->push_back(pListOfRawsToBlankInFrame->size());
		NDArray* pNDArrayBlankingInfo;
		pNDArrayBlankingInfo = NDArray::createNDArray<dimIndexType>(NDArrayBlankingInfoDims, pListOfRawsToBlankInFrame);
		pNDArraysBlankingInfo->push_back(pNDArrayBlankingInfo);
	}
	// Only data field (group of NDArrays containing data and spatial dimensions) is overwritten, temporal dimensions are not changed.
	setData(pNDArraysBlankingInfo);
	masksFormat = ROWSTOBLANK;
}

/**
 * @brief Converts sampling mask information as a vector of 0 or 1 values (1 if line has been captured, 0 otherwise) to a
 * group of line numbers (only numbers of the lines to be blanked, i.e., marked as not used)
 *
 * External format example for a image of 8 lines: {1, 0, 1, 1, 0, 0, 0, 1}, corresponding internal format {1, 4, 5, 6}.
 */
void SamplingMasksData::rowmask2pixelmask() {
	if(masksFormat != ROWMASK) {
		return;
	}
	//pRowNumbersToBeBlankedVector->resize(0);
	dimIndexType elementData;
	std::vector<NDArray*>* pNDArraysBlankingInfo = new std::vector<NDArray*>;
	std::vector<cl_uchar>* pPixelMaskFormatNDArray;
	cl_uchar pixelMaskFormatElement;
	// Number of image lines is the first spatial dimension of the first NDArray (all NDArray with the same number of lines)
	numberOfImageLines = getData()->at(0)->getDims()->at(0);
	for(dimIndexType frame = 0; frame < getDynDimsTotalSize(); frame++) {  // loop for all frames
		// Only 1 spatial dimension per NDArray (width or number of columns of row vector)
		pPixelMaskFormatNDArray = new std::vector<cl_uchar>; // set to nullptr during createNDArray
		for(dimIndexType elementIndex = 0; elementIndex < NDARRAYWIDTH(getNDArray(frame)); elementIndex ++) {
			elementData = static_cast<const ConcreteNDArray<dimIndexType>*>(getNDArray(frame))->getHostData()->at(elementIndex);
			switch(elementData) {
			case 0: // if 0 line has not been captured, it must be blanked
				pixelMaskFormatElement = 0;
				break;
			case 1: // if 1 line has been captured, it must not be blanked
				pixelMaskFormatElement = 255;
				break;
			default:
				BTTHROW(std::invalid_argument("sampling maks data element must be 0 or 1 (it is " +
						std::to_string(elementData) + ")"), "SamplingMasksData::rowmask2pixelmask");
			};
			// Create row of elements 0 (row has not been captured) or 255 (has been captured)
			std::vector<cl_uchar> pixelMaskFormatRow = std::vector<cl_uchar>(kDataNumCols, pixelMaskFormatElement);
			if (pPixelMaskFormatNDArray->empty()) {
				*pPixelMaskFormatNDArray = std::move(pixelMaskFormatRow);
			}
			else {
				pPixelMaskFormatNDArray->reserve(pPixelMaskFormatNDArray->size() + pixelMaskFormatRow.size());
				std::move(std::begin(pixelMaskFormatRow), std::end(pixelMaskFormatRow), std::back_inserter(*pPixelMaskFormatNDArray));
				pixelMaskFormatRow.clear();
			}
		}
		std::vector<dimIndexType>* NDArrayBlankingInfoDims;
		NDArrayBlankingInfoDims = new std::vector<dimIndexType>;
		NDArrayBlankingInfoDims->resize(2);
		NDArrayBlankingInfoDims->at(WIDTHPOS) = kDataNumCols;
		NDArrayBlankingInfoDims->at(HEIGHTPOS) = NDARRAYWIDTH(getNDArray(frame));
		NDArray* pNDArrayBlankingInfo;
		pNDArrayBlankingInfo = NDArray::createNDArray<cl_uchar>(NDArrayBlankingInfoDims, pPixelMaskFormatNDArray);
		pNDArraysBlankingInfo->push_back(pNDArrayBlankingInfo);
	}
	// Only data field (group of NDArrays containing data and spatial dimensions) is overwritten, temporal dimensions are not changed.
	masksFormat = PIXELMASK;
	elementDataType = TYPEID_CL_UCHAR;
	elementSize = NDArray::getElementSize(elementDataType);
	setData(pNDArraysBlankingInfo);
}

/**
 * @brief Converts sampling mask information as group of line numbers, starting at 0, (only numbers of the lines to be blanked,
 * marked as not used) to a vector of 0 or 1 values (1 if line has been captured, 0 otherwise).
 *
 * External format example for a image of 8 lines: {1, 0, 1, 1, 0, 0, 0, 1}, corresponding internal format {1, 4, 5, 6}.
 *
 */
void SamplingMasksData::rowstoblank2rowmask() {
	if(masksFormat != ROWSTOBLANK) {
		return;
	}
	dimIndexType elementData, lastLineNotProcessedId;
	std::vector<NDArray*>* pNDArraysExternalFormatBlankingInfo = new std::vector<NDArray*>;
	for(dimIndexType i = 0; i < getDynDimsTotalSize(); i++) {  // loop for all frames
		std::vector<dimIndexType>* pListOfStatusOfLines;
		pListOfStatusOfLines = new std::vector<dimIndexType>;
		lastLineNotProcessedId = 0;
		// Only 1 spatial dimension per NDArray
		for(dimIndexType elementIndex = 0; elementIndex < getData()->at(i)->getDims()->at(0); elementIndex++) {
			elementData = static_cast<const ConcreteNDArray<dimIndexType>*>(getData()->at(i))->getHostData()->at(elementIndex);
			for(dimIndexType lineIndex = lastLineNotProcessedId; lineIndex < elementData; lineIndex++) {
				pListOfStatusOfLines->push_back(1); // all lines captured previous to a non-captured line
			}
			pListOfStatusOfLines->push_back(0); // line with number equal to elementData has not been captured;
			lastLineNotProcessedId = elementData + 1;
		}
		/*
		 * If the list of ids of lines not captured has been processed and the id of last line not processed is
		 * less than number of image lines - 1, we have to add 1s for the range
		 * [last_line_not_processed_id, number_of_image_lines-1] to show that this lines have been captured
		 */
		for(dimIndexType lineIndex = lastLineNotProcessedId; lineIndex < numberOfImageLines; lineIndex++) {
			pListOfStatusOfLines->push_back(1);
		}
		std::vector<dimIndexType>* NDArrayExternalFormatBlankingInfoDims;
		NDArrayExternalFormatBlankingInfoDims = new std::vector<dimIndexType>;
		NDArrayExternalFormatBlankingInfoDims->push_back(numberOfImageLines);
		NDArray* pNDArrayExternalFormatBlankingInfo;
		pNDArrayExternalFormatBlankingInfo = NDArray::createNDArray<dimIndexType>(NDArrayExternalFormatBlankingInfoDims, pListOfStatusOfLines);
		pNDArraysExternalFormatBlankingInfo->push_back(pNDArrayExternalFormatBlankingInfo);
	}
	// Only data field (group of NDArrays containing data and spatial dimensions) is overwritten, temporal dimensions are not changed.
	setData(pNDArraysExternalFormatBlankingInfo);
	masksFormat = ROWMASK;
}

} /* namespace OpenCLIPER */
#undef SAMPLINGMASKSDATA_DEBUG
