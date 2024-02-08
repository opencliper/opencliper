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
#ifndef INCLUDE_OPENCLIPER_DATA_HPP_
#define INCLUDE_OPENCLIPER_DATA_HPP_

#include <OpenCLIPER/NDArray.hpp>
#include <OpenCLIPER/DeviceDataProperties.hpp>
#include <OpenCLIPER/MatVarInfo.hpp>
#include <OpenCLIPER/hostKernelFunctions.hpp>
#include <LPISupport/Utils.hpp>
#include <thread>

namespace OpenCLIPER {
class CLapp;
class NDArray;

/// Class Data - Class that includes data and properties common to k-space and x-space images.
class Data: public std::enable_shared_from_this<Data> {
	/// XData must have access to private part of Data
	friend class XData;
	/// KData must have access to private part of Data
	friend class KData;
	/// SensitivityMapsData must have access to private part of Data
	friend class SensitivityMapsData;
	/// SamplingMasksData must have access to private part of Data
	friend class SamplingMasksData;
	/// DataProperties must have access to protected part of Data
	friend class DeviceDataProperties;
	/// CLapp must have access to protected part of Data
	friend class CLapp;
	/// Trajectory must have access to private part of Data
	friend class Trajectories;
	/// MatVarDimsData must have access to private part of Data
	friend class MatVarDimsData;

    public:
        //---------------------------------
	// Constructors and destructor
	//---------------------------------

	// Constructors: empty data
	explicit Data(ElementDataType elementDataType = TYPEID_COMPLEX);

	// From a bunch of NDarrays
	Data(std::vector<NDArray*>*& pNDArrays, ElementDataType elementDataType = TYPEID_COMPLEX);
	Data(std::vector<NDArray*>*& pNDArrays, std::vector<dimIndexType>*& pDynDims, ElementDataType elementDataType = TYPEID_COMPLEX);

	// From given dimensions, uninitialized: single NDArray
	Data(dimIndexType width, ElementDataType elementDataType = TYPEID_COMPLEX);
	Data(dimIndexType width, dimIndexType height, ElementDataType elementDataType = TYPEID_COMPLEX);
	Data(dimIndexType width, dimIndexType height, dimIndexType depth, ElementDataType elementDataType = TYPEID_COMPLEX);
	Data(std::vector<dimIndexType>*& pSpatialDims, ElementDataType elementDataType = TYPEID_COMPLEX);

	// From given dimensions, uninitialized: several NDArrays with no temporal dimensions
	Data(dimIndexType numNDArrays, std::vector<dimIndexType>*& pSpatialDims, ElementDataType elementDataType = TYPEID_COMPLEX);
	Data(std::vector<std::vector<dimIndexType>*>*& pSpatialDims, ElementDataType elementDataType = TYPEID_COMPLEX);

	// From given dimensions, uninitialized: several NDArrays with spatial and temporal dimensions
	Data(dimIndexType numNDArrays, std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, ElementDataType elementDataType = TYPEID_COMPLEX);
	Data(std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, ElementDataType elementDataType = TYPEID_COMPLEX);

	// From given dimensions and an std::vector: single NDArray
        template<typename T>
	Data(dimIndexType width, std::vector<T>*& pData);
        template<typename T>
	Data(dimIndexType width, dimIndexType height, std::vector<T>*& pData);
        template<typename T>
	Data(dimIndexType width, dimIndexType height, dimIndexType depth, std::vector<T>*& pData);
        template<typename T>
	Data(std::vector<dimIndexType>*& pSpatialDims, std::vector<T>*& pData);

	// From given dimensions and a vector of std::vectors: severall NDArrays with spatial and temporal dimensions
        template<typename T>
	Data(std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<T>*>*& pData);

        template<typename T>
	Data(std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<T>*>*& pData);

        // From another Data object (copy constructors)
	Data(Data* sourceData, bool copyData = false, bool copyDataToDevice = true);
	Data(std::shared_ptr<Data> sourceData, bool copyData = false, bool copyDataToDevice = true);
	Data(std::shared_ptr<Data> sourceData, ElementDataType newElementDataType);

        // Destructor
	virtual ~Data();


	//---------------------------------
        // Getters / Setters
	//---------------------------------

	// Getters
	std::vector<const NDArray*>*   getNDArrays() const;
	const NDArray*                 getNDArray(dimIndexType NDArrayIndex) const;
	std::vector<const NDArray*>*   getData() const;
	const NDArray*                 getDataAtDynPos(const std::vector<dimIndexType>& dynIndexes) const;
	const index1DType              get1DIndexFromDynPos(const std::vector<dimIndexType>& dynIndexes) const;
	const void*                    getHostBuffer(dimIndexType index = 0) const;
	void*                          getHostBuffer(dimIndexType index = 0) { return const_cast<void*>(const_cast<const Data*>(this)->getHostBuffer(index)); }
	cl::Buffer*                    getCompleteDeviceBuffer();
	cl::Buffer*                    getDeviceBuffer();
	cl::Buffer*                    getDeviceBuffer(dimIndexType index);
	void*                          getHIPDeviceBuffer();
	cl::Buffer*                    getDataDimsAndStridesDeviceBuffer();

	/**
	 * @brief Gets the CLapp associated to this Data object
	 * @return pointer to the CLapp
	 */
	const std::shared_ptr<CLapp> getApp() const {
	    return pCLapp;
	}

	/**
	 * @brief Returns true of all NDArrays have the same spatial dimensions.
	 * @return true if all NDArrays have the same spatial dimensions
	 */
	const bool getAllSizesEqual() const {
	    return allSizesEqual;
	}

	/**
	 * @brief Get temporal dimensions vector.
	 * @return vector of temporal dimensions
	 */
	const std::vector<dimIndexType>* getDynDims() const {
		if(pDynDims.get()->empty())
			BTTHROW(std::invalid_argument("Invalid number of temporal dimensions (0). Did you forget a call to setDynDims?"), "Data::getDynDims");
	    return pDynDims.get();
	}

	/**
	 * @brief Get size of temporal dimensions vector.
	 * @return size of vector of temporal dimensions
	 */
	numberOfDimensionsType getNDynDims() const {
	    return pDynDims->size();
	}

	const index1DType getDynDimsTotalSize() const;

	/**
	 * @brief Gets vector of spatial and temporal data dimensions stored in host memory.
	 * @return pointer to vector of spatial and temporal data dimensions
	 */
	const std::vector<dimIndexType>* getDataDimsAndStridesVector() const {
	    return pDataDimsAndStridesVector.get();
	}

	const void* getDataDimsAndStridesHostBuffer() const;

	/**
	 * @brief Gets value describing data type used for data elments in this object.
	 * @return value describing data type used
	 */
	ElementDataType getElementDataType() const {
	    return elementDataType;
	}

	//Setters
	void setData(NDArray*& pNDArray, bool copyData = false);
	void setData(std::vector<NDArray*>*& pNDArrays, bool copyData = false);
	void setDynDims(std::vector<dimIndexType>*& pDynDims);

	//---------------------------------
	// Other methods
	//---------------------------------
	void setApp(std::shared_ptr<CLapp> pCLapp, bool copyDataToDevice);
	const std::vector<const NDArray*>* getFragment(FragmentSpecif specif);
	static std::string buildFileNamePrefix(const std::string &prefix, const std::vector<dimIndexType>* pDims);
	static std::string buildFileNameSuffix(std::string suffix = "", std::string fileExtension = "raw");
	void saveRawData(const std::string &fileName);
	void saveRawData(const std::string &fileNamePrefix, std::vector<std::string> &fileNameSuffixes,
			     const std::string &fileNameExtension = ".raw");

	// Pure virtual method, must be reimplemented (dimensions depend on the existience of number of coils class field)
	virtual void calcDataDims() = 0;
	virtual void calcDataStrides(dimIndexType deviceMemBaseAddrAlignInBytes);
	virtual void calcDimsAndStridesVector(dimIndexType deviceMemBaseAddrAlignInBytes);
	static std::map<std::string, matvar_t*>* readMatlabVariablesFromFile(std::string fileName, std::vector<std::string>
		variableNames);
	static std::map<std::string, matvar_t*>* readMatlabVariablesFromFile(std::string fileName);

	/**
	 * @brief Returns the size of the base element of stored data
	 *
	 * @return the value of the size of the base element
	 */
	size_t getElementSize() {
	    return elementSize;
	}
	/**
	 * @brief returns the handle associated to this Data object (used in Data list managed by CLapp object)
	 *
	 * @return the associated handle
	 */
	DataHandle getHandle() const {
	    return myDataHandle;
	}
	const std::string hostBufferToString(std::string title, dimIndexType index);
	virtual void device2Host(bool queueFinish = true);

	void waitLoadEnd();

	//---------------------------------
        // host/kernel functions
	//---------------------------------

        // Functions related to spatial dimensions
        uint getNumSpatialDims();
        uint getNumNDArrays();
        uint getNDArray1DIndex(uint coilIndex, uint temporalDimIndexes[]);
        uint getNDArrayTotalSize(uint NDArray1DIndex);
        uint getNDArrayStride();
        uint getSpatialDimStride(uint spatialDimIndex, uint NDArray1DIndex);
        uint getSpatialDimSize(uint spatialDimIndex, uint NDArray1DIndex);

        // Functions related to coils
        uint getNumCoils();
        uint getCoilDim();
        uint getCoilStride(uint NDArray1DIndex);

        // Functions related to temporal dimensions
        uint getNumTemporalDims();
        uint getTemporalDim(uint tempDim);
        uint getTemporalDimStride(uint temporalDimIndex, uint NDArray1DIndex);
        uint getTemporalDimSize(uint temporalDimIndex);

        // Functions related to dimensions in general (not particular to spatial, coil or temporal dimensions)
        uint getDimSize(uint dimIndex, uint NDArray1DIndex);
        uint getElementStride(uint NDArray1DIndex);
        uint getDimStride(uint dimIndex, uint NDArray1DIndex);
        const void* getNextElement(dimIndexType nDim, dimIndexType curNDArray, dimIndexType curOffset);

	//---------------------------------
	// Show related stuff
	//---------------------------------
	struct ShowParms {
	    unsigned timeDimension;	// No. of temporal (dynamic) dimension to use as time when showing videos
	    float fps;			// frames per second to show, if there is at least one temporal (dynamic) dimension
	    float scaleFactor;		// Window scaling factor: 1.0 = no scaling, more than 1 = upscale, less than 1 = downscale
    	    bool normalize;		// normalize data before showing it
	    float normFactor;		// If normalize==true, multiply all pixels by this value before showing (0 means the average value of all NDArrays shifted to 0.5)
	    float waitTime;		// milliseconds. 0 means wait forever (or key pressed, if keyEnds=true)
	    ComplexPart complexPart;	// part of complex data to show (if the data to show is complex, ignored otherwise)
	    std::string title;		// name and identifier of display window
	    unsigned firstFrame;	// First frame to show (to display a subset of frames within whole the temporal sequence)
	    unsigned nFrames;		// Number of frames to show, starting from firstFrame. 0 means as many frames as available along timeDimension
	    unsigned loops;		// # times to loop from firstFrame to lastFrame. 0 means until key pressed
	    bool keyEnds;		// Key press finishes the show loop. Note: if loops=0 and keyEnds=false, you have an infinite loop


	    ShowParms(): timeDimension(0), fps(30), scaleFactor(1), normalize(true), normFactor(0), waitTime(0), complexPart(ComplexPart::ABS),
			 title("Data view"), firstFrame(0), nFrames(0), loops(0), keyEnds(true) {}
	};

	void show(ShowParms* parms = nullptr);
	void checkMatvarDims(matvar_t* matvar, const std::vector<dimIndexType>* pDims);

    protected:
	// Virtual "constructors"
	virtual std::shared_ptr<Data> clone(bool deepCopy=true) const = 0;
	virtual std::shared_ptr<Data> clone(ElementDataType newElementDataType) const = 0;

	/**
	 * @brief Sets handle (used in Data list managed by CLapp object) for this object.
	 *
	 * @param handle new handle value for the object
	 */
	void setHandle(DataHandle handle) {
	    myDataHandle = handle;
	}
	void loadRawData(const std::string &fileNamePrefix, std::vector<std::vector< dimIndexType >*>*& pArraysSpatialDims,
			     std::vector <dimIndexType>*& pTemporalDims, std::vector<std::string> &fileNameSuffixes,
			     const std::string &fileNameExtension = ".raw");
	void loadRawData(const std::string &fileName, const std::vector< dimIndexType>* pArraysSpatialDims, dimIndexType numOfNDArrays, bool asyncLoad = false);
	void internalSetData(std::vector<NDArray*>*& pNDArrays, bool copyData=false);
	void createFromNDArraysVector(std::vector<NDArray*>*& pNDArrays, std::vector<dimIndexType>*& pTempDims, ElementDataType elementDataType);
	void createFromDimensions(dimIndexType numNDArrays, std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, ElementDataType elementDataType);
	void calcDataAlignedSize(dimIndexType NDArray1DIndex, dimIndexType deviceMemBaseAddrAlignInBytes);
	void checkValiditySpatialDimensions(const std::vector<dimIndexType>* pDims);
	void checkValiditySpatialDimensions(const std::vector<std::vector<dimIndexType>*>* pArraysDims);
	void loadMatlabHostData(matvar_t* matvar, dimIndexType numOfSpatialDimensions, dimIndexType numNDArraysToRead);
	void fillMatlabVarInfo(mat_t* matfp, std::string matVarname, MatVarInfo* pMatVarInfo);
	void readDimsMatlabVar(const matvar_t* pDimsMatVar, std::vector<dimIndexType>* pCompleteSpatialDimsVector, numCoilsType& numCoils, std::vector<dimIndexType>* pTemporalDimsVector);

	// Associations
	/** @brief Images data stored as a vector of pointers to NDArrays objects */
	std::unique_ptr<std::vector<std::unique_ptr<NDArray>>> pNDArrays = nullptr;
	/** @brief temporal field for storing data for getData() method (internally this clases uses smart pointer, externally data is
	 * returned as standar pointers */
	std::vector<const NDArray*>* pNDArraysForGet = nullptr;
	/** @brief Pointer to vector of temporal dimensions of the stored group of images */ // default is empty
	std::unique_ptr<std::vector<dimIndexType>> pDynDims;
	/** @brief image spatial and temporal dimensions and their strides (field data type is valid for kernel parameters) */
	std::unique_ptr<std::vector<dimIndexType>> pDataDimsAndStridesVector;
    private:
	static constexpr const char* errorPrefix = "OpenCLIPER::Data::";
	/**
	 * @brief 1 if all NDArrays have the same size for every spatial dimensions, 0 otherwise
	 *
	 * (bool type must not be used because is
	 * not supported in OpenCL kernels, and this field has to be copied to data structures used by kernels)
	 */
	dimIndexType allSizesEqual = 1;
	/// @brief Size (in bytes) of NDArray element
	size_t elementSize = 0;
	/// @brief Data type of NDArray element (see @ref ElementDataType)
	ElementDataType elementDataType = std::type_index(typeid(void));
	/** @brief pointer to CLapp object */
	std::shared_ptr<CLapp> pCLapp = nullptr;
	DataHandle myDataHandle = INVALIDDATAHANDLE;
	bool checkNDArrayIndex(dimIndexType index, dimIndexType nDArraySize);

	void commonFieldInitialization(ElementDataType elementDataType);
	void checkNDArraysSizesAndSetAllSizesEqual();
	std::unique_ptr<std::thread> pFileLoaderThread = nullptr;
	std::shared_ptr<Data> dataForSaving;
};
}
/* namespace OpenCLIPER */
#endif /* INCLUDE_OPENCLIPER_DATA_HPP_ */
