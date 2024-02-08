#include <OpenCLIPER/MatVarDimsData.hpp>
#include <OpenCLIPER/ConcreteNDArray.hpp>
#include <LPISupport/InfoItems.hpp>

#define MATVARDIMSDATA_DEBUG

#if !defined NDEBUG && defined MATVARDIMSDATA_DEBUG
    #define MATVARDIMSDATA_CERR(x) CERR(x)
#else
    #define MATVARDIMSDATA_CERR(x)
    #undef MATVARDIMSDATA_DEBUG
#endif

namespace OpenCLIPER {
/**
* @brief Constructor that stores data of a matlab variable previously read (usually from a KData object) as a group of
* NDArray objects.
*
* @param[in] pMatlabVar matlab variable previously read from file
*/
MatVarDimsData::MatVarDimsData(const std::shared_ptr<CLapp>& pCLapp, matvar_t* pMatlabVar):
		Data(std::type_index(typeid(dimIndexType))) {
    if(pMatlabVar == nullptr) {
	BTTHROW(std::invalid_argument(std::string(errorPrefix) + "pointer to matlab variable for MatVarDimsData is nullptr"), "MatVarDimsData::MatVarDimsData");
    }
    // Vector of temporal dimensions for this object (no temporal dimensions, only spatial)
    std::vector<dimIndexType>* pDynDims = new std::vector<dimIndexType>;
    setDynDims(pDynDims);

    // Number of NDArrays is 1, number of spatial dimensions is 2 (Warning!! minimum matlab variable number of dimensions is 2, a vector is actually a 1xn array)
    dimIndexType numOfNDArraysToBeRead = 1, nSpatialDims = 2;
    MATVARDIMSDATA_CERR("MatVarDimsData reading data...\n");
    ((Data*)(this))->loadMatlabHostData(pMatlabVar, nSpatialDims, numOfNDArraysToBeRead);
    MATVARDIMSDATA_CERR("Done.\n");
    if (getElementDataType() != std::type_index(typeid(dimIndexType))) {
	BTTHROW(std::invalid_argument("Unsupported format for malab dimensions variable (must be a vector of unsigned integers)"), "MatVarDimsData::MatVarDimsData");
    }
    setApp(pCLapp, true);
}

MatVarDimsData::MatVarDimsData(const std::shared_ptr<CLapp>& pCLapp, const std::vector<dimIndexType>* pDataSpatialDims, dimIndexType allSizesEqual, numCoilsType nCoils,
			       const std::vector<dimIndexType>* pDataTemporalDims):
		Data(std::type_index(typeid(dimIndexType))) {
    // Vector storing all dimensions information of Data subclass object with format common to host/kernel code
    std::vector<dimIndexType>* pNDArrayDimsVar = new std::vector<dimIndexType>();
    pNDArrayDimsVar->push_back(pDataSpatialDims->size()); // Number of spatial dimensions
    pNDArrayDimsVar->push_back(allSizesEqual); // 1 if all NDArrays have the same size
    pNDArrayDimsVar->push_back(nCoils); // Number of coils
    pNDArrayDimsVar->push_back(pDataTemporalDims->size()); // Number of temporal dimensions
    // Append every temporal dimension
    pNDArrayDimsVar->insert(std::end(*pNDArrayDimsVar), std::begin(*pDataTemporalDims), std::end(*pDataTemporalDims));
    // Append every spatial dimension
    pNDArrayDimsVar->insert(std::end(*pNDArrayDimsVar), std::begin(*pDataSpatialDims), std::end(*pDataSpatialDims));

    // Spatial dims of this object
    std::vector<dimIndexType>* pDimsVarSpatialDims = new std::vector<dimIndexType>;
    pDimsVarSpatialDims->push_back(pNDArrayDimsVar->size());
    NDArray* pLocalNDArray = NDArray::createNDArray(pDimsVarSpatialDims, pNDArrayDimsVar);
    std::vector<NDArray*>* pNDArrays = new std::vector<NDArray*>();
    pNDArrays->push_back(pLocalNDArray);
    internalSetData(pNDArrays);
    setApp(pCLapp, true);
}

/**
 * @brief Constructor that creates a MatVarDimsData object from another MatVarDimsData object.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData source MatVarDimsData object
 * @param[in] copyData if true also data (not only dimensions) are copied from sourceData object to this object
 */
MatVarDimsData::MatVarDimsData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<MatVarDimsData>& sourceData, bool copyData): Data(sourceData, copyData) {
    setApp(pCLapp, copyData);
}

/**
 * @brief Constructor that creates an uninitialized MatVarDimsData object from another MatVarDimsData object.
 * Element data type for the new object is specified by the caller
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData Object to copy data structure from
 * @param[in] newElementDataType Data type of the newly created object
 */
MatVarDimsData::MatVarDimsData(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<MatVarDimsData>& sourceData, ElementDataType newElementDataType):
		   Data(sourceData, newElementDataType) {

    setApp(pCLapp, false);
}

/**
 * @brief Virtual copy "constructor"
 * @param[in] deepCopy copy stored data if true; only data structure if false (shallow copy)
 * @return A copy of this object
 */
std::shared_ptr<Data> MatVarDimsData::clone(bool deepCopy) const {
    return std::make_shared<MatVarDimsData>(pCLapp, shared_from_this(), deepCopy);
}

/**
 * @brief Virtual copy "constructor"
 * @param[in] newElementDataType data type of the newly created object. Shallow copy is implied.
 * @return An empty copy of this object that contains elements of type "newElementDataType"
 */
std::shared_ptr<Data> MatVarDimsData::clone(ElementDataType newElementDataType) const {
    return std::make_shared<MatVarDimsData>(pCLapp, shared_from_this(), newElementDataType);
}

void MatVarDimsData::getAllDims(std::vector<dimIndexType>* pCompleteSpatialDimsVector, numCoilsType& numCoils, std::vector<dimIndexType>* pTemporalDimsVector) {
    dimIndexType nSpatialDims, nTemporalDims, spatialDimSize, temporalDimSize;
    dimIndexType firstSpatialDimIndex, lastSpatialDimIndex, firstTemporalDimIndex, lastTemporalDimIndex;
    const ConcreteNDArray<dimIndexType>* pTypedSourceData = dynamic_cast<const ConcreteNDArray<dimIndexType>*>(this->getNDArray(0));
    nSpatialDims = pTypedSourceData->getHostData()->at(NSD_POS);
    numCoils = pTypedSourceData->getHostData()->at(NCOILS_POS);
    nTemporalDims = pTypedSourceData->getHostData()->at(NTD_POS);
    firstTemporalDimIndex = NTD_POS + 1;
    lastTemporalDimIndex = firstTemporalDimIndex + nTemporalDims - 1;
    firstSpatialDimIndex = lastTemporalDimIndex + 1;
    lastSpatialDimIndex = firstSpatialDimIndex + nSpatialDims - 1;
    pCompleteSpatialDimsVector->resize(0);
    for(auto i = firstSpatialDimIndex; i <= lastSpatialDimIndex; i++) {
	spatialDimSize = pTypedSourceData->getHostData()->at(i);
	pCompleteSpatialDimsVector->push_back(spatialDimSize);
    }
    pTemporalDimsVector->resize(0);
    for(auto i = firstTemporalDimIndex; i <= lastTemporalDimIndex; i++) {
	temporalDimSize = pTypedSourceData->getHostData()->at(i);
	pTemporalDimsVector->push_back(temporalDimSize);
    }
}

void MatVarDimsData::matlabSaveVariable(mat_t* matfp) {
    // No temporal dimensions, number of elements is the size of the unique NDArray
    dimIndexType numMatVarElements = pNDArrays->at(0)->size();
    if(NDARRAYWIDTH(pNDArrays->at(0)) == 0) {
	BTTHROW(std::invalid_argument("Invalid data size (width is 0)"), "MatVarDimsData::matlabSaveVariable");
    }
    MatVarInfo* pMatVarInfo = new MatVarInfo(elementDataType, numMatVarElements);
    // Update matlab dimensions and rank with spatial dimensions (no number of coils nor temporal dimensions)
    pMatVarInfo->updateDimsAndRank(pNDArrays->at(0)->getDims());
    fillMatlabVarInfo(matfp, matVarNameDims, pMatVarInfo);
}
}
#undef MATVARDIMSDATA_DEBUG
