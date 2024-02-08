#include<fstream>
#include<sstream>
#include<string>
#include <OpenCLIPER/Data.hpp>
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <OpenCLIPER/processes/Complex2Real.hpp>
#include <OpenCLIPER/processes/SumReduce.hpp>
#include <OpenCLIPER/processes/NormalizeShow.hpp>
#include <OpenCLIPER/processes/ReshapeShow.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <CvPlot/cvplot.h>
#include <OpenCLIPER/MatVarDimsData.hpp>
#include <OpenCLIPER/hostKernelFunctions.hpp>
#include <OpenCLIPER/InvalidDimension.hpp>

// Uncomment to show class-specific debug messages
#define DATA_DEBUG

#if !defined NDEBUG && defined DATA_DEBUG
    #define DATA_CERR(x) CERR(x)
#else
    #define DATA_CERR(x)
    #undef DATA_DEBUG
#endif

namespace OpenCLIPER {

//*********************************
// Constructors and destructor
//*********************************

//------------
// Empty Data
//------------
/**
 * @brief Constructor that creates an empty Data object.
 * @param[in] elementDataType data type of base element
 */
Data::Data(ElementDataType elementDataType) {
    commonFieldInitialization(elementDataType); // It also calls setDynDims
    pNDArraysForGet = new std::vector<const NDArray*>;
    pNDArrays = nullptr;
}

//-------------------------
// From a bunch of NDarrays
//-------------------------

/**
 * @brief Constructor that creates a Data object from a vector of NDArrays.
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion. Temporal dimensions vector is initialized to a 1D vector
 * with a value equal to the number of NDArrays.
 * @param[in,out] pNDArrays pointer to vector of pointers to NDArray objects
 * @param[in] elementDataType data type of base element
 */
Data::Data(std::vector<NDArray*>*& pNDArrays, ElementDataType elementDataType) {
    // As there is no explicit pDynDims vector, we create a 1D-vector with a value
    // equal to the number of NDArrays
    // Error: SenstivityMaps (subclass of Data) does not have temporal dimensions (number of temporal dimensions must be 0)
    //std::vector<dimIndexType>* pDynDims = new std::vector<dimIndexType>({pNDArrays->size()});
    std::vector<dimIndexType>* pDynDims = new std::vector<dimIndexType>();
    createFromNDArraysVector(pNDArrays, pDynDims, elementDataType);
}

/**
 * @brief Constructor that creates a Data object from a vector of NDArrays and a vector with temporal dimensions (see @ref
 * createFromNDArraysVector).
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in,out] pNDArrays pointer to vector of pointers to NDArray objects
 * @param[in,out] pDynDims pointer to vector of temporal dimensions
 * @param[in] elementDataType data type of base element
 */
Data::Data(std::vector<NDArray*>*& pNDArrays, std::vector<dimIndexType>*& pDynDims, ElementDataType elementDataType) {
    createFromNDArraysVector(pNDArrays, pDynDims, elementDataType);
}


//------------------------------------------------------
// From given dimensions, uninitialized: single NDArray
//------------------------------------------------------

/**
 * @brief Constructor that creates an uninitialized Data object containing a single 1-dimensional NDArray from given width
 *
 * @param[in] width size of the NDArray along the first dimension
 * @param[in] elementDataType data type of base element
 */
Data::Data(dimIndexType width, ElementDataType elementDataType) {
    // Create an empty temporal dims vector
    auto pTempDims = new std::vector<dimIndexType>();

    // Create a vector of pointers to 1 copy of a vector containing spatial dims
    auto pSpatialDims = new std::vector<dimIndexType>({width});
    auto pSpatialDimsCopies = new std::vector<std::vector<dimIndexType>*>(1, pSpatialDims);

    createFromDimensions(1, pSpatialDimsCopies, pTempDims, elementDataType);
}

/**
 * @brief Constructor that creates an uninitialized Data object containing a single 2-dimensional NDArray from given width and height
 *
 * @param[in] width size of the NDArray along the first dimension
 * @param[in] height size of the NDArray along the second dimension
 * @param[in] elementDataType data type of base element
 */
Data::Data(dimIndexType width, dimIndexType height, ElementDataType elementDataType) {
    // Create an empty temporal dims vector
    auto pTempDims = new std::vector<dimIndexType>();

    // Create a vector of pointers to 1 copy of a vector containing spatial dims
    auto pSpatialDims = new std::vector<dimIndexType>({width, height});
    auto pSpatialDimsCopies = new std::vector<std::vector<dimIndexType>*>(1, pSpatialDims);

    createFromDimensions(1, pSpatialDimsCopies, pTempDims, elementDataType);
}

/**
 * @brief Constructor that creates an uninitialized Data object containing a single 3-dimensional NDArray from given width, height and depth
 *
 * @param[in] width size of the NDArray along the first dimension
 * @param[in] height size of the NDArray along the second dimension
 * @param[in] depth size of the NDArray along the third dimension
 * @param[in] elementDataType data type of base element
 */
Data::Data(dimIndexType width, dimIndexType height, dimIndexType depth, ElementDataType elementDataType) {
    // Create an empty temporal dims vector
    auto pTempDims = new std::vector<dimIndexType>();

    // Create a vector of pointers to 1 copy of a vector containing spatial dims
    auto pSpatialDims = new std::vector<dimIndexType>({width, height, depth});
    auto pSpatialDimsCopies = new std::vector<std::vector<dimIndexType>*>(1, pSpatialDims);

    createFromDimensions(1, pSpatialDimsCopies, pTempDims, elementDataType);
}

/**
 * @brief Constructor that creates an uninitialized Data object containing a single NDArray from given spatial dimensions
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in,out] pSpatialDims pointer to vector of spatial dimensions
 * @param[in] elementDataType data type of base element
 */
Data::Data(std::vector<dimIndexType>*& pSpatialDims, ElementDataType elementDataType) {
    // Create an empty temporal dims vector
    auto pTempDims = new std::vector<dimIndexType>();

    // Create a vector of pointers to 1 copy of pSpatialDims
    auto pSpatialDimsCopies = new std::vector<std::vector<dimIndexType>*>(1, pSpatialDims);

    createFromDimensions(1, pSpatialDimsCopies, pTempDims, elementDataType);

    pSpatialDims = nullptr;
}


//------------------------------------------------------------------------------------
// From given dimensions, uninitialized: several NDArrays with no temporal dimensions
//------------------------------------------------------------------------------------

/**
 * @brief Constructor that creates an uninitialized Data object containing several equally-sized NDArrays with given spatial dimensions and no temporal dimensions
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in] numNDArrays number of NDArrays the new Data object should contain
 * @param[in,out] pSpatialDims pointer to vector of spatial dimensions
 * @param[in] elementDataType data type of base element
 */
Data::Data(dimIndexType numNDArrays, std::vector<dimIndexType>*& pSpatialDims, ElementDataType elementDataType) {
    // Create an empty temporal dims vector
    std::vector<dimIndexType>* pTempDims = new std::vector<dimIndexType>();

    auto pSpatialDimsCopies = new std::vector<std::vector<dimIndexType>*>(1, pSpatialDims);
    createFromDimensions(numNDArrays, pSpatialDimsCopies, pTempDims, elementDataType);

    pSpatialDims = nullptr;
}

/**
 * @brief Constructor that creates an uninitialized Data object containing several arbitrarily-sized NDArrays with given spatial dimensions and no temporal dimensions.
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in,out] pSpatialDims pointer to vector of pointers to spatial dimensions vectors
 * @param[in] elementDataType data type of base element
 */
Data::Data(std::vector<std::vector<dimIndexType>*>*& pSpatialDims, ElementDataType elementDataType) {
    // Create an empty temporal dims vector
    std::vector<dimIndexType>* pTempDims = new std::vector<dimIndexType>();

    createFromDimensions(pSpatialDims->size(), pSpatialDims, pTempDims, elementDataType);
}

//---------------------------------------------------------------------------------------------
// From given dimensions, uninitialized: several NDArrays with spatial and temporal dimensions
//---------------------------------------------------------------------------------------------

/**
 * @brief Constructor that creates an uninitialized Data object containing several equally-sized NDArrays with given spatial and temporal dimensions
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in] numNDArrays number of NDArrays the new Data object should contain
 * @param[in,out] pSpatialDims pointer to vector of spatial dimensions
 * @param[in,out] pTempDims pointer to vector of temporal dimensions
 * @param[in] elementDataType data type of base element
 */
Data::Data(dimIndexType numNDArrays, std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, ElementDataType elementDataType) {
    auto pSpatialDimsCopies = new std::vector<std::vector<dimIndexType>*> (1, pSpatialDims);
    createFromDimensions(numNDArrays, pSpatialDimsCopies, pTempDims, elementDataType);
    delete pSpatialDimsCopies;
    pSpatialDims = nullptr;
}

/**
 * @brief Constructor that creates an uninitialized Data object containing several arbitrarily-sized NDArrays with given spatial and temporal dimensions.
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in,out] pSpatialDims pointer to vector of pointers to spatial dimensions vectors
 * @param[in,out] pTempDims pointer to vector of temporal dimensions
 * @param[in] elementDataType data type of base element
 */
Data::Data(std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, ElementDataType elementDataType) {
    createFromDimensions(pSpatialDims->size(), pSpatialDims, pTempDims, elementDataType);
}


//---------------------------------------------------------
// From given dimensions and an std::vector: single NDArray
//---------------------------------------------------------

/**
 * @brief Constructor that creates a Data object containing a single 1-dimensional NDArray from given width and a data vector
 *
 * @param[in] width size of the NDArray along the first dimension
 * @param[in,out] pData pointer to the data to be transferred to the NDArray
 * @param[in] elementDataType data type of base element
 */
template<typename T>
Data::Data(dimIndexType width, std::vector<T>*& pData) {
    // Create an empty temporal dims vector
    auto pTempDims = new std::vector<dimIndexType>();

    // Create a vector containing spatial dims
    auto pSpatialDims = new std::vector<dimIndexType>({width});

    // Create an NDArray from spatial dims and data
    auto pNDArray = NDArray::createNDArray(pSpatialDims, pData);

    // Create a vector of pointers to NDArrays from our single NDArray
    auto pNDArrays = new std::vector<NDArray*>(1, pNDArray);

    // Finish creation of this object
    createFromNDArraysVector(pNDArrays, pTempDims, std::type_index(typeid(T)));
}

template Data::Data(dimIndexType width, std::vector<complexType>*& pData);
template Data::Data(dimIndexType width, std::vector<realType>*& pData);
template Data::Data(dimIndexType width, std::vector<dimIndexType>*& pData);
template Data::Data(dimIndexType width, std::vector<cl_uchar>*& pData);

/**
 * @brief Constructor that creates a Data object containing a single 2-dimensional NDArray from given width and height,  and a data vector
 *
 * @param[in] width size of the NDArray along the first dimension
 * @param[in] height size of the NDArray along the second dimension
 * @param[in,out] pData pointer to the data to be transferred to the NDArray
 * @param[in] elementDataType data type of base element
 */
template<typename T>
Data::Data(dimIndexType width, dimIndexType height, std::vector<T>*& pData) {
    // Create an empty temporal dims vector
    auto pTempDims = new std::vector<dimIndexType>();

    // Create a vector containing spatial dims
    auto pSpatialDims = new std::vector<dimIndexType>({width, height});

    // Create an NDArray from spatial dims and data
    auto pNDArray = NDArray::createNDArray(pSpatialDims, pData);

    // Create a vector of pointers to NDArrays from our single NDArray
    auto pNDArrays = new std::vector<NDArray*>(1, pNDArray);

    // Finish creation of this object
    createFromNDArraysVector(pNDArrays, pTempDims, std::type_index(typeid(T)));
}

template Data::Data(dimIndexType width, dimIndexType height, std::vector<complexType>*& pData);
template Data::Data(dimIndexType width, dimIndexType height, std::vector<realType>*& pData);
template Data::Data(dimIndexType width, dimIndexType height, std::vector<dimIndexType>*& pData);
template Data::Data(dimIndexType width, dimIndexType height, std::vector<cl_uchar>*& pData);

/**
 * @brief Constructor that creates a Data object containing a single 3-dimensional NDArray from given width, height and depth, and a data vector
 *
 * @param[in] width size of the NDArray along the first dimension
 * @param[in] height size of the NDArray along the second dimension
 * @param[in] depth size of the NDArray along the third dimension
 * @param[in,out] pData pointer to the data to be transferred to the NDArray
 * @param[in] elementDataType data type of base element
 */
template<typename T>
Data::Data(dimIndexType width, dimIndexType height, dimIndexType depth, std::vector<T>*& pData) {
    // Create an empty temporal dims vector
    auto pTempDims = new std::vector<dimIndexType>();

    // Create a vector containing spatial dims
    auto pSpatialDims = new std::vector<dimIndexType>({width, height, depth});

    // Create an NDArray from spatial dims and data
    auto pNDArray = NDArray::createNDArray(pSpatialDims, pData);

    // Create a vector of pointers to NDArrays from our single NDArray
    auto pNDArrays = new std::vector<NDArray*>(1, pNDArray);

    // Finish creation of this object
    createFromNDArraysVector(pNDArrays, pTempDims, std::type_index(typeid(T)));
}

template Data::Data(dimIndexType width, dimIndexType height, dimIndexType depth, std::vector<complexType>*& pData);
template Data::Data(dimIndexType width, dimIndexType height, dimIndexType depth, std::vector<realType>*& pData);
template Data::Data(dimIndexType width, dimIndexType height, dimIndexType depth, std::vector<dimIndexType>*& pData);
template Data::Data(dimIndexType width, dimIndexType height, dimIndexType depth, std::vector<cl_uchar>*& pData);

/**
 * @brief Constructor that creates a Data object containing a single NDArray from given spatial dimensions and a data vector
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in,out] pSpatialDims pointer to vector of spatial dimensions
 * @param[in,out] pData pointer to the data to be transferred to the NDArray
 * @param[in] elementDataType data type of base element
 */
template<typename T>
Data::Data(std::vector<dimIndexType>*& pSpatialDims, std::vector<T>*& pData) {
    // Create an empty temporal dims vector
    auto pTempDims = new std::vector<dimIndexType>();

    // Create an NDArray from spatial dims and data
    auto pNDArray = NDArray::createNDArray(pSpatialDims, pData);

    // Create a vector of pointers to NDArrays from our single NDArray
    auto pNDArrays = new std::vector<NDArray*>(1, pNDArray);

    // Finish creation of this object
    createFromNDArraysVector(pNDArrays, pTempDims, std::type_index(typeid(T)));
}

template Data::Data(std::vector<dimIndexType>*& pSpatialDims, std::vector<complexType>*& pData);
template Data::Data(std::vector<dimIndexType>*& pSpatialDims, std::vector<realType>*& pData);
template Data::Data(std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pData);
template Data::Data(std::vector<dimIndexType>*& pSpatialDims, std::vector<cl_uchar>*& pData);


//-----------------------------------------------------------------------------------------------------------
// From given dimensions and a vector of std::vectors: severall NDArrays with spatial and temporal dimensions
//-----------------------------------------------------------------------------------------------------------
/**
 * @brief Constructor that creates a Data object containing several NDArrays equal in size from given spatial dimensions and a data vector
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in,out] pSpatialDims pointer to vector of spatial dimensions (all NDArrays equal in size)
 * @param[in,out] pTempDims pointer to vector of temporal dimensions
 * @param[in,out] pData pointer to the data to be transferred to the NDArray: one std::vector<>* per NDArray
 * @param[in] elementDataType data type of base element
 */
template<typename T>
Data::Data(std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<T>*>*& pData) {
    // Create a vector of pointers to NDArrays to store data
    auto pNDArrays = new std::vector<NDArray*>;

    // Create NDArrays from spatial dims and data and store them in our NDArray vector
    for(unsigned i=0; i < pData->size(); i++) {
        // Each NDArray has its own copy of spatial dimensions. Feed them their own copy of pSpatialDims
        auto pSpatialDimsCopy = new std::vector<dimIndexType> (*pSpatialDims);
        pNDArrays->push_back(NDArray::createNDArray(pSpatialDimsCopy, pData->at(i)));
    }

    // Free input vectors and set them to null
    delete(pData);
    pData=nullptr;
    delete(pSpatialDims);
    pSpatialDims=nullptr;

    // Finish creation of this object
    createFromNDArraysVector(pNDArrays, pTempDims, std::type_index(typeid(T)));
}

template Data::Data(std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<complexType>*>*& pData);
template Data::Data(std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<realType>*>*& pData);
template Data::Data(std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<dimIndexType>*>*& pData);
template Data::Data(std::vector<dimIndexType>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<cl_uchar>*>*& pData);

/**
 * @brief Constructor that creates a Data object containing several NDArrays of different size from given spatial dimensions and a data vector
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in,out] pSpatialDims pointer to vector of vectors of spatial dimensions: one std::vector<>* per nDArray
 * @param[in,out] pTempDims pointer to vector of temporal dimensions
 * @param[in,out] pData pointer to the data to be transferred to the NDArray: one std::vector<>* per NDArray
 * @param[in] elementDataType data type of base element
 */
template<typename T>
Data::Data(std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<T>*>*& pData) {
    // Create a vector of pointers to NDArrays to store data
    auto pNDArrays = new std::vector<NDArray*>;

    // Create NDArrays from spatial dims and data and store them in our NDArray vector
    for(unsigned i=0; i < pData->size(); i++)
        pNDArrays->push_back(NDArray::createNDArray(pSpatialDims->at(i), pData->at(i)));

    // Free now-empty input vectors and set them to null
    delete(pData);
    pData=nullptr;
    delete(pSpatialDims);
    pSpatialDims=nullptr;

    // Finish creation of this object
    createFromNDArraysVector(pNDArrays, pTempDims, std::type_index(typeid(T)));
}

template Data::Data(std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<complexType>*>*& pData);
template Data::Data(std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<realType>*>*& pData);
template Data::Data(std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<dimIndexType>*>*& pData);
template Data::Data(std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, std::vector<std::vector<cl_uchar>*>*& pData);


//---------------------------------------------
// From another Data object (copy constructors)
//---------------------------------------------

/**
 * @brief Constructor that creates a Data object with spatial and temporal dimensions set, dimensions are got from
 * another Data object.
 *
 * @param[in] sourceData Data object source of data, spatial and temporal dimensions
 * @param[in] copyData if true also data (not only dimensions) are copied from NDArray sourceData object to this object
 */
Data::Data(Data* sourceData, bool copyData, bool copyToDevice) {
	sourceData->pCLapp->device2Host(sourceData->myDataHandle, true);
    commonFieldInitialization(sourceData->getElementDataType());
    pNDArraysForGet = new std::vector<const NDArray*>;
    std::vector<NDArray*>* pLocalData = new std::vector<NDArray*>;
    std::vector<dimIndexType>* pLocalDynDims = new std::vector<dimIndexType>(*(sourceData->getDynDims()));
    setDynDims(pLocalDynDims);
    for(dimIndexType i = 0; i <  sourceData->getData()->size(); i++) {
	NDArray* pLocalNDArray;
	auto dims = new std::vector<dimIndexType>(*(sourceData->getNDArray(i)->getDims()));
	if (sourceData->getNDArray(i)->getDataHandle() != INVALIDDATAHANDLE) {
		pLocalNDArray = NDArray::createNDArray(sourceData->getHostBuffer(i), dims, elementDataType);
	} else {
		pLocalNDArray = NDArray::createNDArray(sourceData->getNDArray(i)->getHostDataAsVoidPointer(), dims, elementDataType);
	}
	//pLocalNDArray = NDArray::createNDArray(sourceData->getHostBuffer(i), dims, elementDataType);
	//pLocalNDArray = NDArray::createNDArray(sourceData->getNDArray(i), copyData, elementDataType);
	pLocalData->push_back(pLocalNDArray);
    }
    setData(pLocalData, copyToDevice);
}

/**
 * @brief Constructor that creates a Data object with spatial and temporal dimensions set, from another Data object.
 * Data element type is the same as in the source Data object.
 *
 * @param[in] sourceData Data object source of data, spatial and temporal dimensions
 * @param[in] copyData if true also data (not only dimensions) are copied from NDArray sourceData object to this object
 */
Data::Data(std::shared_ptr<Data> sourceData, bool copyData, bool copyToDevice): Data(sourceData.get(), copyData, copyToDevice) {
}

/**
 * @brief Constructor that creates an empty Data object with spatial and temporal dimensions set, from another Data object.
 * Data element type is specified by the caller.
 *
 * @param[in] sourceData Data object source of data, spatial and temporal dimensions
 * @param[in] newElementDataType Data type of the newly created object
 */
Data::Data(std::shared_ptr<Data> sourceData, ElementDataType newElementDataType) {
    commonFieldInitialization(newElementDataType);
    pNDArraysForGet = new std::vector<const NDArray*>;
    std::vector<NDArray*>* pLocalData = new std::vector<NDArray*>;
    std::vector<dimIndexType>* pLocalDynDims = new std::vector<dimIndexType>(*(sourceData->getDynDims()));
    setDynDims(pLocalDynDims);
    for(dimIndexType i = 0; i < sourceData->getData()->size(); i++) {
	NDArray* pLocalNDArray;
	pLocalNDArray = NDArray::createNDArray(sourceData->getData()->at(i), false, elementDataType);
	pLocalData->push_back(pLocalNDArray);
    }
    setData(pLocalData);
}


//-----------
// Destructor
//-----------

/**
 * @brief Destructor, frees previously reserved memory for data structures.
 */
Data::~Data() {
    DATA_CERR("~Data() begins..." << std::endl);

    // Must wait for load calls to end before trying to delete loaded data
    waitLoadEnd();

    if(myDataHandle != INVALIDDATAHANDLE) {
	pCLapp->delData(myDataHandle);
    }
    // Free host data vectors
    if(pNDArrays != nullptr) {
	for(dimIndexType i = 0; i < pNDArrays->size(); i++) {
	    pNDArrays->at(i) = nullptr;
	}
    }

    // Free vector of dimensions and strides
    pDataDimsAndStridesVector = nullptr;

    pCLapp = nullptr;

    DATA_CERR("~Data() ends..." << std::endl);
}

/**
 * @brief Creates a Data object from a vector of NDArrays and a vector of temporal dimensions.
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in,out] pNDArrays pointer to vector of pointers to NDArray objects
 * @param[in,out] pDynDims pointer to vector of temporal dimensions
 */
void Data::createFromNDArraysVector(std::vector<NDArray*>*& pNDArrays, std::vector<dimIndexType>*& pTempDims, ElementDataType elementDataType) {
    // Set various class attributes
    commonFieldInitialization(elementDataType);
    pNDArraysForGet = new std::vector<const NDArray*>;
    setDynDims(pTempDims);

    // Set data from passed argument
    internalSetData(pNDArrays);
}

/**
 * @brief Create an uninitialized Data object consisting of n NDArrays with given spatial and temporal dimensions.
 *
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this
 * object, and parameter value is set to nullptr after method completion.
 * @param[in] numNDArrays number of NDArrays the new Data object should contain
 * @param[in,out] pSpatialDims pointer to vector of pointers to spatial dimensions vectors of each NDArray. The pointed-to vector should contain either as many pointers as
NDArrays to create or just one pointer. If just one pointer is passwd, all NDArrays shall be equal size
 * @param[in,out] pTempDims pointer to vector of temporal dimensions
 * @param[in] elementDataType data type of base element
 */
void Data::createFromDimensions(dimIndexType numNDArrays, std::vector<std::vector<dimIndexType>*>*& pSpatialDims, std::vector<dimIndexType>*& pTempDims, ElementDataType elementDataType) {
    // Check if we are given just one vector in pSpatialDims or as many vectors as numNDArrays. Set increment accordingly.
    dimIndexType numSpatialDims = pSpatialDims->size();
    dimIndexType spatialDimsIncrement;
    if(numSpatialDims == numNDArrays)
        spatialDimsIncrement = 1;
    else if(numSpatialDims == 1)
        spatialDimsIncrement = 0;
    else
        BTTHROW(std::invalid_argument("Size of pSpatialDims should be either equal to numNDArrays or 1"), "Data::Data(numNDArrays, pSpatialDims, pTempDims, elementDataType");

    // Set various class attributes
    commonFieldInitialization(elementDataType);
    pNDArraysForGet = new std::vector<const NDArray*>;
    setDynDims(pTempDims);  // disowns pTempDims from caller

    // Create empty NDArrays and push them back into this object
    pNDArrays = std::unique_ptr <std::vector<std::unique_ptr<NDArray>>>(new std::vector<std::unique_ptr<NDArray>>);
    std::unique_ptr<NDArray> unique_ptr_NDArray;
    dimIndexType spatialDimsIndex = 0;
    for(dimIndexType ndArrayIndex = 0; ndArrayIndex < numNDArrays; ndArrayIndex++) {
	// pSpatialDims may have numNDArrays components or just one.
	// As createNDArray will take ownership of passed pSpatialDims, create local copies on the fly
	auto pLocalSpatialDims = new std::vector<dimIndexType> (*pSpatialDims->at(spatialDimsIndex));
	NDArray* pNDArray = NDArray::createNDArray(pLocalSpatialDims, elementDataType);

	unique_ptr_NDArray = std::unique_ptr <NDArray> (pNDArray);
	pNDArrays->push_back(std::move(unique_ptr_NDArray));

	spatialDimsIndex += spatialDimsIncrement;
    }

    // Disown pSpatialDims components
    for(dimIndexType pSpatialDimsIndex = 0; pSpatialDimsIndex < pSpatialDims->size(); pSpatialDimsIndex++)
	delete(pSpatialDims->at(pSpatialDimsIndex));

    // Disown pSpatialDims itself
    delete(pSpatialDims);
    pSpatialDims = nullptr;

    // Set allSizesEqual flag
    checkNDArraysSizesAndSetAllSizesEqual();
}

/**
 * @brief Load contents of matlab variable into a Data object (containing a group of NDArray objects)
 * @param[in] matvar matlab array variable previously read from file
 * @param[in] numOfSpatialDimensions number of spatial dimensions in matlab variable (rest of dimensions can be temporal dimensions or
 * number of coils)
 * @param[in] numNDArraysToRead number of NDArrays to be read from matlab variable
 * @throw std::invalid_argument if element datatype is not supported
 */
void Data::loadMatlabHostData(matvar_t* matvar, dimIndexType numOfSpatialDimensions, dimIndexType numNDArraysToRead) {
#ifdef DATA_DEBUG
    BEGIN_TIME(bTReadingKDataMatVar);
    DATA_CERR("  reading data from matlab variable... ");
#endif
    switch(matvar->class_type) {
	case MAT_C_SINGLE:
	    if(matvar->isComplex) {
		elementDataType = TYPEID_COMPLEX;
	    }
	    else {
		elementDataType = TYPEID_REAL;
	    }
	    break;
	case MAT_C_UINT32:
	    if(!matvar->isComplex) {
		elementDataType = TYPEID_INDEX;
	    }
	    else {
		BTTHROW(std::invalid_argument("Unsupported element data type in matlab data"), "Data::loadMatlabHostData");
	    }
	    break;
	case MAT_C_UINT8:
	    if(!matvar->isComplex) {
		elementDataType = TYPEID_CL_UCHAR;
	    }
	    else {
		BTTHROW(std::invalid_argument("Unsupported element data type in matlab data"), "Data::loadMatlabHostData");
	    }
	    break;
	default:
	    BTTHROW(std::invalid_argument("Unsupported element data type in matlab data"), "Data::loadMatlabHostData");
    }

    // Note: commonFieldInitialization resets pDynDims, which is already set. Set elementDataType/elementSize here instead of calling it
    elementSize = NDArray::getElementSize(elementDataType);

    pNDArrays = std::unique_ptr <std::vector<std::unique_ptr<NDArray>>>(new std::vector<std::unique_ptr<NDArray>>);

    dimIndexType nDArrayNumElems = 1;
    for(dimIndexType i = 0; i < numOfSpatialDimensions; i ++) {
	nDArrayNumElems *= matvar->dims[i];
    }
    dimIndexType nDArrayOffset = 0;
    for(unsigned int i = 0; i < numNDArraysToRead; i++) {
	NDArray* pNDArray;
	pNDArray = NDArray::createNDArray(matvar, numOfSpatialDimensions, nDArrayOffset);
	pNDArrays->push_back(std::unique_ptr < NDArray >(pNDArray));
	// If number of spatial dimensions is 1, the offset for the next NDArray in elements is
	// the number of NDArrays (matlab saves arrays by columns)
	nDArrayOffset += nDArrayNumElems;
    }
    Mat_VarFree(matvar);
    matvar = NULL;
#ifdef DATA_DEBUG
    DATA_CERR("done, ");
    END_TIME(eTReadingKDataMatVar);
    TIME_DIFF_TYPE diffTReadingKDataMatVar;
    TIME_DIFF(diffTReadingKDataMatVar, bTReadingKDataMatVar, eTReadingKDataMatVar);
    DATA_CERR("elapsed time: " << diffTReadingKDataMatVar << " s" << std::endl);
#endif
}

/**
 * @brief Read matlab variable with information about spatial and temporal dimensions of a KData or XData
 *
 * @param[in] pDimsMatVar matlab variable with information about dimensions
 * @param[out] pCompleteSpatialDimsVector vector with spatial dimensions sizes
 * @param[out] nTemporalDims number of temporal dimensions
 * @throw std::invalid_argument if element datatype of dimensions array is not supported
 */
void Data::readDimsMatlabVar(const matvar_t* pDimsMatVar, std::vector<dimIndexType>* pCompleteSpatialDimsVector, numCoilsType& numCoils,
			     std::vector<dimIndexType>* pTemporalDimsVector) {
    if((pDimsMatVar->isComplex) || (pDimsMatVar->class_type != MAT_C_UINT32)) {
	BTTHROW(std::invalid_argument("Matlab dims variable should be an array of 32 bits unsigned integers"), "Data::readDimsMatlabVar");
    }
    char* data = (char*)pDimsMatVar->data;
    dimIndexType offsetInBytes = Mat_SizeOf(pDimsMatVar->data_type);
    dimIndexType nSpatialDims, nTemporalDims, spatialDimSize, temporalDimSize; //, numOfDimensions;
    dimIndexType firstSpatialDimIndex, lastSpatialDimIndex, firstTemporalDimIndex, lastTemporalDimIndex;
    // Minimum number of dimensions of matlab variable (matlab rank) is 2, but Dims variable is a 1-dimensional array (first dimension is 1, second dimension is Dims vector size)
    //numOfDimensions = pDimsMatVar->dims[1];
    // First numOfDimensions-2 positions (indexes from 0 to numOfDimensions-3) are sizes of spatial dimensions
    // The position with index numOfDimensions-2) is the number of coils
    // The last position (index numOfDimensions-1) is the number of temporal dimensions
    nSpatialDims = *((dimIndexType*)(data + offsetInBytes * (MatVarDimsData::NSD_POS)));
    numCoils = *((dimIndexType*)(data + offsetInBytes * (MatVarDimsData::NCOILS_POS)));
    nTemporalDims = *((dimIndexType*)(data + offsetInBytes * (MatVarDimsData::NTD_POS)));
    firstTemporalDimIndex = MatVarDimsData::NTD_POS + 1;
    lastTemporalDimIndex = firstTemporalDimIndex + nTemporalDims - 1;
    firstSpatialDimIndex = lastTemporalDimIndex + 1;
    lastSpatialDimIndex = firstSpatialDimIndex + nSpatialDims - 1;
    for(auto i = firstSpatialDimIndex; i <= lastSpatialDimIndex; i++) {
	spatialDimSize = *((dimIndexType*)(data + (offsetInBytes * i)));
	pCompleteSpatialDimsVector->push_back(spatialDimSize);
    }
    for(auto i = firstTemporalDimIndex; i <= lastTemporalDimIndex; i++) {
	temporalDimSize = *((dimIndexType*)(data + (offsetInBytes * i)));
	pTemporalDimsVector->push_back(temporalDimSize);
    }
}

/**
 * @brief Initialices Data class fields.
 * @param[in] elementDataType data type of base element
 */
void Data::commonFieldInitialization(ElementDataType elementDataType) {
	std::vector<dimIndexType>* pTemporalDims = new std::vector<dimIndexType>();
	setDynDims(pTemporalDims);
    pDataDimsAndStridesVector = std::unique_ptr<std::vector<index1DType>>(new std::vector<index1DType>(NUMINITIALPOSITIONSDIMSINFO, 0));
    this->elementDataType = elementDataType;
    elementSize = NDArray::getElementSize(elementDataType);
}

/**
 * @brief Checks if all NDArrays have the same spatial dimensions and set accordingly <i>allSizesEqual</i> class field.
 * @throw std::invalid_argument if data is empty
 */
void Data::checkNDArraysSizesAndSetAllSizesEqual() {
    std::vector<dimIndexType> NDArrayDims;
    if(pNDArrays == nullptr) {
	BTTHROW(std::invalid_argument("Empty data, cannot check if all NDArrays have same size, aborting"), "Data::checkNDArraysSizesAndSetAllSizesEqual");
    }
    if(pNDArrays->size() == 0) {
	BTTHROW(std::invalid_argument("Empty data, cannot check if all NDArrays have same size, aborting"), "Data::checkNDArraysSizesAndSetAllSizesEqual");
    }
    NDArrayDims = *(pNDArrays->at(0)->pDims);
    for(dimIndexType i = 1; i < pNDArrays->size(); i ++) {
	if(NDArrayDims != *(pNDArrays->at(i)->pDims)) {
	    allSizesEqual = 0;
	    return;
	}
    }
    allSizesEqual = 1;
}


/**
 * @brief Binds CLapp object to this Data object and adds itself to the list of Data objects managed by the CLapp object.
 *
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] copyDataToDevice copy host data to device
 */
void Data::setApp(std::shared_ptr<CLapp> pCLapp, bool copyDataToDevice) {
    try {
	this->pCLapp = pCLapp;
	if(pCLapp != nullptr) {
	    myDataHandle = pCLapp->addData(this, copyDataToDevice);
	    for (dimIndexType i=0; i<getNDArrays()->size(); i++) {
	    	pNDArrays->at(i)->setDataHandle(myDataHandle);
	    }
	}
    }
    catch(cl::Error& err) {
	BTTHROW(CLError(err), "Data::setApp");
    }
    catch(std::exception& err) {
	BTTHROW(err, "Data::setApp");
    }
}

// Getters
/**
 * @brief Gets a read-only pointer to images data.
 * @return pointer to a vector of images (objects of NDArray type)
 */
std::vector<const NDArray*>* Data::getNDArrays() const {
    // pNDArraysForGet stores a COPY of pHostData,
    // it must not be deallocated explicitly
    // (their pointed data is managed by pHostData),
    // but the vector can be emptied (vector data pointers are
    // deallocated, but not the data, which is still pointed
    // from pHostData)
    pNDArraysForGet->resize(0);
    if(pNDArrays == nullptr)
	return pNDArraysForGet;
    for(auto i = pNDArrays->begin(); i < pNDArrays->end(); ++i) {
	pNDArraysForGet->push_back((*i).get());
    }
    //return pResult.get();
    return pNDArraysForGet;
}

/**
 * @brief Gets a read-only pointer to stored data.
 * @return pointer to a vector of objects of NDArray type (every object contains an array of related elements, pixels of an image, for example)
 */
std::vector<const NDArray*>* Data::getData() const {
    return (std::vector<const NDArray*>*)(getNDArrays());
}

/**
 * @brief Gets a read-only pointer to one image data.
 * @param[in] NDArrayIndex NDarray index for the vector of NDArrays
 * @return const pointer to a NDArray object
 */
const NDArray* Data::getNDArray(dimIndexType NDArrayIndex) const {
    return pNDArrays->at(NDArrayIndex).get();
}

/**
 * @brief Gets a pointer to one NDArray object (image data) from the vector of NDArray objects.
 * @param[in] dynIndexes vector with values for the temporal indexes used for indexing the NDArray objects vector
 * @return pointer to NDArray object at the specified position
 */
const NDArray* Data::getDataAtDynPos(const std::vector<dimIndexType>& dynIndexes) const {
    index1DType index1D = get1DIndexFromDynPos(dynIndexes);
    return pNDArrays->at(index1D).get();
}

/**
 * @brief Gets a 1D index for the NDArray objects vector from the vector with temporal indexes.
 * @param[in] dynIndexes vector with values for the temporal indexes used for indexing the NDArray objects vector
 * @return 1D index with the position of the NDArray at the specified temporal indexes
 */
const index1DType Data::get1DIndexFromDynPos(const std::vector<dimIndexType>& dynIndexes) const {
    index1DType index1D = 0, mult = 1;
    for(index1DType dynVectorIndex = 0; dynVectorIndex < dynIndexes.size(); dynVectorIndex++) {
	mult = 1;
	if(dynVectorIndex > 0) {
	    for(index1DType dynDimsIndex = 0; dynDimsIndex <= dynVectorIndex - 1; dynDimsIndex++) {
		mult = mult * (pDynDims->at(dynDimsIndex));
	    }
	    index1D += dynIndexes.at(dynVectorIndex) * mult;
	}
    }
    printf("index1D: %d\n", index1D);
    return index1D;
}

/**
 * @brief Get number of temporal frames (product of values of each DynDims vector position).
 * @return number of temporal frames
 */
const index1DType Data::getDynDimsTotalSize() const {
    index1DType size = 1;
    for(index1DType i = 0; i < pDynDims->size(); i++) {
	size = size * pDynDims->at(i);
    }
    return size;
}

/**
 * @brief Gets pointer to data of a NDArray stored in host memory as a buffer
 * (pointer to object of class cl::Buffer). Delgates task to CLapp object.
 * @param[in] index NDarray index for the vector of NDArrays
 * @return raw pointer to object of class cl::Buffer
 */
const void* Data::getHostBuffer(dimIndexType index) const
{
    // If pCLapp is not valid or this Data has not been bound to pCLapp return the
    // pointer to the data array inside pHostData vector (as a generic pointer)
	// std::cerr << "Data::getHostBuffer, pCLapp: " << pCLapp << ", dataHandle: " << myDataHandle << std::endl;
    if((pCLapp == nullptr) || (myDataHandle == INVALIDDATAHANDLE)) {
	return getNDArray(index)->getHostDataAsVoidPointer();
    }
    return pCLapp->getHostBuffer(myDataHandle, index);
}

/**
 * @brief Gets pointer to array of dimensions and strides and data of all NDArrays of a Data object stored in contiguous device memory as a buffer
 * (pointer to object of class cl::Buffer). Delegates task to CLapp object.
 * @return raw pointer to object of class cl::Buffer
 */
cl::Buffer* Data::getCompleteDeviceBuffer() {
    return pCLapp->getCompleteDeviceBuffer(myDataHandle);
}

/**
 * @brief Gets pointer to data of all NDArrays of a Data object stored in device memory as a buffer
 * (pointer to object of class cl::Buffer). Delegates task to CLapp object.
 * @return raw pointer to object of class cl::Buffer
 */
cl::Buffer* Data::getDeviceBuffer() {
    return pCLapp->getDeviceBuffer(myDataHandle);
}

/**
 * @brief Gets HIP pointer to data of all NDArrays of a Data object stored in device memory as a buffer
 * (raw void* pointer). Delegates task to CLapp object.
 * @return raw void* HIP pointer to device data
 */
void* Data::getHIPDeviceBuffer() {
    return pCLapp->getHIPDeviceBuffer(myDataHandle);
}

/**
 * @brief Gets pointer to data of all NDArrays of a Data object stored in device memory as a buffer
 * (pointer to object of class cl::Buffer). Delegates task to CLapp object.
 * @param[in] index NDarray index for the vector of NDArrays
 * @return raw pointer to object of class cl::Buffer
 */
cl::Buffer* Data::getDeviceBuffer(dimIndexType index) {
    return pCLapp->getDeviceBuffer(myDataHandle, index);
}

/**
 * @brief Gets pointer to spatial and temporal data dimensions stored in host memory as an array.
 * @return generic pointer to array of spatial and temporal data dimensions
 */
const void* Data::getDataDimsAndStridesHostBuffer() const {
    return pCLapp->getDataDimsAndStridesHostBuffer(myDataHandle);
}

/**
 * @brief Gets pointer to spatial and temporal data dimensions stored in device memory as an array.
 * @return generic pointer to array of spatial and temporal data dimensions
 */
cl::Buffer* Data::getDataDimsAndStridesDeviceBuffer() {
    return pCLapp->getDataDimsAndStridesDeviceBuffer(myDataHandle);
}

/**
 * @brief Converts data of pHostBuffer field (pointer to raw float data) to a text representation.
 *
 * @param[in] title title for data text representation
 * @param[in] index index of the NDArray to get representation for
 * @return string with text representation of data in pHostBuffer
 */
const std::string Data::hostBufferToString(std::string title, dimIndexType index) {
    const void* pElementsArray;
    pElementsArray = getHostBuffer(index);
    return getNDArray(index)->nDArrayElementsToString(title, pElementsArray);
}

/**
 * @brief Stores vector of standard pointers to NDArrays objects as smart pointer to vector of smart pointers
 * @param[in,out] pNDArrays vector of standard pointers to NDArray objects
 * @param[in] copyData copy data from host to device after storing input data
 */
void Data::internalSetData(std::vector<NDArray*>*& pNDArrays, bool copyDataToDevice) {
    if(this->pNDArrays != nullptr) {
	this->pNDArrays->resize(0); // empty the pNDArrays attribute
    }
    else {
	this->pNDArrays =
	    std::unique_ptr <std::vector<std::unique_ptr<NDArray>>>
	    (new std::vector<std::unique_ptr<NDArray>>);
    }
    for(auto i = pNDArrays->begin(); i < pNDArrays->end(); ++i) {
	for(auto j = pNDArrays->begin(); j < i; j++) {
	    assert(*i != *j);
	}
	this->pNDArrays->push_back(std::unique_ptr < NDArray > (*i));
    }
    delete(pNDArrays);
    pNDArrays = nullptr;
    checkNDArraysSizesAndSetAllSizesEqual();

    // copy input data to device if so requested
    if(copyDataToDevice)
        setApp(pCLapp, true);
}

/**
 * @brief Checks if index is valid for NDArray spatial dimensions.
 * @param[in] index index to be checked
 * @param[in] nDArraySize vector with spaial dimensions of NDArray
 */
bool Data::checkNDArrayIndex(dimIndexType index, dimIndexType nDArraySize) {
    if((index < 0) || (index > nDArraySize)) {
	std::cerr << "NDArray index out of range: " << index << "(NDArray size: " << std::to_string(nDArraySize) << ")\n";
	return false;
    }
    return true;
}

/**
 * @brief Store image data (one NDArray pointer) in the class field pData.
 * @param[in,out] pNDArray image data as a pointer to a single NDArray.
 * @param[in] copyData copy data from host to device after storing input data
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this object, and parameter value is set to nullptr after method completion.
 */
void Data::setData(NDArray*& pNDArray, bool copyData) {
    // Create a vector of NDArrays containing just the input NDArray
    auto pNDArrays = new std::vector<NDArray*>(1);
    pNDArrays->at(0) = pNDArray;
    pNDArray = nullptr;

    internalSetData(pNDArrays, copyData);
}

/**
 * @brief Store image data (vector to NDArray pointers) in the class field pData.
 * @param[in,out] pNDArrays image data as a vector of NDArrays.
 * @param[in] copyData copy data from host to device after storing input data
 * All parameters of type *& (reference to pointer) have move semantics: ownership of memory is moved from parameter to this object, and parameter value is set to nullptr after method completion.
 */
void Data::setData(std::vector<NDArray*>*& pNDArrays, bool copyData) {
    internalSetData(pNDArrays, copyData);
}

/**
 * @brief Sets vector of temporal dimensions. All parameters of type *& (reference to pointer) have move semantics: ownership of
 * memory is moved from parameter to this object, and parameter value is set to nullptr after method completion.
 * @param[in,out] pDynDims pointer to vector of temporal dimensions
 */
void Data::setDynDims(std::vector<dimIndexType>*& pDynDims) {
    this->pDynDims.reset(pDynDims);
    pDynDims = nullptr;

    // Understand nullptr temporal dimensions as a single, one-length temporal dimension
    if(this->pDynDims == nullptr) {
	this->pDynDims = std::unique_ptr<std::vector<dimIndexType>>(new std::vector<dimIndexType>({1}));
	DATA_CERR("Data::setDynDims: nullptr pDynDims given; set to single temporal dimension of length 1\n");
    }

    // Understand existent but empty temporal dimension vector as a single, one-length temporal dimension
    if(this->pDynDims->size() == 0) {
	this->pDynDims = std::unique_ptr<std::vector<dimIndexType>>(new std::vector<dimIndexType>({1}));
	DATA_CERR("Data::setDynDims: empty pDynDims given; set to single temporal dimension of length 1\n");
    }

    // Check for zero-length dimensions and set them to 1 if found
    auto& dynDims = *this->pDynDims;
    for(auto&& i : dynDims)
	if(i == 0) {
	    i = 1;
	    DATA_CERR("Data::setDynDims: zero-length temporal dimension found; set to length=1\n");
	}

}

/**
 * @brief Build a file name prefix of the form \<prefix\>\<width\>x\<height\> (for a 2D dimensions vector) or
 * \<prefix\>\<width\>x\<height\>x\<depth\> (for a 3D dimensions vector).
 * @param[in] prefix string with the prefix part of the name
 * @param[in] pDims vector of dimensions for extracting widht, height and depth (if present) values
 * @return string with the format \<prefix\>\<rows\>x\<cols\> (for a 2D pDims) or
 * \<prefix\>\<width\>x\<height\>x\<depth\> (for a 3D pDims)
 */
std::string Data::buildFileNamePrefix(const std::string &prefix, const std::vector<dimIndexType>* pDims) {
    std::ostringstream fileNamePrefix;
    fileNamePrefix << prefix;
    if(pDims != nullptr) {
	dimIndexType dimsSize = pDims->size();
	for(dimIndexType i = 0; i < dimsSize; i++) {
	    fileNamePrefix << pDims->at(i);
	    if(i != (dimsSize - 1)) {
		fileNamePrefix << "x";
	    }
	}
    }
    return fileNamePrefix.str();
}

/**
 * @brief Build a file name sufix part of the form \<suffix\>.\<fileExtension\>.
 * @param[in] suffix string with the suffix part of the name (default value is empty string)
 * @param[in] fileExtension string with de extension part of the name (default value is "raw")
 * @return string with the format \<suffix\>.\<fileExtension\>
 */
std::string Data::buildFileNameSuffix(const std::string suffix, const std::string fileExtension) {
    return suffix + "." + fileExtension;
}

void loadRawDataForThread(Data* pData, const std::string &fileName, const std::vector< dimIndexType >* pArraySpatialDims,
			   dimIndexType numOfNDArrays) {
    std::vector<NDArray*>* pNDArrays;
    pNDArrays = new std::vector<NDArray*>;
    std::vector<dimIndexType>* pAuxSpatialDims;
    NDArray* pAuxNDArray;
    DATA_CERR("Loading CFL file " << fileName << "... ");
    std::fstream f;
    LPISupport::Utils::openFile(fileName, f, std::ios::in|std::ios::binary, "Data::loadRawData");

    for(dimIndexType i = 0; i < numOfNDArrays ; i++) {
        // pAuxSpatialDims is used as parameter with move semantics for createNDArray method
        pAuxSpatialDims = new std::vector<dimIndexType>(*(pArraySpatialDims)); // pDims parameter contents are copied to a new vector
        pAuxNDArray = NDArray::createNDArray(f, pAuxSpatialDims, pData->getElementDataType());
        pNDArrays->push_back(pAuxNDArray); // copy NDArray pointer and add it to vector
    }
    DATA_CERR("done" << std::endl);
    pData->setData(pNDArrays);
    f.close();
}

/**
 * @brief Load data from a file to hostData (as a group of NDarray objects). Data stored as raw format in order rows, columns, coils and temporal dims
 * (number of rows and columns is the same for every image)
 * @param[in] fileName file name
 * @param[in] pArraysSpatialDims pointer to a vector with pointers to vectors with the dimensions of every image
 * @param[in] numOfNDArrays number of NDArrays to be read
 */

void Data::loadRawData(const std::string &fileName, const std::vector< dimIndexType >* pArraySpatialDims,
			   dimIndexType numOfNDArrays, bool asyncLoad) {
	//std::shared_ptr<Data> xDataForLoading = shared_from_this();
	if (asyncLoad) {
		this->pFileLoaderThread = std::unique_ptr<std::thread>(new std::thread(loadRawDataForThread, this, fileName, pArraySpatialDims, numOfNDArrays));
	}
	else
		loadRawDataForThread(this, fileName, pArraySpatialDims, numOfNDArrays);
}

void Data::waitLoadEnd() {
	if (pFileLoaderThread != nullptr) {
		pFileLoaderThread->join();
		pFileLoaderThread = nullptr;
	}
}

/**
 * @brief Load data of a group of files to hostData (every file contains one image and it is stored into a NDArray object).
 *
 * Files of this group have names with the format \<fileNamePrefix\>\<fileNameSuffix(i)\>\<fileNameExtension\>,
 * where \<i\> is the index for indentifying the suffix of every image.
 * @param[in] fileNamePrefix fixed part of the file name before the variable part
 * @param[in,out] pArraysSpatialDims pointer to a vector with pointers to vectors with the dimensions of every image
 * @param[in,out] pTemporalDims pointer to vector of temporal dimensions
 * @param[in] fileNameSuffixes vector with the variable part of the name for every image
 * @param[in] fileNameExtension extension of the file name
 */
void Data::loadRawData(const std::string &fileNamePrefix, std::vector<std::vector< dimIndexType >*>*& pArraysSpatialDims,
			   std::vector <dimIndexType>*& pTemporalDims,
			   std::vector<std::string> &fileNameSuffixes, const std::string &fileNameExtension) {
    setDynDims(pTemporalDims);
    std::vector<NDArray*>* pNDArrays;
    pNDArrays = new std::vector<NDArray*>;
    // if file name suffixes vector is empty we must add an empty string value (without suffixes no file is read)
    if(fileNameSuffixes.size() == 0) {
	fileNameSuffixes.push_back("");
    }
    DATA_CERR("Reading files with prefix: " << fileNamePrefix << " (" << fileNameSuffixes.size() << " file(s))... ");
    for(dimIndexType i = 0; i < fileNameSuffixes.size() ; i++) {
	std::ostringstream fileNameStream;
	std::vector<dimIndexType>* pAuxDims;
	pAuxDims = new std::vector<dimIndexType>(*(pArraysSpatialDims->at(i))); // parameter pDims contents are copied to new vector
	fileNameStream << fileNamePrefix;
	if(fileNameSuffixes.at(i) != "") {
	    fileNameStream << fileNameSuffixes.at(i);
	}
	fileNameStream << fileNameExtension;
	std::string completeFileName = fileNameStream.str();
	NDArray* pTempNDArray;
	pTempNDArray = NDArray::createNDArray(completeFileName, pAuxDims, elementDataType);
	pNDArrays->push_back(pTempNDArray); // copy NDArray pointer and add it to vector
    }
    DATA_CERR("done" << std::endl);
    this->setData(pNDArrays);
    delete(pArraysSpatialDims);
    pArraysSpatialDims = nullptr;
}

/**
 * @brief Save hostBuffer of every NDArray contained object to a group of files.
 *
 * File names have format \<fileNamePrefix\>\<fileNameSuffix(i)\>\<fileNameExtension\>,
 * where \<i\> is the index for indentifying the suffix of every image.
 * @param[in] fileNamePrefix fixed part of the file name before the variable part
 * @param[in] fileNameSuffixes vector with the variable part of the name for every image
 * @param[in] fileNameExtension extension of the file name
 */
void Data::saveRawData(const std::string &fileName) {
    //for (auto i = pNDArrays->begin(); i != pNDArrays->end(); i++) {
    std::fstream f;
    LPISupport::Utils::openFile(fileName, f, std::ofstream::out|std::ofstream::trunc|std::ofstream::binary, "Data::saveRawData");
    const void* data;
    for(size_t i = 0; i < pNDArrays->size(); i++) {
	data = getHostBuffer(i);
	f.write(reinterpret_cast<const char*>(data), getNDArray(i)->size() * elementSize);
    }
    f.close();
}

/**
 * @brief Save hostBuffer of every NDArray contained object to a group of files.
 *
 * File names have format \<fileNamePrefix\>\<fileNameSuffix(i)\>\<fileNameExtension\>,
 * where \<i\> is the index for indentifying the suffix of every image.
 * @param[in] fileNamePrefix fixed part of the file name before the variable part
 * @param[in] fileNameSuffixes vector with the variable part of the name for every image
 * @param[in] fileNameExtension extension of the file name
 */
void Data::saveRawData(const std::string &fileNamePrefix, std::vector<std::string> &fileNameSuffixes,
			   const std::string &fileNameExtension) {
    dimIndexType index(0);
    if(pNDArrays->size() > fileNameSuffixes.size()) {
	std::ostringstream errorInfo;
	errorInfo << "number fileNameSuffixes (" << fileNameSuffixes.size() << ") must be equal to number of NDArrays ("
		  << pNDArrays->size() << ")\n";
	BTTHROW(std::invalid_argument(errorInfo.str()), "Data::saveRawHostData");
    }
    //for (auto i = pNDArrays->begin(); i != pNDArrays->end(); i++) {
    for(size_t i = 0; i < pNDArrays->size(); i++) {
	std::ostringstream fileNameStream;
	fileNameStream << fileNamePrefix << fileNameSuffixes.at(index);
	fileNameStream << fileNameExtension;

	std::ofstream f(fileNameStream.str());
	const void* data;
	data = getHostBuffer(i);
	f.write(reinterpret_cast<const char*>(data), getNDArray(i)->size() * elementSize);
	f.close();
	fileNameStream.clear();
	index++;
    }
}

/**
 * @brief Get a fragment from stored images.
 * @param[in] specific fragment specification
 * @return group of images (pointer to read-only vectors to NDArrays)
 */
const std::vector<const NDArray*>* Data::getFragment(FragmentSpecif specif) {
    std::vector<const NDArray*>* pResult = new std::vector<const NDArray*>;
    for(auto i = pNDArrays->begin(); i < pNDArrays->end(); ++i) {
	pResult->push_back((*i).get());
    }
    return pResult;
}

/**
 * @brief Calculates image/volume size in number of real elements rounded up to the nearest multiple of device memory base
 * address alignment, and stores it in dataStridesVector field after storing spatial dimensions strides.
 * @param[in] NDArray1DIndex index of the NDArray to be used for getting spatial dimensions strides
 * @param deviceMemBaseAddrAlignInBytes device memory alignment in bytes
 */
void Data::calcDataAlignedSize(dimIndexType NDArray1DIndex, dimIndexType deviceMemBaseAddrAlignInBytes) {
    dimIndexType numSpatialDims = pNDArrays->at(NDArray1DIndex)->getNDims();
    dimIndexType lastSpatialDimSize = pNDArrays->at(NDArray1DIndex)->getDims()->at(numSpatialDims - 1);
    std::vector<dimIndexType>* pSpatialDimsStridesVector =
	pNDArrays->at(NDArray1DIndex)->calcUnaligned1DArrayStridesFromNDArrayDims();
    // NDArray data size unaligned measured in kernel known elements (float -not complex-,  uint, etc.)
    dimIndexType NDArrayDataUnalignedSizeInElements = pSpatialDimsStridesVector->at(numSpatialDims - 1) * lastSpatialDimSize;
    // Memory alignment measured in kernel known elements (memory aligment in bytes / size of kernel known element in bytes).
    // Element size is 8 bytes for complex data NDArray, but 4 for floats, known by kernels.
    // This way, effective elementSize is stored elemens size / first spatial stride value (2 for complex data, 1 for float, uint, etc).
    dimIndexType memoryAlignmentInElements = deviceMemBaseAddrAlignInBytes / (elementSize / pSpatialDimsStridesVector->at(ElementStridePos));
    dimIndexType NDArrayDataAlignedSize = CLapp::roundUp(NDArrayDataUnalignedSizeInElements, memoryAlignmentInElements);
    pDataDimsAndStridesVector->insert(pDataDimsAndStridesVector->end(), pSpatialDimsStridesVector->begin(),
				      pSpatialDimsStridesVector->end());
    delete(pSpatialDimsStridesVector);
    pDataDimsAndStridesVector->push_back(NDArrayDataAlignedSize);
}

/**
 * @brief Stores image/volume spatial and temporal dimensions in dataDims field (dataDims data type is valid to be used for
 * kernel parameter).
 * @throw std::invalid_argument if data is empty
 */
void Data::calcDataDims() {
    std::string errorMessage;
    if(pDataDimsAndStridesVector->size() > NUMINITIALPOSITIONSDIMSINFO) {  // DataDimsAndStridesVector already set, must be reset
	// pDataDimsAndStridesVector is initialized with NUMINITIALPOSITIONSDIMSINFO positions (the unique known and constant positions)
	pDataDimsAndStridesVector->resize(NUMINITIALPOSITIONSDIMSINFO);
    }
    errorMessage.append(errorPrefix);
    errorMessage.append("data empty, cannot calculate spatial dimensions");

    if(pNDArrays == nullptr) {
	BTTHROW(std::invalid_argument(errorMessage.c_str()), "Data::calcDataDims");
    }
    if(pNDArrays->size() == 0) {
	BTTHROW(std::invalid_argument(errorMessage.c_str()), "Data::calcDataDims");
    }
    // number of spatial dims for the first NDArray (the same for all NDArrays if allSizesEqual class field is 1)
    uint numSpatialDims = pNDArrays->at(0)->getNDims();
    pDataDimsAndStridesVector->at(NumSpatialDimsPos) = numSpatialDims;
    pDataDimsAndStridesVector->at(AllSizesEqualPos) = allSizesEqual;
    uint numTemporalDims = getNDynDims();
    pDataDimsAndStridesVector->at(NumTemporalDimsPos) = numTemporalDims;
    for(uint temporalDimId = 0; temporalDimId < numTemporalDims; temporalDimId++) {
	pDataDimsAndStridesVector->push_back(getDynDims()->at(temporalDimId));
    }

    //uint FirstSpatialDimsPos = FirstTemporalDimPos + numTemporalDims;
    for(uint spatialDimId = 0; spatialDimId < numSpatialDims; spatialDimId++) {
	pDataDimsAndStridesVector->push_back(pNDArrays->at(0)->getDims()->at(spatialDimId));
    }

    if(!allSizesEqual) { // store information of spatial dimensions for last (pNDArrays->size())-1 NDArrays
	for(uint NDArrayIndex = 1; NDArrayIndex < pNDArrays->size(); NDArrayIndex++) {
	    numSpatialDims = pNDArrays->at(NDArrayIndex)->getNDims();
	    for(uint spatialDimId = 0; spatialDimId < numSpatialDims; spatialDimId++) {
		pDataDimsAndStridesVector->push_back(pNDArrays->at(NDArrayIndex)->getDims()->at(spatialDimId));
	    }
	}
    }
}

/**
 * @brief Calculates strides associated to temporal dimensions and first following stride, for accessing data stored as 1D arrays of real
 * elements, and stores them in dataStridesVector field.
 *
 * The first following stride will be the coil stride or the first frame dimension stride
 * depending on the subclass of Data (coil stride for KData and SensitivityMapsData, first frame dimension stride for XData
 * and SamplingMaskData).
 * @param deviceMemBaseAddrAlignInBytes device memory alignment in bytes
 */
void Data::calcDataStrides(dimIndexType deviceMemBaseAddrAlignInBytes) {
    dimIndexType NumNDArrays;
    if(allSizesEqual) {
	NumNDArrays = 1;
    }
    else {
	NumNDArrays = pNDArrays->size();
    }
    for(dimIndexType NDArray1DIndex = 0; NDArray1DIndex < NumNDArrays; NDArray1DIndex++) {
	// Calculate all spatial strides and the following one (rounded up to multiple of deviceMemBaseAddrAlignInBytes/8
	// bytes per complex element (2 floats)
	calcDataAlignedSize(NDArray1DIndex, deviceMemBaseAddrAlignInBytes);
	//dimIndexType lastStride = pDataStridesVector->at(pDataStridesVector->size()-1);
	dimIndexType lastStride = pDataDimsAndStridesVector->at(pDataDimsAndStridesVector->size() - 1);
	dimIndexType numCoils = pDataDimsAndStridesVector->at(NumCoilsPos);
	if(numCoils == 0) { // XData
	    // Last stride is first temporal dimension stride for XData
	    for(dimIndexType temporalDimId = 1; temporalDimId < getNDynDims(); temporalDimId++) {
		lastStride *= pDynDims->at(temporalDimId - 1); // new stride is equal to previous stride * previous dimension
		pDataDimsAndStridesVector->push_back(lastStride);
	    }
	}
	else {
	    // Last stride is coil stride for KData or SensitivityMapsData
	    // stride for first temporal dimension is equal to coils stride * number_of_coils
	    lastStride *= numCoils;
	    for(dimIndexType temporalDimId = 0; temporalDimId < getNDynDims(); temporalDimId++) {
		pDataDimsAndStridesVector->push_back(lastStride);
		lastStride *= pDynDims->at(temporalDimId);  // new stride is equal to previous stride * previous dimension
	    }
	}
    }
}

/**
 * Stores image/volume spatial and temporal dimensions in DataDimsAndStrides field
 * @brief Calculation of data spatial and temporal dimensions and associated strides for accessing data stored as 1D arrays of real
 * elements (see @ref calcDataDims and @ref calcDataStrides).
 * @param deviceMemBaseAddrAlignInBytes device memory alignment in bytes
 */
void Data::calcDimsAndStridesVector(dimIndexType deviceMemBaseAddrAlignInBytes) {
    calcDataDims();
    calcDataStrides(deviceMemBaseAddrAlignInBytes);
    DATA_CERR("dimsAndStridesVector: ");
    for(size_t i = 0; i < pDataDimsAndStridesVector->size() - 1; i++) {
	DATA_CERR(pDataDimsAndStridesVector->at(i) << ", ");
    }
    DATA_CERR(pDataDimsAndStridesVector->at(pDataDimsAndStridesVector->size() - 1) << std::endl);
}

/**
 * @brief Checks if every spatial dimension is multiple of VECTORDATATYPESIZE/2.
 * @param[in] pDims pointer to vector with spatial dimensions
 * @throw invalid_argument if checks failed for at least one dimension
 */
void Data::checkValiditySpatialDimensions(const std::vector<dimIndexType>* pDims) {
    std::ostringstream errorMsgStream;
    if(pDims == nullptr) {
	BTTHROW(std::invalid_argument("NDArray dimensions not set yet"), "Data::checkValiditySpatialDimensions");
    }
    dimIndexType dimsSize = 1; //pDims->size();
    for(dimIndexType i = 0; i < dimsSize; i++) {
	if((pDims->at(i)) % (VECTORDATATYPESIZE / 2) != 0) {
	    errorMsgStream << "NDArray dimension " << i << " is not multiple of " << (VECTORDATATYPESIZE / 2) << " (is " << pDims->at(i) << ")";
	    BTTHROW(std::invalid_argument(errorMsgStream.str()), "Data::checkValiditySpatialDimensions");
	}
    }
}

/**
 * @brief Checks if every spatial dimension of every NDArray is multiple of VECTORDATATYPESIZE/2.
 * @param[in] pArraysDims pointer to vector of pointerts to vectors with spatial dimensions
 * @throw invalid_argument if checks failed for at least one dimension of one NDArray
 */
void Data::checkValiditySpatialDimensions(const std::vector<std::vector<dimIndexType>*>* pArraysDims) {
    for(dimIndexType i = 0; i < pArraysDims->size(); i++) {
	checkValiditySpatialDimensions(pArraysDims->at(i));
    }
}

/**
 * @brief Read one or more matlab variables from a file with matlab format
 * @param[in] fileName name of the file in matlab format
 * @param[in] variableNames vector with name of variables to be read
 * @return map of matlab variables (keys are variable names, values are matlab variables)
 * @throw std::invalid_argument if matlab file cannot be opened or variable cannot be found in matlab file
 */
std::map<std::string, matvar_t*>* Data::readMatlabVariablesFromFile(std::string fileName, std::vector<std::string> variableNames) {
#ifdef DATA_DEBUG
    BEGIN_TIME(bTReadingMatlabFile);
    DATA_CERR("Reading matlab file...\n");
#endif
    mat_t* matfp;
    matvar_t* matvar;
    std::map<std::string, matvar_t*>* pMatlabVariablesMap = new std::map<std::string, matvar_t*>;
    matfp = Mat_Open(fileName.c_str(), MAT_ACC_RDONLY);
    if(NULL == matfp) {
	BTTHROW(std::invalid_argument("Error opening MAT file '" + fileName + "'!"), "Data::readMatlabVariablesFromFile");
	return pMatlabVariablesMap;
    }
    for(auto variableName : variableNames) {
	matvar = Mat_VarRead(matfp, variableName.c_str());
	if(NULL == matvar) {
	    BTTHROW(std::invalid_argument("Variable '" + variableName + "' not found, or error reading MAT file"), "Data::readMatlabVariablesFromFile");
	}
	else {
	    (*pMatlabVariablesMap)[variableName] = matvar;
#ifdef DATA_DEBUG
	    Mat_VarPrint(matvar, 1);
#endif
	}
    }
    Mat_Close(matfp);
#ifdef DATA_DEBUG
    DATA_CERR("Done\n");
    END_TIME(eTReadingMatlabFile);
    TIME_DIFF_TYPE diffTReadingMatlabFile;
    TIME_DIFF(diffTReadingMatlabFile, bTReadingMatlabFile, eTReadingMatlabFile);
    DATA_CERR("Elapsed time: " << diffTReadingMatlabFile << " s" << std::endl);
#endif
    return pMatlabVariablesMap;
}

/**
 * @brief Read all matlab variables stored in a file with matlab format
 * @param[in] fileName name of the file in matlab format
 * @return map of matlab variables (keys are variable names, values are matlab variables)
 * @throw std::invalid_argument if matlab file cannot be opened
 */
std::map<std::string, matvar_t*>* Data::readMatlabVariablesFromFile(std::string fileName) {
    mat_t* matfp;
    matvar_t* matvar;
    std::map<std::string, matvar_t*>* pMatlabVariablesMap = new std::map<std::string, matvar_t*>;
    matfp = Mat_Open(fileName.c_str(), MAT_ACC_RDONLY);
    if(NULL == matfp) {
	BTTHROW(std::invalid_argument("Error opening MAT file '" + fileName + "'!"), "Data::readMatlabVariablesFromFile");
	//throw(std::invalid_argument("Error opening MAT file '" + fileName + "'!"));
	return pMatlabVariablesMap;
    }

    while((matvar = Mat_VarReadNext(matfp)) != NULL) {
	// sanity check: since rank is a signed int in matio and we use unsigned vars to store dimensions, we could be fooled by a corrupted matlab file
	if(matvar->rank < 2)
	    BTTHROW(std::invalid_argument("Data::readMatlabVariablesFromFile: invalid rank in matlab variable"), "Data::readMatlabVariablesFromFile");

	(*pMatlabVariablesMap)[matvar->name] = matvar;
#ifdef DATA_DEBUG
	Mat_VarPrint(matvar, 1);
#endif
    }
    Mat_Close(matfp);
    return pMatlabVariablesMap;
}

/**
 * @brief Fills a previously created matlab variable with data from this object data
 *
 * @param[in] matfp pointer to open matlab format file
 * @param[in] matVarName of the matlab variable to be filled with data
 * @param[in] pMatVarInfo pointer to a MatVarInfo object (storing matlab info about data: class_type, data_type, data values and options)
 */
void Data::fillMatlabVarInfo(mat_t* matfp, std::string matVarName, MatVarInfo* pMatVarInfo) {
    matvar_t* matvar;
    dimIndexType numNDArrayElements = getNDArray(0)->size();
    dimIndexType matlabNDArrayOffset = 0;
    void* pNDArrayData;
    for(dimIndexType i = 0; i < getNDArrays()->size(); i++) {
	pNDArrayData = getHostBuffer(i);
	pMatVarInfo->set((const dimIndexType*) pDataDimsAndStridesVector.get()->data(),
					    matlabNDArrayOffset, (const NDArray*) pNDArrays->at(i).get(), pNDArrayData);
	matlabNDArrayOffset += numNDArrayElements;
    }
    matvar = Mat_VarCreate(matVarName.c_str(), pMatVarInfo->getClassType(), pMatVarInfo->getDataType(), pMatVarInfo->getRank(), pMatVarInfo->getDims(), pMatVarInfo->getData(),
			   pMatVarInfo->getOpt());
    if(NULL == matvar) {
	BTTHROW(std::invalid_argument(std::string("Error creating variable ") + matVarName), "Data::fillMatlabVarInfo");
    }
    else {
	Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_ZLIB);
	Mat_VarFree(matvar);
    }
}

/**
 * @brief Copy data stored in device memory to host memory (from buffers). Delegates task to CLapp object.
 * @param[in] queueFinish true if queue.finish() method must be called to guarantee that kernel execution has finished before
 * copying data back to host memory
 */
void Data::device2Host(bool queueFinish) {
    pCLapp->device2Host(getHandle(), queueFinish);
}


//---------------------------------
// host/kernel functions
//---------------------------------

// Functions related to spatial dimensions
uint Data::getNumSpatialDims() {
    return ::getNumSpatialDims(getHostBuffer());
}

uint Data::getNumNDArrays() {
    return ::getNumNDArrays(getHostBuffer());
}

uint Data::getNDArray1DIndex(uint coilIndex, uint temporalDimIndexes[]) {
    return ::getNDArray1DIndex(getHostBuffer(), coilIndex, temporalDimIndexes);
}

uint Data::getNDArrayTotalSize(uint NDArray1DIndex) {
    return ::getNDArrayTotalSize(getHostBuffer(), NDArray1DIndex);
}

uint Data::getNDArrayStride() {
    return ::getNDArrayStride(getHostBuffer());
}

uint Data::getSpatialDimStride(uint spatialDimIndex, uint NDArray1DIndex) {
    return ::getSpatialDimStride(getHostBuffer(), spatialDimIndex, NDArray1DIndex);
}

uint Data::getSpatialDimSize(uint spatialDimIndex, uint NDArray1DIndex) {
    return ::getSpatialDimSize(getHostBuffer(), spatialDimIndex, NDArray1DIndex);
}

// Functions related to coils
uint Data::getNumCoils() {
    return ::getNumCoils(getHostBuffer());
}

uint Data::getCoilDim() {
    return ::getCoilDim(getHostBuffer());
}

uint Data::getCoilStride(uint NDArray1DIndex) {
    return ::getCoilStride(getHostBuffer(), NDArray1DIndex);
}

// Functions related to temporal dimensions
uint Data::getNumTemporalDims() {
    return ::getNumTemporalDims(getHostBuffer());
}

uint Data::getTemporalDim(uint tempDim) {
    return ::getTemporalDim(getHostBuffer(), tempDim);
}

uint Data::getTemporalDimStride(uint temporalDimIndex, uint NDArray1DIndex) {
    return ::getTemporalDimStride(getHostBuffer(), temporalDimIndex, NDArray1DIndex);
}

uint Data::getTemporalDimSize(uint temporalDimIndex) {
    return ::getTemporalDimSize(getHostBuffer(), temporalDimIndex);
}

// Functions related to dimensions in general (not particular to spatial, coil or temporal dimensions)
uint Data::getDimSize(uint dimIndex, uint NDArray1DIndex) {
    return ::getDimSize(getHostBuffer(), dimIndex, NDArray1DIndex);
}

uint Data::getElementStride(uint NDArray1DIndex) {
    return ::getElementStride(getHostBuffer(), NDArray1DIndex);
}

uint Data::getDimStride(uint dimIndex, uint NDArray1DIndex) {
    return ::getDimStride(getHostBuffer(), dimIndex, NDArray1DIndex);
}

const void* Data::getNextElement(dimIndexType nDim, dimIndexType curNDArray, dimIndexType curOffset) {
    return ::getNextElement(getHostBuffer(), nDim, curNDArray, curOffset);
}


//---------------------------------
// Show-related stuff
//---------------------------------

void Data::show(ShowParms* inParms) {
    // if the caller does not specify parameters, set default ones
    std::unique_ptr<ShowParms> parms;
    if(inParms)
	parms = std::unique_ptr<ShowParms>(new ShowParms(*inParms));
    else
	parms = std::unique_ptr<ShowParms>(new ShowParms());

    // Some sanity checks
    if(allSizesEqual != 1)
	BTTHROW(std::invalid_argument("show: Every NDArray in the Data object must be the same size"), "Data::show");

    auto numNDArrays = getNumNDArrays();
    if(numNDArrays == 0)
	BTTHROW(std::invalid_argument("show: the Data object to show does not contain any NDArrays"), "Data::show");

    // Transform data if necessary (complex to real, normalize)
    // and select OpenCV data type accordingly
    auto pIn = shared_from_this();
    std::shared_ptr<Data> pShow = pIn;
    int opencvDataType;
    if(elementDataType == TYPEID_COMPLEX ||
	    elementDataType == TYPEID_REAL) {

	// If data is complex, convert it to real first
	if(elementDataType == TYPEID_COMPLEX) {
	    DATA_CERR("Data::show(): converting complex data to real\n");
	    pShow = pIn->clone(TYPEID_REAL);
	    auto complex2real = Process::create<Complex2Real>(getApp(), pIn, pShow);
	    auto initParms = std::make_shared<Complex2Real::InitParameters>(parms->complexPart);
	    complex2real->setInitParameters(initParms);
	    complex2real->init();
	    complex2real->launch();

	    // Set input for the (possible) next transformation
	    pIn = pShow;
        }

	if(parms->normalize) {
	    // If we are told to normalize but not a normalizing factor, guess it from the data
	    if(parms->normFactor == 0) {
		DATA_CERR("Data::show(): normalizing data (norm. factor guessed from data)\n");

		// Sum up all elements in this data object for the normalizeShow process
		auto pSum = std::make_shared<XData>(pCLapp, 1 * sizeof(realType), TYPEID_REAL);
		auto sumReduce = Process::create<SumReduce>(getApp(), pIn, pSum);
		sumReduce->init();
		sumReduce->launch();

		// "Normalize": scale data so that the sample mean sits at 0.5
		if(pShow.get() == this)
		    pShow = pIn->clone(false);	// At this point, pIn must contain real data
		auto normalizeShow = Process::create<NormalizeShow>(getApp(), pIn, pShow);
		auto launchParms = std::make_shared<NormalizeShow::LaunchParameters>(pSum);
		normalizeShow->setLaunchParameters(launchParms);
		normalizeShow->init();
		normalizeShow->launch();
	    }
	    // if we are given a normalizing factor, just multiply the data with it
	    else {
		DATA_CERR("Data::show(): normalizing data (norm. factor given by user)\n");

		if(pShow.get() == this)
		    pShow = pIn->clone(false);	// At this point, pIn must contain real data;
		auto scalarMultiply = Process::create<ScalarMultiply>(getApp(), pIn, pShow);
		auto launchParms = std::make_shared<ScalarMultiply::LaunchParameters>(parms->normFactor);
		scalarMultiply->setLaunchParameters(launchParms);
		scalarMultiply->init();
		scalarMultiply->launch();
	    }

	    // Set input for the (possible) next transformation
	    pIn = pShow;
	}

	// Set OpenCV data type to 32bit-float-1channel
	opencvDataType = CV_32FC1;
    }
    else if(elementDataType == TYPEID_INDEX) {
	if(parms->normalize)
	    CERR("Data::show(): NDArrays of integral type can't be normalized. Showing unnormalized.");

	// Set OpenCV data type to 32bit-signed-1channel
	opencvDataType = CV_32SC1;
    }
    else if(elementDataType == TYPEID_CL_UCHAR) {
	if(parms->normalize)
	    CERR("Data::show(): NDArrays of integral type can't be normalized. Showing unnormalized.");

	// Set OpenCV data type to 32bit-signed-1channel
	opencvDataType = CV_8UC1;
    }
    else
	BTTHROW(std::invalid_argument("show: unsupported Data type"), "Data::show");

    // If there are more than one slice/coil to show, paint them in a convenient mosaic canvas
    uint nTiles;
    std::string tileText;
    nTiles=pShow->getSpatialDimSize(2, 0);
    if(nTiles >= 2)
	tileText="Slice";
    else {
	nTiles=pIn->getNumCoils();
	if(nTiles >= 2)
	    tileText="Coil";
    }

    // storage for tile coordinates
    std::vector<std::array<uint,2>> tileCoords;
    if(nTiles >= 2) {
        DATA_CERR("Data::show(): generating mosaic canvas\n");
	auto reshapeShow = Process::create<ReshapeShow>(getApp());
	reshapeShow->setInput(pIn);
        auto initParms = std::make_shared<ReshapeShow::InitParameters>(parms->timeDimension);
        reshapeShow->setInitParameters(initParms);
	reshapeShow->init();
	reshapeShow->launch();

	// ReshapeShow creates its own output Data object
	pShow = reshapeShow->getCanvas();

	// Get tile coordinates
	tileCoords = reshapeShow->getTileCoords();

	// Set input for the (possible) next transformation
	pIn = pShow;
    }
    else
	tileCoords.push_back({0,0});

    // Get data from device
    pCLapp->getCommandQueue().finish();
    pShow->device2Host();

    // Get first, last, and number of frames to show
    auto tempDims = getDynDims();
    unsigned int availableFrames = tempDims->at(parms->timeDimension);

    // Check frame bounds, calculate nFrames accordingly
    unsigned firstFrame = parms->firstFrame;
    if(firstFrame<0)
	firstFrame = 0;
    else if(firstFrame >= availableFrames)
	firstFrame = availableFrames - 1;

    dimIndexType nFrames = parms->nFrames;
    if((nFrames <= 0) || (nFrames > availableFrames))
	nFrames = availableFrames;

    // Get dimensions of the show window
    int winDims[2];
    winDims[0] = pShow->getSpatialDimSize(0, 0);
    winDims[1] = pShow->getSpatialDimSize(1, 0);

    // We need as many cv::Mat objects as the number of frames to show
    std::vector<cv::Mat> mats(nFrames);

    // At this point, we have a bunch of 2D frames to show, so we can construct a cv::Mat object from our host buffer in one zero-copy shot

    // Build vector of mat objects from respective host buffers. Wrap around available frames if necessary
    // Note: when calling cv::Mat, rows first, cols second
    dimIndexType frame = firstFrame;
    for(dimIndexType cvFrame = 0; cvFrame < nFrames; cvFrame++) {
	mats[cvFrame] = cv::Mat(winDims[1], winDims[0], opencvDataType, pShow->getHostBuffer(pShow->getNDArray1DIndex(0, &frame)));
	if(++frame >= availableFrames)
	    frame = 0;
    }

    // Overlay text on each tile if there are more than one, but only if the window is large enough to accomodate it
    if(nTiles >= 2 && winDims[0] >= 50 && winDims[1] >= 15) {
	for(dimIndexType cvFrame = 0; cvFrame < nFrames; cvFrame++) {
	    for(dimIndexType tile = 0; tile < tileCoords.size(); tile++) {
		std::stringstream s;
		s.clear();
		s << tileText << ' ' << tile;

		// Draw a shadow text to enhance readability over almost-white images
		cv::Point textOrig;
		textOrig.x = tileCoords[tile][0]+1;
		textOrig.y = tileCoords[tile][1]+13;
		cv::putText(mats[cvFrame], s.str(), textOrig, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, cv::Scalar::all(0), 1);

		// Draw actual text
		textOrig.x = tileCoords[tile][0];
		textOrig.y = tileCoords[tile][1]+12;
		cv::putText(mats[cvFrame], s.str(), textOrig, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, cv::Scalar::all(255), 1);
	    }
	}
    }

    // Create a cvPlot window
    auto axes = CvPlot::plotImage(mats[0]);
    CvPlot::Window w(parms->title,axes, static_cast<unsigned>(parms->scaleFactor * winDims[1]), static_cast<unsigned>(parms->scaleFactor * winDims[0]));

    // Show!
    int key;
    unsigned int loop=0;
    for(;;) {
	bool keySaysBreak;
	bool loopSaysBreak;

	for(dimIndexType cvFrame = 0; cvFrame < nFrames; cvFrame++) {
            axes.find<CvPlot::Image>()->setMat(mats[cvFrame]);
            w.update();

	    // Note: CV docs say waitKey returns (int)(-1) if no key pressed but some versions return (int)255
	    key = cv::waitKey(1000 / parms->fps);
	    keySaysBreak = parms-> keyEnds & ((key != -1) && (key != 255));
	    if(keySaysBreak)
		break;
	}

	loopSaysBreak = (parms->loops != 0) && (++loop >= parms->loops);
	if(keySaysBreak || loopSaysBreak)
	    break;
    }
}

void Data::checkMatvarDims(matvar_t* matvar, const std::vector<dimIndexType>* pDims) {
	dimIndexType nDims = pDims->size();
		// matvar->dimx[0] is 1 for a row vector
	if ((matvar->dims[1] != pDims->at(0))) {
		throw(InvalidDimension("Data::checkMatvarSpatialDims", 0, matvar->dims[1], pDims->at(0)));
	}
	if(nDims > 1) {
		if ((matvar->dims[0] != pDims->at(1))) {
			throw(InvalidDimension("Data::checkMatvarSpatialDims", 1, matvar->dims[0], pDims->at(1)));
		}
	}
	for (dimIndexType i = 2; i < nDims; i++) {
		if (matvar->dims[i] != pDims->at(i)) {
			throw(InvalidDimension("Data::checkMatvarSpatialDims", i, matvar->dims[i], pDims->at(i)));
		}
	}

}

} /* namespace OpenCLIPER */

#undef DATA_DEBUG
