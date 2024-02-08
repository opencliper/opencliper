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

#include <OpenCLIPER/Trajectories.hpp>
#include <LPISupport/InfoItems.hpp>

// Uncomment to show class-specific debug messages
#define TRAJECTORIES_DEBUG

#if !defined NDEBUG && defined TRAJECTORIES_DEBUG
    #define TRAJECTORIES_CERR(x) CERR(x)
#else
    #define TRAJECTORIES_CERR(x)
    #undef TRAJECTORIES_DEBUG
#endif

namespace OpenCLIPER {
/**
* @brief Constructor that stores data of a matlab variable previously read (usually from a KData object) as a group of
* NDArray objects.
*
* @param[in] pMatlabVar matlab variable previously read from file
* @param[in] numOfSpatialDimensions the number of spatial dimensions of the data read
* @param[in] automaticStoreOnDevice sets the automatic copy of data from host memory to device memory feature if true
* (default value: true)
* @param[in] host2DeviceSync format used for storing data in device memory (buffers, images or both)
*/
Trajectories::Trajectories(const std::shared_ptr<CLapp>& pCLapp, matvar_t* pMatlabVar, dimIndexType numOfSpatialDimensions, SyncSource host2DeviceSync):
    Data() {
    if(pMatlabVar == nullptr) {
	BTTHROW(std::invalid_argument("pointer to matlab variable for trajectories is nullptr"), "Trajectories::Trajectories");
    }

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

#ifdef TRAJECTORIES_DEBUG
    BEGIN_TIME(bTReadingTrajectoriesDataMatVar);
    TRAJECTORIES_CERR("Trajectories reading data...\n");
#endif

    ((Data*)(this))->loadMatlabHostData(pMatlabVar, numOfSpatialDimensions, numOfNDArraysToBeRead);

    #ifdef TRAJECTORIES_DEBUG
    TRAJECTORIES_CERR("Done\n");
    END_TIME(eTReadingTrajectoriesDataMatVar);
    TIME_DIFF_TYPE diffTReadingTrajectoriesDataMatVar;
    TIME_DIFF(diffTReadingTrajectoriesDataMatVar, bTReadingTrajectoriesDataMatVar, eTReadingTrajectoriesDataMatVar);
    TRAJECTORIES_CERR("Elapsed time: " << diffTReadingTrajectoriesDataMatVar << " s" << std::endl);
#endif

    setApp(pCLapp, true);
}

/**
 * @brief Constructor that creates a Trajectories object from another Trajectories object.
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData source Trajectories object
 * @param[in] copyData if true also data (not only dimensions) are copied from sourceData object to this object
 */
Trajectories::Trajectories(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Trajectories>& sourceData, bool copyData): Data(sourceData, copyData) {
    setApp(pCLapp, copyData);
}

/**
 * @brief Constructor that creates an uninitialized Trajectories object from another Trajectories object.
 * Element data type for the new object is specified by the caller
 * @param[in] pCLapp pointer to CLapp object (contains an initialized OpenCL environment)
 * @param[in] sourceData Object to copy data structure from
 * @param[in] newElementDataType Data type of the newly created object
 */
Trajectories::Trajectories(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Trajectories>& sourceData, ElementDataType newElementDataType): Data(sourceData, newElementDataType) {
    setApp(pCLapp, false);
}

/**
 * @brief Virtual copy "constructor"
 * @param[in] deepCopy copy stored data if true; only data structure if false (shallow copy)
 * @return A copy of this object
 */
std::shared_ptr<Data> Trajectories::clone(bool deepCopy) const {
    return std::make_shared<Trajectories>(pCLapp, shared_from_this(), deepCopy);
}

/**
 * @brief Virtual copy "constructor"
 * @param[in] newElementDataType data type of the newly created object. Shallow copy is implied.
 * @return An empty copy of this object that contains elements of type "newElementDataType"
 */
std::shared_ptr<Data> Trajectories::clone(ElementDataType newElementDataType) const {
    return std::make_shared<Trajectories>(pCLapp, shared_from_this(), newElementDataType);
}


} // namespace OpenCLIPER

#undef TRAJECTORIES_DEBUG
