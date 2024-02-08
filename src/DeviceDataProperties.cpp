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
 * DeviceDataProperties.cpp
 *
 *  Created on: 10 de nov. de 2016
 *      Author: manrod
 */
#include <OpenCLIPER/DeviceDataProperties.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <OpenCLIPER/Data.hpp>
#include <OpenCLIPER/cl2hip.hpp>

// Uncomment to show class-specific debug messages
//#define DEVICEDATAPROPERTIES_DEBUG

#if !defined NDEBUG && defined DEVICEDATAPROPERTIES_DEBUG
    #define DEVICEDATAPROPERTIES_CERR(x) CERR(x)
#else
    #define DEVICEDATAPROPERTIES_CERR(x)
    #undef DEVICEDATAPROPERTIES_DEBUG
#endif

namespace OpenCLIPER {
DeviceDataProperties::DeviceDataProperties(const std::shared_ptr<CLapp>& pCLapp, std::shared_ptr<Data> pDataArg, bool copyDataToDevice) {
    queue = pCLapp->getCommandQueue();
    context = pCLapp->getContext();
    selected_device = pCLapp->getDevice();
#ifdef HAVE_HIP
    hipDevice = pCLapp->getHIPDevice();
    cl2hipKernel = pCLapp->getKernel("getDevicePointer");
#endif

    // store device base memory alignment for selected device (read in bits, stored in bytes)
    deviceMemBaseAddrAlignInBytes = pCLapp->getDevice().getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() / 8;

    this->pData = pDataArg.get();
    setApp(copyDataToDevice);
}

DeviceDataProperties::DeviceDataProperties(const std::shared_ptr<CLapp>& pCLapp, Data* pDataArg, bool copyDataToDevice) {
    queue = pCLapp->getCommandQueue();
    context = pCLapp->getContext();
    selected_device = pCLapp->getDevice();
#ifdef HAVE_HIP
    hipDevice = pCLapp->getHIPDevice();
    cl2hipKernel = pCLapp->getKernel("getDevicePointer");
#endif

    // store device base memory alignment for selected device (read in bits, stored in bytes)
    deviceMemBaseAddrAlignInBytes = pCLapp->getDevice().getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() / 8;

    this->pData = pDataArg;
    setApp(copyDataToDevice);
}

DeviceDataProperties::~DeviceDataProperties() {
    delApp();
    pData = nullptr;
}

/**
 * @brief Gets a generic pointer to host memory mapped to
 * device memory that stores an OpenCL buffer (pointer to object of class cl::Buffer).
 * @param[in] NDArrayIndex NDarray index for the vector of NDArrays
 * @return raw pointer to host memory (or nullptr if NDArrayIndex >= size of vector of NDArrays host buffers)
 */
void* DeviceDataProperties::getHostBuffer(dimIndexType NDArrayIndex) const {
    if(NDArrayIndex >= pHostBuffers->size()) {
	return nullptr;
    }
    return pHostBuffers->at(NDArrayIndex);
}

/**
 * @brief Gets pointer to data stored in device memory as a buffer
 * (pointer to object of class cl::Buffer).
 * @param[in] NDArrayIndex NDarray index for the vector of NDArrays
 * @return raw pointer to object of class cl::Buffer (or nullptr if NDArrayIndex >= size of vector of NDArrays device buffers)
 */

cl::Buffer* DeviceDataProperties::getDeviceBuffer(dimIndexType NDArrayIndex) {
    if(NDArrayIndex >= pDeviceBuffers->size()) {
	return nullptr;
    }
    return pDeviceBuffers->at(NDArrayIndex);
}

/**
 * @brief Binds a CLapp object (for accessing to OpenCL basic functions) to this Data object.
 * and copies images data from host memory to device memory.
 */
void DeviceDataProperties::setApp(bool copyDataToDevice) {
    std::stringstream ostream;
    ostream << errorPrefix;

    pData->calcDimsAndStridesVector(deviceMemBaseAddrAlignInBytes);

    if(!host2DeviceCommonChecks()) {
	return;
    }
    // Create empty device buffer and store it in
    // pData->getNDArrays()->at(i)->deviceBuffer
    createEmptyDeviceBuffers();
    // Map deviceBuffer to hostBuffer, storing mapped pointer in
    // pHostBuffer
    mapDeviceBufferToHost();
    mapDimsAndStrideDeviceBufferToHost();
    host2Device(copyDataToDevice);

#ifdef HAVE_HIP
    if(hipDevice != -1)
	pHIPDeviceBuffer = cl2hip(*pDeviceBuffer, context, queue, cl2hipKernel, hipDevice);
    else
	pHIPDeviceBuffer = nullptr;
#endif

}

/**
 * @brief Create empty device buffers for a Data object.
 *
 * Create 1 device buffer (on contiguous memory) for dataOffset, all NDArrays and dimensions and strides array, and 1 device
 * subbuffer for dataOffset, 1 device subbuffer per NDArray, and 1 device subbuffer for dimensions and strides array.
 */
void DeviceDataProperties::createEmptyDeviceBuffers() {
    dimIndexType minIndex, maxIndex;
    dimIndexType numberOfNDArrays = pData->getNDArrays()->size();
    dimIndexType sizeOfNDArrayInBytes;

    minIndex = 0;
    maxIndex = numberOfNDArrays - 1;
    dimIndexType offsetContiguousMemoryBetweenNDArraysInBytes;
    dimIndexType totalSizeOfContiguousMemoryInBytes = 0;

    DEVICEDATAPROPERTIES_CERR("CL_DEVICE_MEM_BASE_ADDR_ALIGN: " << deviceMemBaseAddrAlignInBytes * 8 << " bits" << std::endl);
    try {
	// Add size of dimensions and strides array plus offset rounded up to a multiple of CL_DEVICE_MEM_BASE_ADDR_ALIGN to size of contiguous
	// device memory
	dimsAndStridesArraySubbuferRoundedSize = CLapp::roundUp((pData->getDataDimsAndStridesVector()->size() + 1) * sizeof(dimIndexType), deviceMemBaseAddrAlignInBytes);
	totalSizeOfContiguousMemoryInBytes += dimsAndStridesArraySubbuferRoundedSize;

	// Add size of every NDArray (every NDArray may have a different size) to size of contiguous device memory

	for(dimIndexType i = minIndex; i <= maxIndex; i++) {
	    sizeOfNDArrayInBytes = (pData->getNDArrays()->at(i)->size()) * pData->getElementSize();
	    offsetContiguousMemoryBetweenNDArraysInBytes = CLapp::roundUp(sizeOfNDArrayInBytes, deviceMemBaseAddrAlignInBytes);
	    allNDArraysRoundedSizeInBytes += offsetContiguousMemoryBetweenNDArraysInBytes;
	}
	totalSizeOfContiguousMemoryInBytes += allNDArraysRoundedSizeInBytes;
	pCompleteDeviceBuffer = new cl::Buffer(context, CL_MEM_READ_WRITE, totalSizeOfContiguousMemoryInBytes, NULL);

	dimIndexType offsetSubbufferInBytes = 0;
	cl_buffer_region* pBufferCreateInfo;
	// First subbuffer contains array of data dimensions and strides plus offset to this array from first NDArray subbuffer
	pBufferCreateInfo = new cl_buffer_region({offsetSubbufferInBytes, dimsAndStridesArraySubbuferRoundedSize});
	pDataDimsAndStridesDeviceBuffer = new cl::Buffer(pCompleteDeviceBuffer->createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
		pBufferCreateInfo));
	delete(pBufferCreateInfo);

	// Update offsetSubbufferInBytes for future uses
	offsetSubbufferInBytes += dimsAndStridesArraySubbuferRoundedSize;
	dataDimsAndStridesOffset = dimsAndStridesArraySubbuferRoundedSize;

	// Offset from buffer beginning to start of first NDArray data
	dataStartOffset = offsetSubbufferInBytes;
	DEVICEDATAPROPERTIES_CERR("DeviceDataProperties::dataStartOffset: " << dataStartOffset << " bytes" << std::endl);
	DEVICEDATAPROPERTIES_CERR("DeviceDataProperties::dataDimsAndStridesOffset: " << dataDimsAndStridesOffset << " bytes" << std::endl);

	for(dimIndexType index = minIndex; index <= maxIndex; index++) {
	    sizeOfNDArrayInBytes = (pData->getNDArrays()->at(index)->size()) * pData->getElementSize();
	    // offsetIncrementInBytes is rounded up to nearest multiple of CL_DEVICE_MEM_BASE_ADDR_ALIGN
	    offsetContiguousMemoryBetweenNDArraysInBytes = CLapp::roundUp(sizeOfNDArrayInBytes, deviceMemBaseAddrAlignInBytes);
	    //DEVICEDATAPROPERTIES_CERR("offsetIncrementInBytes: " << offsetIncrementInBytes << std::endl);
	    pBufferCreateInfo = new cl_buffer_region({offsetSubbufferInBytes, sizeOfNDArrayInBytes});
	    pDeviceBuffers->push_back(
		new cl::Buffer(pCompleteDeviceBuffer->createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
			       pBufferCreateInfo)));
	    offsetSubbufferInBytes += offsetContiguousMemoryBetweenNDArraysInBytes;
	    delete(pBufferCreateInfo);
	    // Error: address of temporal object cannot be used
	    /*
	    pData->getNDArrays()->at(index)->pDeviceBuffer = &(pContiguousMemoryDeviceBuffer->createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
	            pBufferCreateInfo));
	     */
	}

	// Subbufer that contains all NDArray data buffers (without array of dimensions and strides)
	pBufferCreateInfo = new cl_buffer_region({dataStartOffset, allNDArraysRoundedSizeInBytes});
	pDeviceBuffer = new cl::Buffer(pCompleteDeviceBuffer->createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
				       pBufferCreateInfo));
	delete(pBufferCreateInfo);
    }
    catch(cl::Error& err) {
	BTTHROW(CLError(err), "DeviceDataProperties::createEmptyDeviceBuffers");
    }
    catch(std::exception& err) {
	BTTHROW(err, "DeviceDataProperties::createEmptyDeviceBuffers");
    }
}

/**
 * @brief Allocates a host memory region mapped to a device memory region previously allocated for a data OpenCL buffer (host
 * region is stored in pHostBuffer class field).
 */
void DeviceDataProperties::mapDeviceBufferToHost() {
    // queue.flush is not needed because blocking_map parameter (second parameter) is set to CL_TRUE (operation is blocked
    // until map is completed)
    // queue.flush();
    try {
	pCompleteHostBuffer =
	    queue.enqueueMapBuffer(*(pCompleteDeviceBuffer), CL_TRUE,
				   CL_MAP_READ | CL_MAP_WRITE, 0,
				   dimsAndStridesArraySubbuferRoundedSize + allNDArraysRoundedSizeInBytes);
	char* pHostBuffer;
	dimIndexType sizeOfNDArrayInBytes;
	// First buffer is for dimensions and strides; second and following for NDArrays data
	pHostBuffer = ((char*) pCompleteHostBuffer) + dimsAndStridesArraySubbuferRoundedSize;
	for(index1DType index = 0; index <= pData->getNDArrays()->size() - 1; index++) {
	    pHostBuffers->push_back(pHostBuffer);
	    sizeOfNDArrayInBytes = pData->getNDArrays()->at(index)->size() * pData->getElementSize();
	    pHostBuffer += CLapp::roundUp(sizeOfNDArrayInBytes, deviceMemBaseAddrAlignInBytes);
	}
    }
    catch(cl::Error& err) {
	BTTHROW(CLError(err), "DeviceDataProperties::mapDeviceBufferToHost");
    }
    catch(std::exception& err) {
	BTTHROW(err, "DeviceDataProperties::mapDeviceBufferToHost");
    }

}

/**
 * @brief Allocates a host memory region mapped to a device memory region previously allocated for the data dimensions and
 * strides OpenCL buffer (host region is  stored in pDataDimsAndStridesHostBuffer class variable).
 */
void DeviceDataProperties::mapDimsAndStrideDeviceBufferToHost() {
    if(pCompleteHostBuffer != nullptr) {
	pDataDimsAndStridesHostBuffer = pCompleteHostBuffer;
    }
}

/**
 * @brief Common checks before copying data from host memory to device memory.
 *
 * Checks if pData pointer is not null, and if size of pData pointed vector size is > 0.
 */
bool DeviceDataProperties::host2DeviceCommonChecks() {
    std::stringstream ostream;
    ostream << errorPrefix;
    if(pData == nullptr) {
	ostream << HOST2DEVICECOMMONCHECKSERRORPREFIX
		<< HOST2DEVICECOMMONCHECKSERRORSUFIX
		<< std::endl;
	DEVICEDATAPROPERTIES_CERR(ostream.str());
	return false;
    }
    if(pData->getNDArrays()->size() == 0) {
	ostream << HOST2DEVICECOMMONCHECKSERRORPREFIX
		<< HOST2DEVICECOMMONCHECKSERRORSUFIX
		<< std::endl;
	DEVICEDATAPROPERTIES_CERR(ostream.str());
	return false;
    }
    return true;
}

/**
 * @brief Common checks for every element before copying data from host memory to device memory.
 *
 * Checks pData->getNDArrays()->at(index) pointer is not null, and if width and height are > 0 (data should be 2D or 3D).
 * @param[in] width image width
 * @param[in] height image height
 * @param[in] depth image depth
 * @param[in] index index of data (selects an data from a data set belonging to a Data object)
 */
bool DeviceDataProperties::host2DeviceCommonChecksForElement(dimIndexType& width, dimIndexType& height,
	dimIndexType& depth, dimIndexType index) {
    std::stringstream ostream;
    ostream << errorPrefix;
    if(pData->getNDArrays()->at(index) == nullptr) {
	ostream << HOST2DEVICECOMMONCHECKSERRORPREFIX << "for " << index
		<< " element is "
		<< HOST2DEVICECOMMONCHECKSERRORSUFIX << std::endl;
	DEVICEDATAPROPERTIES_CERR(ostream.str());
	return false;
    }
    width = NDARRAYWIDTH(pData->getNDArrays()->at(index));
    height = NDARRAYHEIGHT(pData->getNDArrays()->at(index));
    depth = NDARRAYDEPTH(pData->getNDArrays()->at(index));
    if(depth == 0)
	depth = 1;
    // NDArray empty, non-recoverable error
    if((width == 0) || (height == 0)) {
	ostream << HOST2DEVICECOMMONCHECKSERRORPREFIX
		<< "for " << index
		<< " element is "
		<< HOST2DEVICECOMMONCHECKSERRORSUFIX << std::endl;
	DEVICEDATAPROPERTIES_CERR(ostream.str());
	return false;
    }
    return true;
}

/**
 * @brief Copy data stored in host memory to device memory (as buffers, images or both), common tasks for 1 NDArray..
 * @param[in] index index of NDArray (selects a NDArray object from the group of NDArray objects belonging to this Data object)
 */
void DeviceDataProperties::host2DeviceCommon(const dimIndexType index) {
    std::stringstream ostream;
    ostream << errorPrefix;

    try {
	dimIndexType width, height, depth;
	if(!host2DeviceCommonChecksForElement(width, height,
					      depth, index)) {
	    return;
	}

	const void* data;
	data = (const void*)(pData->getNDArray(index)->getHostDataAsVoidPointer());
	if(data == nullptr) {
	    ostream << HOST2DEVICECOMMONERRORPREFIX << "for " << index << " element is " << HOST2DEVICECOMMONERRORSUFIX
		    << std::endl;
	    DEVICEDATAPROPERTIES_CERR(ostream.str());
	    return;
	}

	copyHostDataToMappedHostBuffer(index);
	// Force copy of hostBuffer to its mapped deviceBuffer
	// (synchronization between device and mapped host memory
	// is not automatic)
	// queue.enqueueWriteBuffer(*(getDeviceBuffer(index)), CL_TRUE, 0,
	queue.enqueueWriteBuffer(*(getDeviceBuffer(index)), CL_FALSE, 0,
				 (pData->getNDArrays()->at(index)->size()) * pData->getElementSize(),
				 getHostBuffer(index), { });
	// queue.flush is not needed because blocking_write parameter (second parameter) is set to CL_TRUE (operation is blocked
	// until map is completed)
	//queue.flush();
    }
    catch(cl::Error& err) {
	BTTHROW(CLError(err), "DeviceDataProperties::host2DeviceCommon");
    }
    catch(std::exception& err) {
	BTTHROW(err, "DeviceDataProperties::host2DeviceCommon");
    }
}

/**
 * @brief Copy data stored in host memory to device memory (as buffers, images or both).
 */
void DeviceDataProperties::host2Device(bool copyDataToDevice) {
    if(pData == nullptr) {
	return;
    }
    if(!host2DeviceCommonChecks()) {
	return;
    }
    if (copyDataToDevice) {
    //if (true) {
	for(dimIndexType i = 0; i < pData->getNDArrays()->size(); i++) {
	    host2DeviceCommon(i);
	}
    }
    copyDimsAndStridesVectorDataToMappedHostAndDeviceBuffer();
    queue.finish();
}

/**
 * @brief Copy data stored as a vector in host memory to an OpenCL buffer in host memory (this memory has been previously
 * mapped to device memory).
 * @param[in] i index of image (selects an image from the group of images belonging to a Data object)
 */
void DeviceDataProperties::copyHostDataToMappedHostBuffer(dimIndexType i) {
    //void *memcpy(void *dest, const void *src, dimIndexType n)
    void* origin;
    origin = pData->getNDArray(i)->getHostDataAsVoidPointer();
    memcpy(getHostBuffer(i), origin, (pData->getNDArrays()->at(i)->size()) * pData->getElementSize());
}

/**
 * @brief Copy data dimensions and strides stored as a vector in host memory to an OpenCL buffer in host memory (this memory has
 * been previously mapped to device memory) and synchronizes host mapped memory an device memory.
 */
void DeviceDataProperties::copyDimsAndStridesVectorDataToMappedHostAndDeviceBuffer() {
    try {    //void *memcpy(void *dest, const void *src, dimIndexType n)
	memcpy(pDataDimsAndStridesHostBuffer, pData->getDataDimsAndStridesVector()->data(), pData->getDataDimsAndStridesVector()->size()*sizeof(dimIndexType));
	// store offset to dimensions and strides vector in the last (not first) position of pDataDimsAndStridesOffsetHostBuffer.
	// This way, for reading this value from a pointer to first element of image data we only have to go backwards 1 position
	// of a uint buffer
	dimIndexType numOfPosDataDimsAndStridesHostBuffer =
	    dimsAndStridesArraySubbuferRoundedSize / sizeof(dimIndexType);
	((dimIndexType*)pDataDimsAndStridesHostBuffer)[numOfPosDataDimsAndStridesHostBuffer - 1] = dataDimsAndStridesOffset;

	// Force copy of hostBuffer to its mapped deviceBuffer
	// (synchronization between device and mapped host memory
	// is not automatic)
	queue.enqueueWriteBuffer(*(pDataDimsAndStridesDeviceBuffer), CL_TRUE, 0,
				 dimsAndStridesArraySubbuferRoundedSize,
				 pDataDimsAndStridesHostBuffer, { });
    }
    catch(cl::Error& err) {
	BTTHROW(CLError(err), "DeviceDataProperties::copyDimsAndStridesVectorDataToMappedHostAndDeviceBuffer");
    }
    catch(std::exception& err) {
	BTTHROW(err, "DeviceDataProperties::copyDimsAndStridesVectorDataToMappedHostAndDeviceBuffer");
    }
}

/**
 * @brief Common checks before copying data from device memory to host memory.
 *
 * Checks if pData pointer is not null, and if size of pData pointed vector size is > 0.
 */
bool DeviceDataProperties::device2HostCommonChecks() {
    std::stringstream ostream;
    ostream << errorPrefix;
    if(pData == nullptr) {
	ostream << DEVICE2HOSTERRORPREFIX << DEVICE2HOSTERRORSUFIX
		<< std::endl;
	DEVICEDATAPROPERTIES_CERR(ostream.str());
	return false;
    }
    if(pData->getNDArrays()->size() == 0) {
	ostream << DEVICE2HOSTERRORPREFIX << DEVICE2HOSTERRORSUFIX
		<< std::endl;
	DEVICEDATAPROPERTIES_CERR(ostream.str());
	return false;
    }
    return true;
}

/**
 * @brief Common checks for every element before copying data from device memory to host memory.
 *
 * Checks pData->getNDArrays()->at(index) pointer is not null, and if width and height are > 0 (data should be 2D or 3D).
 * @param[in] width image width
 * @param[in] height image height
 * @param[in] depth image depth
 * @param[in] index index of NDArray (selects a NDArray object from the group of NDArray objects belonging to this Data object)
 */
bool DeviceDataProperties::device2HostCommonChecksForElement(dimIndexType& width, dimIndexType& height,
	dimIndexType& depth, const dimIndexType index) {
    std::stringstream ostream;
    ostream << errorPrefix;
    if(pData->getNDArrays()->at(index) == nullptr) {
	ostream << DEVICE2HOSTERRORPREFIX << "for " << index
		<< " element is "
		<< DEVICE2HOSTERRORSUFIX << std::endl;
	DEVICEDATAPROPERTIES_CERR(ostream.str());
	return false;
    }
    // NDArray empty, non-recoverable error
    width = NDARRAYWIDTH(pData->getNDArrays()->at(index));
    height = NDARRAYHEIGHT(pData->getNDArrays()->at(index));
    depth = NDARRAYDEPTH(pData->getNDArrays()->at(index));
    if(depth == 0)
	depth = 1;
    if((width == 0) || (height == 0)) {
	ostream << DEVICE2HOSTERRORPREFIX
		<< "for " << index
		<< " element is "
		<< DEVICE2HOSTERRORSUFIX << std::endl;
	DEVICEDATAPROPERTIES_CERR(ostream.str());
	return false;
    }
    return true;
}

/**
 * @brief Copy data stored in device memory to host memory (from buffers or images), common tasks for 1 NDArray..
 * @param[in] index index of NDArray (selects a NDArray object from the group of NDArray objects belonging to this Data object)
 */
void DeviceDataProperties::device2HostCommon() {
    try {
	std::stringstream ostream;
	ostream << errorPrefix;
	if(!device2HostCommonChecks()) {
	    return;
	}

	for(dimIndexType index = 0; index <= pData->getNDArrays()->size() - 1; index++) {
	    if(getDeviceBuffer(index) == nullptr) {
		ostream << DEVICE2HOSTERRORPREFIX << "for " << index
			<< " element is " << DEVICE2HOSTERRORSUFIX << std::endl;
		DEVICEDATAPROPERTIES_CERR(ostream.str());
		return;
	    }
	}
	// Fix dimensions
	DEVICEDATAPROPERTIES_CERR("pCompleteDeviceBuffer size: " <<
				  pCompleteDeviceBuffer->getInfo<CL_MEM_SIZE>() << " bytes\n");
	DEVICEDATAPROPERTIES_CERR("dimsAndStridesArraySubbuferRoundedSize + allNDArraysRoundedSizeInBytes: " <<
				  dimsAndStridesArraySubbuferRoundedSize + allNDArraysRoundedSizeInBytes << " bytes\n");
	queue.enqueueReadBuffer(*(pCompleteDeviceBuffer), CL_TRUE, 0,
				dimsAndStridesArraySubbuferRoundedSize + allNDArraysRoundedSizeInBytes,
				pCompleteHostBuffer, { });
    }
    catch(cl::Error& err) {
	BTTHROW(CLError(err), "DeviceDataProperties::device2HostCommon");
    }
    catch(std::exception& err) {
	BTTHROW(err, "DeviceDataProperties::device2HostCommon");
    }
}

/**
 * @brief Copy data stored in device memory to host memory (from buffers or images).
 * @param[in] queueFinish true if queue.finish() method must be called to guarantee that kernel execution has finished before
 * copying data back to host memory
 */
void DeviceDataProperties::device2Host(bool queueFinish) {
    if(queueFinish) {
	queue.finish();
    }
    if(pData == nullptr) {
	return;
    }
    if(!device2HostCommonChecks()) {
	return;
    }
    device2HostCommon();
}

/**
 * @brief Unbind a CLapp object (for accessing to OpenCL basic functions) from this Data object.
 *
 * Mapped memory is unmapped, and device image and buffer memory is freed.
 */
void DeviceDataProperties::delApp() {
    try {
	if(pData == nullptr) {
	    return;
	}
	if(pData->getNDArrays()->size() == 0) {
	    return;
	}

	// free device memory subbufer for dimensions vector and set corresponding host subbuffer pointer to nullptr
	// (this pointer point to the first part of the complete host buffer that is pinned to the complete device memory buffer)
	if((getDataDimsAndStridesHostBuffer() != nullptr) && (getDataDimsAndStridesDeviceBuffer() != nullptr)) {
	    delete(pDataDimsAndStridesDeviceBuffer);
	    pDataDimsAndStridesDeviceBuffer = nullptr;
	    pDataDimsAndStridesHostBuffer = nullptr;
	}

	// free device memory subbuffers and set corresponding host subbufer pointers to
	// nullptr (these pointers point to parts of the complete host buffer that is pinned
	// to the complete device memory buffer)
	if((getHostBuffer(0) != nullptr) && (getDeviceBuffer(0) != nullptr)) {
	    for(dimIndexType i = 0; i < pData->getNDArrays()->size(); i++) {
		delete(getDeviceBuffer(i)); // free of device buffer create by createSubbufer
		pDeviceBuffers->at(i) = nullptr;
		pHostBuffers->at(i) = nullptr;
	    }
	    pDeviceBuffers->resize(0);
	    pHostBuffers->resize(0);
	}

	// free device memory subbuffer containing all NDArrays data (without array of dimensions and strides)
	// there is no pHostBuffer (only vector pHostBuffers as all this pointers are void* and a pointer to first position
	// in host memory is the same as a pointer to the complete NDArrays data host memory)
	if(getDeviceBuffer() != nullptr) {
	    delete(pDeviceBuffer);
	    pDeviceBuffer = nullptr;
	}
	queue.enqueueUnmapMemObject(*(pCompleteDeviceBuffer), pCompleteHostBuffer);
	delete(pCompleteDeviceBuffer);
	pCompleteDeviceBuffer = nullptr;
	pCompleteHostBuffer = nullptr;
    }
    catch(cl::Error& err) {
	BTTHROW(CLError(err), "DeviceDataProperties::delApp");
    }
    catch(std::exception& err) {
	BTTHROW(err, "DeviceDataProperties::delApp");
    }
}
}

