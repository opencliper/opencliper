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
#ifndef INCLUDE_OPENCLIPER_DATAPROPERTIES_HPP_
#define INCLUDE_OPENCLIPER_DATAPROPERTIES_HPP_

#include<OpenCLIPER/defs.hpp>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
    #include<OpenCL/cl.hpp>
#else
    #ifdef HAVE_OPENCL_HPP
	#include<CL/opencl.hpp>
    #else
	#include<CL/cl2.hpp>
    #endif
#endif

#ifdef HAVE_HIP
    #define __HIP_PLATFORM_HCC__
    #include "hip/hip_runtime_api.h"
#endif

namespace OpenCLIPER {
class CLapp;
class Data;
/// Class Data - Class that includes data and properties common to k-space and x-space images.
class DeviceDataProperties {
	friend class Data;
	friend class CLapp;
	friend class Process;
	friend class XImageSum;
    public:
	DeviceDataProperties(const std::shared_ptr<CLapp>& pCLapp, std::shared_ptr<Data> pData, bool copyDataToDevice);
	DeviceDataProperties(const std::shared_ptr<CLapp>& pCLapp, Data* pData, bool copyDataToDevice);
	~DeviceDataProperties();
	void host2Device(bool copyDataToDevice);
	void device2Host(bool queueFinish = true);

	/**
	 * @brief Gets pointer to spatial and temporal data dimensions and their strides stored in host memory as an array.
	 * @return generic pointer to array of spatial and temporal data dimensions and strides
	 */
	const void* getDataDimsAndStridesHostBuffer() const {
	    return pDataDimsAndStridesHostBuffer;
	}

	/**
	 * @brief Gets pointer to spatial and temporal data dimensions and their strides stored in contiguous device memory as a buffer
	 * (pointer to object of class cl::Buffer).
	 * @return raw pointer to object of class cl::Buffer
	 */
	cl::Buffer* getDataDimsAndStridesDeviceBuffer() const {
	    return pDataDimsAndStridesDeviceBuffer;
	}

	cl::Buffer* getCompleteDeviceBuffer() {
	    return pCompleteDeviceBuffer;
	}

	cl::Buffer* getDeviceBuffer() {
	    return pDeviceBuffer;
	}

	void* getHIPDeviceBuffer() {
	    return pHIPDeviceBuffer;
	}

	/**
	 * @brief Gets the offset to start of NDArray data inside the contiguous device memory buffer
	 * @return the offset to first NDArray
	 */
	dimIndexType getDataStartOffset() {
	    return dataStartOffset;
	}

	/**
	 * @brief Gets the offset to dimensions and strides array inside the contiguous device memory buffer
	 * @return the offset to dimensions and strides array
	 */
	dimIndexType getDataDimsAndStridesOffset() {
	    return dataDimsAndStridesOffset;
	}

    protected:
	Data* getData() {
	    return pData;
	}

	void* getHostBuffer(dimIndexType NDArrayIndex) const;
	cl::Buffer* getDeviceBuffer(dimIndexType NDArrayIndex);

    private:
	static constexpr const char* errorPrefix = "DeviceDataProperties::";
	static constexpr const char* SETAPPERRORPREFIX = "setApp: host data";
	static constexpr const char* SETAPPERRORSUFIX = "empty, nothing to store in device memory";
	static constexpr const char* HOST2DEVICECOMMONCHECKSERRORPREFIX = "host2DeviceCommonChecks: Host data ";
	static constexpr const char* HOST2DEVICECOMMONCHECKSERRORSUFIX = "empty, nothing to store in device memory";
	static constexpr const char* HOST2DEVICECOMMONERRORPREFIX = "host2DeviceCommon: Host data ";
	static constexpr const char* HOST2DEVICECOMMONERRORSUFIX = "empty, nothing to store in device memory";
	static constexpr const char* DEVICE2HOSTERRORPREFIX = "device2Host: Device data ";
	static constexpr const char* DEVICE2HOSTERRORSUFIX = "empty, nothing to store in host memory";

	void setApp(bool copyDataToDevice);
	void createEmptyDeviceBuffers();
	void mapDeviceBufferToHost();
	void mapDimsAndStrideDeviceBufferToHost();
	void host2DeviceCommon(const dimIndexType index);
	bool host2DeviceCommonChecks();
	bool host2DeviceCommonChecksForElement(dimIndexType& width, dimIndexType& height, dimIndexType& depth, dimIndexType index);

	void copyHostDataToMappedHostBuffer(dimIndexType index);
	void copyDimsAndStridesVectorDataToMappedHostAndDeviceBuffer();

	bool device2HostCommonChecks();
	bool device2HostCommonChecksForElement(dimIndexType& width, dimIndexType& height,
					       dimIndexType& depth, const dimIndexType index);
	void device2HostCommon();

	void delApp();

	cl::CommandQueue queue;
	cl::Context context;
	cl::Device selected_device;

#ifdef HAVE_HIP
	cl::Kernel cl2hipKernel;
	hipDevice_t hipDevice;
#endif

	// Must be set to pHostData->data value from setData method of ConcreteNDArray (so they must have protected visibility)
	/** @brief buffers in host memory mapped from buffer in device memory. */
	std::vector<void*>* pHostBuffers = new std::vector<void*>();

	/** @brief array of dimensions and strides and data of all NDArrays in contiguous device memory (CPU/GPU) as one cl::Buffer type */
	cl::Buffer* pCompleteDeviceBuffer = nullptr;

	/** @brief data of all NDArrays of a Data object in contiguous device memory (CPU/GPU) as one cl::Buffer type */
	cl::Buffer* pDeviceBuffer = nullptr;

	/// @brief HIP pointer to NDArray data
	void* pHIPDeviceBuffer = nullptr;

	/** @brief data of every NDArray in device memory (CPU/GPU) as a vector of elements of cl::Buffer type. */
	std::vector<cl::Buffer*>* pDeviceBuffers = new std::vector<cl::Buffer*>();

	/** @brief Offset from buffer beginning to start of dimensions and strides array */
	dimIndexType dataStartOffset = -1;

	/** @brief offset from beginning of contiguous device memory to start of dimensions and strides array */
	dimIndexType dataDimsAndStridesOffset = -1;


	/** @brief spatial and temporal image dimensions and their strides in host memory as a void* type*/
	void* pDataDimsAndStridesHostBuffer = nullptr;

	/** @brief spatial and temporal image dimensions and their strides in device memory (CPU/GPU) as a cl::Buffer type */
	cl::Buffer* pDataDimsAndStridesDeviceBuffer = nullptr;

	/// @brief Pointer to object containing data
	Data* pData = nullptr;
	std::vector<DataHandle> subDataHandles;

	/// @brief stores CL_DEVICE_MEM_BASE_ADDR_ALIGN device property
	dimIndexType deviceMemBaseAddrAlignInBytes = 0;

	dimIndexType dimsAndStridesArraySubbuferRoundedSize = 0;

	dimIndexType allNDArraysRoundedSizeInBytes = 0;
	void* pCompleteHostBuffer;
};
}
/* namespace OpenCLIPER */
#endif /* INCLUDE_OPENCLIPER_DATAPROPERTIES_HPP_ */

