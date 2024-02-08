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
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <functional>
#include <OpenCLIPER/DeviceDataProperties.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <OpenCLIPER/Data.hpp>
#include <OpenCLIPER/Process.hpp>
#include <OpenCLIPER/processes/Complex2Real.hpp>
#include <LPISupport/ProgramConfig.hpp>

// Uncomment to show class-specific debug messages
// #define CLAPP_DEBUG

#if !defined NDEBUG && defined CLAPP_DEBUG
    #define CLAPP_CERR(x) CERR(x)
#else
    #define CLAPP_CERR(x)
    #undef CLAPP_DEBUG
#endif

// Debug version of clEnqueueNDRangeKernel which inserts a clFinish() after kernel launch (via gcc -Wl,-wrap).
#ifndef NDEBUG
cl_int __wrap_clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset,
				     const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list,
				     const cl_event* event_wait_list, cl_event* event) {

    cl_int ret;
    ret = __real_clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size,
					num_events_in_wait_list, event_wait_list, event);
    clFinish(command_queue);
    return ret;
}
#endif

// We need a mutex to protect the CLapp::dataMap structure (std::map is not thread-safe)
std::mutex dataMapMutex;

namespace OpenCLIPER {

/// Map with OpenCL error number as keys and strings describing errors as values
std::map<const cl_int, const char*> CLapp::errStrings = {
    {CL_SUCCESS, "CL_SUCCESS"},
    {CL_DEVICE_NOT_FOUND, "CL_DEVICE_NOT_FOUND"},
    {CL_DEVICE_NOT_AVAILABLE, "CL_DEVICE_NOT_AVAILABLE"},
    {CL_COMPILER_NOT_AVAILABLE, "CL_COMPILER_NOT_AVAILABLE"},
    {CL_MEM_OBJECT_ALLOCATION_FAILURE, "CL_MEM_OBJECT_ALLOCATION_FAILURE"},
    {CL_OUT_OF_RESOURCES, "CL_OUT_OF_RESOURCES"},
    {CL_OUT_OF_HOST_MEMORY, "CL_OUT_OF_HOST_MEMORY"},
    {CL_PROFILING_INFO_NOT_AVAILABLE, "CL_PROFILING_INFO_NOT_AVAILABLE"},
    {CL_MEM_COPY_OVERLAP, "CL_MEM_COPY_OVERLAP"},
    {CL_IMAGE_FORMAT_MISMATCH, "CL_IMAGE_FORMAT_MISMATCH"},
    {CL_IMAGE_FORMAT_NOT_SUPPORTED, "CL_IMAGE_FORMAT_NOT_SUPPORTED"},
    {CL_BUILD_PROGRAM_FAILURE, "CL_BUILD_PROGRAM_FAILURE"},
    {CL_MAP_FAILURE, "CL_MAP_FAILURE"},
    {CL_MISALIGNED_SUB_BUFFER_OFFSET, "CL_MISALIGNED_SUB_BUFFER_OFFSET"},
    {CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"},
    {CL_INVALID_VALUE, "CL_INVALID_VALUE"},
    {CL_INVALID_DEVICE_TYPE, "CL_INVALID_DEVICE_TYPE"},
    {CL_INVALID_PLATFORM, "CL_INVALID_PLATFORM"},
    {CL_INVALID_DEVICE, "CL_INVALID_DEVICE"},
    {CL_INVALID_CONTEXT, "CL_INVALID_CONTEXT"},
    {CL_INVALID_QUEUE_PROPERTIES, "CL_INVALID_QUEUE_PROPERTIES"},
    {CL_INVALID_COMMAND_QUEUE, "CL_INVALID_COMMAND_QUEUE"},
    {CL_INVALID_HOST_PTR, "CL_INVALID_HOST_PTR"},
    {CL_INVALID_MEM_OBJECT, "CL_INVALID_MEM_OBJECT"},
    {CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"},
    {CL_INVALID_IMAGE_SIZE, "CL_INVALID_IMAGE_SIZE"},
    {CL_INVALID_SAMPLER, "CL_INVALID_SAMPLER"},
    {CL_INVALID_BINARY, "CL_INVALID_BINARY"},
    {CL_INVALID_BUILD_OPTIONS, "CL_INVALID_BUILD_OPTIONS"},
    {CL_INVALID_PROGRAM, "CL_INVALID_PROGRAM"},
    {CL_INVALID_PROGRAM_EXECUTABLE, "CL_INVALID_PROGRAM_EXECUTABLE"},
    {CL_INVALID_KERNEL_NAME, "CL_INVALID_KERNEL_NAME"},
    {CL_INVALID_KERNEL_DEFINITION, "CL_INVALID_KERNEL_DEFINITION"},
    {CL_INVALID_KERNEL, "CL_INVALID_KERNEL"},
    {CL_INVALID_ARG_INDEX, "CL_INVALID_ARG_INDEX"},
    {CL_INVALID_ARG_VALUE, "CL_INVALID_ARG_VALUE"},
    {CL_INVALID_ARG_SIZE, "CL_INVALID_ARG_SIZE"},
    {CL_INVALID_KERNEL_ARGS, "CL_INVALID_KERNEL_ARGS"},
    {CL_INVALID_WORK_DIMENSION, "CL_INVALID_WORK_DIMENSION"},
    {CL_INVALID_WORK_GROUP_SIZE, "CL_INVALID_WORK_GROUP_SIZE"},
    {CL_INVALID_WORK_ITEM_SIZE, "CL_INVALID_WORK_ITEM_SIZE"},
    {CL_INVALID_GLOBAL_OFFSET, "CL_INVALID_GLOBAL_OFFSET"},
    {CL_INVALID_EVENT_WAIT_LIST, "CL_INVALID_EVENT_WAIT_LIST"},
    {CL_INVALID_EVENT, "CL_INVALID_EVENT"},
    {CL_INVALID_OPERATION, "CL_INVALID_OPERATION"},
    {CL_INVALID_GL_OBJECT, "CL_INVALID_GL_OBJECT"},
    {CL_INVALID_BUFFER_SIZE, "CL_INVALID_BUFFER_SIZE"},
    {CL_INVALID_MIP_LEVEL, "CL_INVALID_MIP_LEVEL"},
    {CL_INVALID_GLOBAL_WORK_SIZE, "CL_INVALID_GLOBAL_WORK_SIZE"},
    {CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR, "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR"},
    {CL_PLATFORM_NOT_FOUND_KHR, "CL_PLATFORM_NOT_FOUND_KHR"},
    //{CL_INVALID_PROPERTY_EXT,"CL_INVALID_PROPERTY_EXT"},
    {CL_DEVICE_PARTITION_FAILED_EXT, "CL_DEVICE_PARTITION_FAILED_EXT"},
    {CL_INVALID_PARTITION_COUNT_EXT, "CL_INVALID_PARTITION_COUNT_EXT"}
    //{CL_INVALID_DEVICE_QUEUE,"CL_INVALID_DEVICE_QUEUE"},
    //{CL_INVALID_PIPE_SIZE,"CL_INVALID_PIPE_SIZE"
};

/// @brief These methods substitute public constructors.
// Not strictly needed now, but this allows us to avoid two-phase initializations in case we need to do something with
/// a fully constructed CLapp before returning (this has happened before)
std::shared_ptr<CLapp> CLapp::create() {
    auto pThisCLapp = create(PlatformTraits(), DeviceTraits());

    return pThisCLapp;
}

std::shared_ptr<CLapp> CLapp::create(const PlatformTraits& platformTraits, const DeviceTraits& deviceTraits) {
    auto pThisCLapp = std::shared_ptr<CLapp> (new CLapp()); //make_shared requires a public constructor. Use std:shared_ptr(new CLapp()) instead
    pThisCLapp->init(platformTraits, deviceTraits);

    return pThisCLapp;
}

/**
 * @brief Destructor
 */
CLapp::~CLapp() {
    const std::lock_guard<std::mutex> lock(dataMapMutex);

#ifdef CLAPP_DEBUG
    size_t dataMapSize = dataMap.size();
    std::cerr<<"dataMap size: " << dataMapSize << std::endl;
    std::cerr<<"Erasing data map..." << std::endl;
#endif

    dataMap.erase(dataMap.begin(), dataMap.end());

#ifdef CLAPP_DEBUG
    std::cerr<<"Done" << std::endl;
    dataMapSize = dataMap.size();
    std::cerr<<"dataMap size: " << dataMapSize << std::endl;
    std::cerr<<"Done\n";
#endif
}

/**
 * @brief Initialize platform, device, context and command queue with default values.
 * This means choosing the a-priori fastest device and its platform.
 */
void CLapp::init() {
    init(PlatformTraits(), DeviceTraits());
}

/**
 * @brief Initializes platform, device, context and command queue.
 * @param[in] platformTraits platform requirements: name, vendor, version, extensions. All may be unspecified
 * @param[in] deviceTraits device requirements: type, name, vendor, version, extensions and command queue properties. All may be unspecified
 * @param[in] quiet do not write anything to stderr
 */
void CLapp::init(const PlatformTraits& platformTraits, const DeviceTraits& deviceTraits, bool quiet) {
    std::string stringValue;

    // Always show platform/device information in debug mode
#ifndef NDEBUG
    quiet = false;
#endif

    // Get platform version we are built with
#if CL_HPP_TARGET_OPENCL_VERSION == 100
    unsigned buildPlatformVersionMajor = 1;
    unsigned buildPlatformVersionMinor = 0;
#elif CL_HPP_TARGET_OPENCL_VERSION == 110
    unsigned buildPlatformVersionMajor = 1;
    unsigned buildPlatformVersionMinor = 1;
#elif CL_HPP_TARGET_OPENCL_VERSION == 120
    unsigned buildPlatformVersionMajor = 1;
    unsigned buildPlatformVersionMinor = 2;
#elif CL_HPP_TARGET_OPENCL_VERSION == 200
    unsigned buildPlatformVersionMajor = 2;
    unsigned buildPlatformVersionMinor = 0;
#endif

    // Precalculate requested platform version numbers if needed
    unsigned requestedPlatformVersionMajor = 1;
    unsigned requestedPlatformVersionMinor = 0;
    if(!platformTraits.version.empty()) {
	unsigned pointPos = platformTraits.version.find('.');

	try {
	    requestedPlatformVersionMajor = std::stoul(platformTraits.version.substr(0, pointPos));
	    requestedPlatformVersionMinor = std::stoul(platformTraits.version.substr(pointPos + 1));
	}
	catch(const std::exception& e) {
	    BTTHROW(CLError(CL_INVALID_PLATFORM, "Invalid platform version specified"), "CLapp::init");
	}
    }

    // Precalculate requested device version numbers if needed
    unsigned requestedDeviceVersionMajor = 1;
    unsigned requestedDeviceVersionMinor = 0;
    if(!deviceTraits.version.empty()) {
	unsigned pointPos = deviceTraits.version.find('.');

	try {
	    requestedDeviceVersionMajor = std::stoul(deviceTraits.version.substr(0, pointPos));
	    requestedDeviceVersionMinor = std::stoul(deviceTraits.version.substr(pointPos + 1));
	}
	catch(const std::exception& e) {
	    BTTHROW(CLError(CL_INVALID_DEVICE, "Invalid device version specified"), "CLapp::init");
	}
    }

    //------------------------------------------------------------
    // 1. Get all suitable platforms according to specified traits
    //------------------------------------------------------------
    std::vector<cl::Platform> candidatePlatforms;

    // Get all available CL platforms
    cl::Platform::get(&candidatePlatforms);

    std::vector<cl::Platform>::iterator currentPlatform = candidatePlatforms.begin();
    while(currentPlatform != candidatePlatforms.end()) {

	// Discard platforms whose version is lower than we built for
	stringValue = currentPlatform->getInfo<CL_PLATFORM_VERSION>();

	// Get currentPlatform's version
	// Returned version format is: "OpenCL <major>.<minor> <platform_specific_details>"
	// The fixed string "OpenCL " is 7 characters long
	// This value is provided by the device driver, so we should be able to trust it and use atoi
	unsigned pointPos = stringValue.find('.', 7);
	unsigned versionEndPos = stringValue.find(' ', 7);
	unsigned thisPlatformVersionMajor = ::atoi(stringValue.substr(7, pointPos - 7).c_str());
	unsigned thisPlatformVersionMinor = ::atoi(stringValue.substr(pointPos + 1, versionEndPos - pointPos - 1).c_str());

	if((buildPlatformVersionMajor > thisPlatformVersionMajor) ||
		((buildPlatformVersionMajor == thisPlatformVersionMajor) && (buildPlatformVersionMinor > thisPlatformVersionMinor))) {
	    CERR("Discarding platform \"" << currentPlatform->getInfo<CL_PLATFORM_NAME>()
		 << "\" because its version is " << thisPlatformVersionMajor << '.' << thisPlatformVersionMinor
		 << " and this program was built with CL_HPP_TARGET_OPENCL_VERSION=" << CL_HPP_TARGET_OPENCL_VERSION << "\n");

	    currentPlatform = candidatePlatforms.erase(currentPlatform);
	    continue;
	}

	// Discard platforms whose version is lower than the requested version
	if(!platformTraits.version.empty()) {
	    if((requestedPlatformVersionMajor > thisPlatformVersionMajor) ||
		    ((requestedPlatformVersionMajor == thisPlatformVersionMajor) && (requestedPlatformVersionMinor > thisPlatformVersionMinor))) {
		CERR("Discarding platform \"" << currentPlatform->getInfo<CL_PLATFORM_NAME>()
		     << "\" because its version \"" << thisPlatformVersionMajor << '.' << thisPlatformVersionMinor
		     << "\" is older than the requested platform version \"" << platformTraits.version << "\"\n");

		currentPlatform = candidatePlatforms.erase(currentPlatform);
		continue;
	    }
	}

	// Discard platforms which don't have devices of the requested type
	std::vector<cl::Device> tmpDevices;
	try {
	    currentPlatform->getDevices(deviceTraits.type, &tmpDevices);
	}
	catch(cl::Error& err) {
	    if(err.err() == CL_DEVICE_NOT_FOUND) {
		CERR("Discarding platform \"" << currentPlatform->getInfo<CL_PLATFORM_NAME>() <<
		     "\" because it doesn't have any devices of the requested type\n");

		currentPlatform = candidatePlatforms.erase(currentPlatform);
		continue;
	    }
	    else
		throw CLError(err) ;
	}

	// Discard platforms not matching the requested name
	if(!platformTraits.name.empty()) {
	    stringValue = currentPlatform->getInfo<CL_PLATFORM_NAME>();
	    if(stringValue.find(platformTraits.name) == std::string::npos) {
		CERR("Discarding platform \"" << currentPlatform->getInfo<CL_PLATFORM_NAME>() <<
		     "\" because it doesn't match the requested platform name \"" << platformTraits.name << "\"\n");

		currentPlatform = candidatePlatforms.erase(currentPlatform);
		continue;
	    }
	}

	// Discard platforms not matching the requested vendor
	if(!platformTraits.vendor.empty()) {
	    stringValue = currentPlatform->getInfo<CL_PLATFORM_VENDOR>();
	    if(stringValue.find(platformTraits.vendor) == std::string::npos) {
		CERR("Discarding platform \"" << currentPlatform->getInfo<CL_PLATFORM_NAME>() <<
		     "\" because it doesn't match the requested platform vendor \"" << platformTraits.vendor << "\"\n");

		currentPlatform = candidatePlatforms.erase(currentPlatform);
		continue;
	    }
	}

	// Discard platforms not matching the requested extensions
	if(!platformTraits.extensions.empty()) {
	    stringValue = currentPlatform->getInfo<CL_PLATFORM_EXTENSIONS>();

	    bool allExtensionsSupported = true;
	    std::vector<std::string>::const_iterator currentExtension = platformTraits.extensions.begin();
	    while(currentExtension != platformTraits.extensions.end()) {
		if(stringValue.find(*currentExtension) == std::string::npos) {
		    CERR("Discarding platform \"" << currentPlatform->getInfo<CL_PLATFORM_NAME>() <<
			 "\" because it doesn't have the requested extension \"" << *currentExtension << "\"\n");
		    allExtensionsSupported = false;
		    break;
		}
	    }

	    if(!allExtensionsSupported) {
		currentPlatform = candidatePlatforms.erase(currentPlatform);
		continue;
	    }
	}

	++currentPlatform;
    } // while(currentPlatform!=candidatePlatforms.end())

    //-------------------------------------------------------------------------------------
    // 2. Get all suitable devices within selected platforms, according to specified traits
    //-------------------------------------------------------------------------------------
    std::vector<cl::Device> candidateDevices;

    currentPlatform = candidatePlatforms.begin();
    while(currentPlatform != candidatePlatforms.end()) {

	// Get devices in the current platform. Remaining platforms must have at least one device of the requested type, so
	// we can do without try/catch
	std::vector<cl::Device> tmpDevices;
	currentPlatform->getDevices(deviceTraits.type, &tmpDevices);
	candidateDevices.insert(candidateDevices.end(), tmpDevices.begin(), tmpDevices.end());

	std::vector<cl::Device>::iterator currentDevice = candidateDevices.begin();
	while(currentDevice != candidateDevices.end()) {

	    // Discard devices not matching the requested name
	    if(!deviceTraits.name.empty()) {
		stringValue = currentDevice->getInfo<CL_DEVICE_NAME>();
		if(stringValue.find(deviceTraits.name) == std::string::npos) {
		    CERR("Discarding device \"" << currentDevice->getInfo<CL_DEVICE_NAME>() <<
			 "\" because it doesn't match the requested device name \"" << deviceTraits.name << "\"\n");

		    currentDevice = candidateDevices.erase(currentDevice);
		    continue;
		}
	    }

	    // Discard devices not matching the requested vendor
	    if(!deviceTraits.vendor.empty()) {
		stringValue = currentDevice->getInfo<CL_DEVICE_VENDOR>();
		if(stringValue.find(deviceTraits.vendor) == std::string::npos) {
		    CERR("Discarding device \"" << currentDevice->getInfo<CL_DEVICE_NAME>() <<
			 "\" because it doesn't match the requested device vendor \"" << deviceTraits.vendor << "\"\n");

		    currentDevice = candidateDevices.erase(currentDevice);
		    continue;
		}
	    }

	    // Discard devices whose version is lower than the requested version
	    if(!deviceTraits.version.empty()) {
		// Get currentDevice's version
		// Returned version format is: "OpenCL <major>.<minor> <vendor_specific_details>"
		// The fixed string "OpenCL " is 7 characters long
		// This value is provided by the device driver, so we should be able to trust it and use atoi
		stringValue = currentDevice->getInfo<CL_DEVICE_VERSION>();
		unsigned pointPos = stringValue.find('.', 7);
		unsigned versionEndPos = stringValue.find(' ', 7);
		unsigned thisDeviceVersionMajor = ::atoi(stringValue.substr(7, pointPos - 7).c_str());
		unsigned thisDeviceVersionMinor = ::atoi(stringValue.substr(pointPos + 1, versionEndPos - pointPos - 1).c_str());

		if((requestedDeviceVersionMajor > thisDeviceVersionMajor) ||
			((requestedDeviceVersionMajor == thisDeviceVersionMajor) && (requestedDeviceVersionMinor > thisDeviceVersionMinor))) {
		    CERR("Discarding device \"" << currentDevice->getInfo<CL_DEVICE_NAME>()
			 << "\" because its version \"" << thisDeviceVersionMajor << '.' << thisDeviceVersionMinor
			 << "\" is older than the requested device version \"" << deviceTraits.version << "\"\n");

		    currentDevice = candidateDevices.erase(currentDevice);
		    continue;
		}
	    }

	    // Discard devices not matching the requested extensions
	    if(!deviceTraits.extensions.empty()) {
		stringValue = currentDevice->getInfo<CL_DEVICE_EXTENSIONS>();
		bool allExtensionsSupported = true;
		std::vector<std::string>::const_iterator currentExtension = deviceTraits.extensions.begin();
		while(currentExtension != deviceTraits.extensions.end()) {
		    if(stringValue.find(*currentExtension) == std::string::npos) {
			CERR("Discarding device \"" << currentDevice->getInfo<CL_DEVICE_NAME>() <<
			     "\" because it doesn't have the requested extension \"" << *currentExtension << "\"\n");

			allExtensionsSupported = false;
			break;
		    }
		}

		if(!allExtensionsSupported) {
		    currentDevice = candidateDevices.erase(currentDevice);
		    continue;
		}
	    }

	    // Discard devices not matching the requested queue properties
	    if(deviceTraits.queueProperties != cl::QueueProperties::None) {
		cl::QueueProperties supportedQueueProperties = static_cast<cl::QueueProperties>(currentDevice->getInfo<CL_DEVICE_QUEUE_PROPERTIES>());
		if((deviceTraits.queueProperties & supportedQueueProperties) != deviceTraits.queueProperties) {
		    CERR("Discarding device \"" << currentDevice->getInfo<CL_DEVICE_NAME>() <<
			 "\" because it doesn't support the requested command queue properties \n");

		    currentDevice = candidateDevices.erase(currentDevice);
		    continue;
		}
	    }

	    ++currentDevice;
	} // while(currentDevice!=candidateDevice.end())

	++currentPlatform;
    } // while(currentPlatform!=candidatePlatforms.end())

    //------------------------------------------------------------------------
    // 3. Decide device to use (and its platform) based on previous filterings
    //------------------------------------------------------------------------

    if(candidatePlatforms.empty())
	BTTHROW(CLError(CL_INVALID_PLATFORM, "None of the existing OpenCL platforms matches requested criteria"), "CLapp::init");

    if(candidateDevices.empty())
	BTTHROW(CLError(CL_INVALID_DEVICE, "None of the existing OpenCL devices matches requested criteria"), "CLapp::init");

    // Sort candidate devices according to their score (descending)
    std::map<long, cl::Device, std::greater<long> > sortedCandidateDevices;
    const auto& constCD = candidateDevices;
    for(auto&& i : constCD)
	sortedCandidateDevices[score(i)] = i;

#ifdef CLAPP_DEBUG
    std::cerr << "Candidate devices are:\n";
    for(auto&& i : sortedCandidateDevices) {
	cl::Platform devicePlatform = cl::Platform(i.second.getInfo<CL_DEVICE_PLATFORM>());
	std::cerr << "    \"" << i.second.getInfo<CL_DEVICE_NAME>() << "\" [score: " << i.first << "] from platform \"" << devicePlatform.getInfo<CL_PLATFORM_NAME>() << "\"\n";
    }
    std::cerr << "\n";
#endif

    //We only manage one device for now. If more than one devices passed all the filters, just choose the first one (which should be the fastest)
    devices.push_back(sortedCandidateDevices.begin()->second);
    platform = devices[0].getInfo<CL_DEVICE_PLATFORM>();

    // Each different combination of platform, platform version, device, device version and driver version must yield a different hash for cached kernels.
    // Precompute the common part here
    auto p = cl::Platform(platform);
    deviceStrings.push_back(p.getInfo<CL_PLATFORM_NAME>() + "/" + p.getInfo<CL_PLATFORM_VERSION>() + "/" + devices[0].getInfo<CL_DEVICE_NAME>() + "/" +
			    devices[0].getInfo<CL_DEVICE_VERSION>() + "/" + devices[0].getInfo<CL_DRIVER_VERSION>() + "/");

    // Try to get PCIe bus id for the chosen device so that we can choose the same device for HIP.
    // There is no C++ wrapper for these calls, so use the C API
    cl_int err;
    enum { busIDStrMaxLength = 32 }; // 32 characters should be more than enough to hold a PCIe busID
    char hipBusID[busIDStrMaxLength] = {0};

    // First try the AMD way
    cl_device_topology_amd topology;
    if((err = clGetDeviceInfo(devices[0](), CL_DEVICE_TOPOLOGY_AMD, sizeof(cl_device_topology_amd), &topology, NULL)) == CL_SUCCESS) {
	if(topology.raw.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD)
	    snprintf(hipBusID, busIDStrMaxLength, "0000:%02x:%02x.%1x", topology.pcie.bus, topology.pcie.device, topology.pcie.function);
	else
	    CLAPP_CERR("AMD call to get PCIe busID reports topology is not PCIe\n");
    }

    // Else try the nVidia way
    else {
	CLAPP_CERR("AMD call to get PCIe busID failed: " << getOpenCLErrorCodeStr(err) << '\n');
	cl_uint nvBus;
	cl_uint nvSlot;
	if((err = clGetDeviceInfo(devices[0](), CL_DEVICE_PCI_BUS_ID_NV, sizeof(cl_uint), &nvBus, NULL)) == CL_SUCCESS) {
	    if((err = clGetDeviceInfo(devices[0](), CL_DEVICE_PCI_SLOT_ID_NV, sizeof(cl_uint), &nvSlot, NULL)) == CL_SUCCESS)
		snprintf(hipBusID, busIDStrMaxLength, "0000:%02x:%02x.0", nvBus, nvSlot);
	    else
		CLAPP_CERR("NV call to get PCIe slot failed: " << getOpenCLErrorCodeStr(err) << '\n');
	}
	else
	    CLAPP_CERR("NV call to get PCIe bus failed: " << getOpenCLErrorCodeStr(err) << '\n');
    }

#ifdef HAVE_HIP
    if(deviceTraits.useHIP) {
	if(hipBusID[0] != 0) {
	    hipError_t hipErr = hipDeviceGetByPCIBusId(&hipDevice, hipBusID);
	    if(hipErr != hipSuccess) {
		hipDevice = -1;
		CERR("Error trying to get HIP device " << hipBusID << '\n');
	    }
	}
	else {
	    hipDevice = -1;
	    CERR("Couldn't find PCIe busID for chosen device\n");
	}
    }
    else
	hipDevice = -1;
#endif

    // Show information about chosen device/platform if requested by the caller
    if(!quiet) {
	std::cerr << "Chosen platform: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n' <<
		  "    OpenCL version: " << platform.getInfo<CL_PLATFORM_VERSION>() << "\n\n";

	std::cerr << "Chosen device: " << devices[0].getInfo<CL_DEVICE_NAME>() << '\n' <<
		  "    OpenCL version: " << devices[0].getInfo<CL_DEVICE_VERSION>() << '\n' <<
		  "    OpenCL C version: " << devices[0].getInfo<CL_DEVICE_OPENCL_C_VERSION>() << '\n' <<
		  "    PCIe busID: " << hipBusID << '\n' <<
		  "    HIP is ";
#ifdef HAVE_HIP
	if(deviceTraits.useHIP) {
	    if(hipDevice == -1) std::cerr << "UN";
	    std::cerr << "AVAILABLE for this device\n\n";
	}
	else
	    std::cerr << "NOT ENABLED in command line\n\n";
#else
	std::cerr << "DISABLED at compile time\n\n";
#endif
    }


    //----------------------------------------------------------------------------------------
    // 4. Create a CL context for all the selected devices and a command queue in each of them
    //----------------------------------------------------------------------------------------
    context = cl::Context(devices); //,nullptr,nullptr,nullptr,&err);

    const auto& constDevices = devices;
    for(auto&& i : constDevices)
	commandQueues.push_back(cl::CommandQueue(context, i, deviceTraits.queueProperties));
}


/**
* @brief Gets kernel (type cl::Kernel) from list of kernels
* @param[in] i index of the kernel in the kernels list
* @return reference to selected kernel
*/
cl::Kernel& CLapp::getKernel(const size_t i) {
    if(!kernelsLoaded) {
	CERR("Automatically loading kernels at first getKernel() call\n");
	loadKernels();
    }

    KernelList::iterator j(kernels.begin());
    std::advance(j, i);
    if(j != kernels.end())
	return j->second.kernel;
    else {
	std::ostringstream s;
	s << "Kernel #" << i << " does not exist";
	BTTHROW(CLError(CL_INVALID_KERNEL_NAME, s.str()), "CLapp::getKernel");
    }

}

/**
 * @brief Gets kernel (type cl::Kernel) from name
 * @param[in] name name of the kernel
 * @return reference to selected kernel
 */
cl::Kernel&
CLapp::getKernel(const std::string& name) {
    if(!kernelsLoaded) {
	CERR("Automatically loading kernels at first getKernel() call\n");
	loadKernels();
    }

    KernelList::iterator j(kernels.find(name));
    if(j != kernels.end())
	return j->second.kernel;
    else {
	std::ostringstream s;
	s << "Kernel \"" << name << "\" does not exist";
	BTTHROW(CLError(CL_INVALID_KERNEL_NAME, s.str()), "CLapp::getKernel");
    }
}

/** @brief Adds a kernel file to be loaded when loadKernels() is called
 * @param[in] sourceFile name of the kernel file to be added
 */
void CLapp::addKernelFile(const std::string& sourceFile) {
    // Processes without a kernel will report an empty string as their kernel file
    if(!sourceFile.empty()) {
        if(!kernelFiles.count(sourceFile)) {
            kernelFiles.insert({sourceFile, KernelFileProperties()});

            if(kernelsLoaded) {
                std::cerr << "Warning: forcing extra kernel load due to new kernel file [" << sourceFile << "] added after loadKernels(). This will stall the queue!\n";
                loadKernels();
            }
        }
        else
            CERR("addKernelFile: not adding already added kernel file [" << sourceFile << "]\n");
    }
}

/** @brief Adds a directory to the kernel directory search list
 * @param[in] kernelDir name of the directory to be added to the list
 */
void CLapp::addKernelDir(const std::string& kernelDir) {
    kernelDirs.insert(kernelDir);
}

/**
 * @brief Loads kernels for currently existing processes
 * @param[in] compilerOptionsArg text string with compiler options
 */
void CLapp::loadKernels(const char* compilerOptionsArg) {
    // Measure kernel load/compilation times starting now
    auto startTime = std::chrono::high_resolution_clock::now();

    // Always include internal kernels (e.g. for show() and any other internal usage) and host/kernel functions (note: do not use addKernelFile from here, as that would cause
    // an infinite loop)
    if(!kernelFiles.count(INTERNAL_KERNELS_FILE))
        kernelFiles.insert({INTERNAL_KERNELS_FILE, KernelFileProperties()});

    // Do not waste time on already loaded kernel files. Build a list with pending files only
    // and return immediately if no kernels left to load
    KernelFileList pendingKernelFiles;
    for(auto&& i: kernelFiles) {
        if(!i.second.loaded)
	    pendingKernelFiles[i.first] = i.second;
	else {
            CLAPP_CERR("loadKernels: not loading already loaded kernel file [" << i.first << "]\n");
            continue;
        }
    }
    if(pendingKernelFiles.empty()) {
	CLAPP_CERR("loadKernels: all kernels already loaded. Nothing to do!\n");
	return;
    }

    // Add include path to compiler options
    std::string compilerOptions;
    compilerOptions.append("-I " KERNEL_INCLUDE_DIR);

#ifdef NDEBUG
    // Add fast math option to compiler options in release mode. Note: this enables optimizations that are unsafe if math arguments and results
    // are not valid (e.g. inf or nan)
    compilerOptions.append(" -cl-fast-relaxed-math");
#endif
    if(compilerOptionsArg != nullptr) {
	// A space must be added to separate options
	compilerOptions.append(" ");
	compilerOptions.append(compilerOptionsArg);
    }

    // The implementation of common header files must be compiled together with every kernel file given by the user.
    // Add any such files here (for now, we only need the host/kernel functions file)
    std::vector<std::string> extraSourceFiles;
    extraSourceFiles.push_back(HOST_KERNEL_FUNCTIONS_FILE);


    /////////////////////////////////////////////////////////////
    // Build a list of directories where to look for source files
    /////////////////////////////////////////////////////////////

    // Kernel directory must be trusted or we'll be open to arbitrary code execution. Trust only:
    // a) absolute pathnames
    // b) directories specified by the programmer via addKernelDir()
    // c) the user's kernel directory ($HOME/KERNEL_USER_DIR)
    // d) the compile-time configured kernel directory
    KernelPathList kernelPaths;
    kernelPaths.clear();

    // Add dirs specified by the programmer
    for(auto&& i: kernelDirs)
	kernelPaths.insert(i + "/");

    // Add user's kernel dir (if $HOME exists)
    char* home = getenv("HOME");
    if(home)
	kernelPaths.insert(std::string(home) + "/" KERNEL_USER_DIR "/");
    else
	std::cerr<<"No $HOME environment variable. Can't use kernel cache\n";

    // Add compile-time kernel dir
    kernelPaths.insert(KERNEL_SOURCE_DIR "/");

    //////////////////////////////////////////////////////////////
    // Look for extra source files and store their full path names
    //////////////////////////////////////////////////////////////
    time_t extraFilesTimestamp = 0;
    bool allExtraFilesFound = true;

    for(auto&& file: extraSourceFiles) {
	struct stat statBuf;
	bool thisFileFound = false;

	// If we are given a full pathname, we don't have to iterate through a list of possible locations
	if(file[0] == '/') {
	    if(::stat(file.c_str(), &statBuf) == 0) {
		thisFileFound = true;

		// Set extra files' timestamp to that of the most recent file
		if(statBuf.st_mtim.tv_sec > extraFilesTimestamp)
		    extraFilesTimestamp = statBuf.st_mtim.tv_sec;
	    }
	}

	// If file is a bare file name, iterate through all possible locations
	else {
	    for(auto&& dir: kernelPaths) {
		std::string path = dir + file;

		if(::stat(path.c_str(), &statBuf) == 0) {
		    thisFileFound = true;

		    // Substitute bare file name with its full path
		    file = path;

		    // Set extra files' timestamp to that of the most recent file
		    if(statBuf.st_mtim.tv_sec > extraFilesTimestamp)
			extraFilesTimestamp = statBuf.st_mtim.tv_sec;
		}
	    }
	}

	if(!thisFileFound) {
	    allExtraFilesFound = false;

#ifdef CLAPP_DEBUG
	    std::ostringstream s;
	    s << "Couldn't find extra source file [" << file << "]\n";
	    s << "Paths tried: [";
	    auto j = kernelPaths.begin();
	    s << *j;
	    ++j;
	    while (j != kernelPaths.end()) {
		s << ',' << *j;
		j++;
	    }
	    s << "]\n";
	    std::cerr << s.str();
#endif
	}
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    // Iterate through kernel files given by the user.
    // Use a cached version if available and up to date; compile and store in cache otherwise
    /////////////////////////////////////////////////////////////////////////////////////////
    cl::Program::Sources sources;
    bool extraSourceFilesLoaded = false;
    size_t totalLoadedKernels = 0;

    // Some CL compilers (read: AMD) generate _s_l_o_w_ code if all sources are compiled together.
    // Whatever the reason, let's compile each source file on its own (together with host/kernel functions)
    for(auto&& currentFile: pendingKernelFiles) {
	std::unique_ptr<cl::Program> program;
	std::string cacheFile, cacheSubdir;

        ///////////////////////////////////////////////////////////
        // Look for a previously cached version of this kernel file
        ///////////////////////////////////////////////////////////
        bool cacheFileFound = false;
	time_t cacheFileTimestamp = 0;
	size_t cacheFileSize;

	if(home) {
	    // Compose file name for the cached version of this kernel file
	    std::stringstream s;
	    s << std::hex << std::hash<std::string>{}(deviceStrings[0] + currentFile.first);
	    cacheSubdir = std::string(KERNEL_USER_DIR "/cache/") + s.str()[0] + "/" + s.str()[1];
	    cacheFile = std::string(home) + "/" + cacheSubdir + "/" + s.str().substr(2);

	    // Get timestamp of cached kernel file (or 0 otherwise)
	    struct stat statBuf;
	    if(::stat(cacheFile.c_str(), &statBuf) == 0) {
		cacheFileFound = true;
		cacheFileTimestamp = statBuf.st_mtim.tv_sec;
		cacheFileSize = statBuf.st_size;
	    }
	    else
		CLAPP_CERR("Cached version of kernel file " << currentFile.first << " (" << cacheFile << ") not found\n");
	}


        ///////////////////////////////////////////////////////////
        // Look for the kernel file itself (i.e. source code)
        ///////////////////////////////////////////////////////////
	bool sourceFileFound = false;
	time_t sourceFileTimestamp = 0;
	struct stat statBuf;

	// If we are given a full pathname, we don't have to iterate through a list of possible locations
	if(currentFile.first[0] == '/') {
	    if(::stat(currentFile.first.c_str(), &statBuf) == 0) {
		sourceFileFound = true;
		sourceFileTimestamp = statBuf.st_mtim.tv_sec;

		// Store full path of found kernel file
		currentFile.second.sourcePath = currentFile.first;
	    }
	    else {
                CLAPP_CERR("Couldn't find source file [" << currentFile.first << "]\n");
	    }
	}

	// If file is a bare file name, iterate through all possible locations
	else {
	    for(auto dir = kernelPaths.begin(); dir != kernelPaths.end() && !sourceFileFound; dir++) {
		std::string path = *dir + currentFile.first;

		int err;
		if((err=::stat(path.c_str(), &statBuf)) == 0) {
		    sourceFileFound = true;
		    sourceFileTimestamp = statBuf.st_mtim.tv_sec;

		    // Store full path of found kernel file
		    currentFile.second.sourcePath = path;
		}
	    }
#ifdef CLAPP_DEBUG
	    if(!sourceFileFound) {
		std::ostringstream s;
		s << "Couldn't find source file [" << currentFile.first << "]\n";
		s << "Paths tried: [";
		auto j = kernelPaths.begin();
		s << *j;
		++j;
		while (j != kernelPaths.end()) {
		    s << ',' << *j;
		    j++;
		}
		s << "]\n";
		std::cerr << s.str();
	    }
#endif
	}

        /////////////////////////////////////////////////////////////////
        // Decide between loading cached version or compiling from source
        /////////////////////////////////////////////////////////////////
	if(cacheFileFound && (cacheFileTimestamp >= sourceFileTimestamp) && (cacheFileTimestamp >= extraFilesTimestamp)) {
	    /////////////////////////////////////////////////////////////////
	    // Load cached version
	    /////////////////////////////////////////////////////////////////
	    cl::Program::Binaries binaries;
	    std::ifstream f;

	    f.open(cacheFile, std::ios::in | std::ios::binary);
	    if(f.is_open()) {
		// cl::Program::Binaries is a vector<vector<unsigned char>>
		auto buffer = new std::vector<unsigned char>(cacheFileSize);
		f.read((char*) buffer->data(), cacheFileSize);
		f.close();

		binaries.clear();
		binaries.push_back(*buffer);
		delete(buffer);
	    }
	    else {
		std::ostringstream s;
		s << "Couldn't open cached kernel file [" << cacheFile << "]\n";
		BTTHROW(CLError(CL_BUILD_PROGRAM_FAILURE, s.str()), "CLapp::loadKernels");
	    }

	    // Create a program from cached binaries
	    program = std::unique_ptr<cl::Program>(new cl::Program(context, devices, binaries));

            /////////////////////////////////////////////////////////////////
	    // Build program from cached binaries
	    /////////////////////////////////////////////////////////////////
#ifndef NDEBUG
	    {
		auto i = devices.begin();
		std::cerr << "Building CL program for devices [" << i->getInfo<CL_DEVICE_NAME>();
		++i;
		while(i != devices.end()) {
		    std::cerr << ',' << i->getInfo<CL_DEVICE_NAME>();
		    ++i;
		}
		std::cerr << "] from cached binaries\n";
	    }
#endif

	    try {
		//Warning: AMD CL compiler may crash if using CL2.0 features in CL1.x compiler mode!
		//Don't forget to pass -cl-std=CL2.0 in compilerOptions if using CL2.0 features.
		program->build(devices, compilerOptions.c_str());

#ifndef NDEBUG
		// Always show compilation log in debug mode
		{
		    std::string buildLog;
		    for(auto&& i: devices) {
			buildLog = program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(i);
			std::cerr << "Build log for device " << i.getInfo<CL_DEVICE_NAME>() << ":";
			if(!buildLog.empty()) {
			    std::cerr << "\n----------------------------------------------------------------------\n"
				    << buildLog << '\n'
				    << "----------------------------------------------------------------------\n"
				    << "\n\n";
			}
			else
			    std::cerr << " <empty>\n";
		    }
		}
#endif
	    }
	    catch(cl::BuildError& err) {
		// Do we still need to do anything useful here?
		throw;
	    }
	}
	else if(sourceFileFound && allExtraFilesFound) {
	    /////////////////////////////////////////////////////////////////
	    // Compile from source and store cached version
	    // Each compilation unit is made of the user specified kernel files plus all common extra files
	    /////////////////////////////////////////////////////////////////
	    std::ifstream f;
	    std::ostringstream currentFileBuffer;

	    // Load extra source files (if not already loaded) and store them in sources
	    if(!extraSourceFilesLoaded) {
		std::ostringstream extraFilesBuffer;
		for(auto&& i: extraSourceFiles) {
		    f.open(i.c_str(), std::ios::in | std::ios::binary);
		    if(f.is_open()) {
			extraFilesBuffer.str() = "";
			extraFilesBuffer << f.rdbuf();
			sources.push_back(extraFilesBuffer.str());
			f.close();
		    }
		    else {
			std::ostringstream s;
			s << "Couldn't open needed source file [" << i.c_str() << "]\n";
			BTTHROW(CLError(CL_BUILD_PROGRAM_FAILURE, s.str()), "CLapp::loadKernels");
		    }
		}
		extraSourceFilesLoaded = true;
	    }

	    // Load current kernel file and append it to sources
	    f.open(currentFile.second.sourcePath, std::ios::in | std::ios::binary);
	    if(f.is_open()) {
		currentFileBuffer.str() = "";
		currentFileBuffer << f.rdbuf();
		sources.push_back(currentFileBuffer.str());
		f.close();
	    }
	    else {
		std::ostringstream s;
		s << "Couldn't open needed source file [" << currentFile.first << "]\n";
		BTTHROW(CLError(CL_BUILD_PROGRAM_FAILURE, s.str()), "CLapp::loadKernels");
	    }

	    // Create a program from kernel file + extra sources
	    program = std::unique_ptr<cl::Program>(new cl::Program(context, sources));

	    // Pop kernel file away from sources, so only extra sources remain
	    sources.pop_back();

            /////////////////////////////////////////////////////////////////
	    // Build program from sources
	    /////////////////////////////////////////////////////////////////
#ifndef NDEBUG
	    {
		auto i = devices.begin();
		std::cerr << "Building CL program for devices [" << i->getInfo<CL_DEVICE_NAME>();
		++i;
		while(i != devices.end()) {
		    std::cerr << ',' << i->getInfo<CL_DEVICE_NAME>();
		    ++i;
		}
		std::cerr << "]...\n";
	    }

	    {
		std::cerr << " from source file(s) [";
		for(auto&& i: extraSourceFiles)
		    std::cerr << i << ',';
		std::cerr << currentFile.second.sourcePath << "]\n";
	    }
#endif

	    try {
		//Warning: AMD CL compiler may crash if using CL2.0 features in CL1.x compiler mode!
		//Don't forget to pass -cl-std=CL2.0 in compilerOptions if using CL2.0 features.
		program->build(devices, compilerOptions.c_str());

#ifndef NDEBUG
		// Always show compilation log in debug mode
		{
		    std::string buildLog;
		    for(auto&& i: devices) {
			buildLog = program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(i);
			std::cerr << "Build log for device " << i.getInfo<CL_DEVICE_NAME>() << ":";
			if(!buildLog.empty()) {
			    std::cerr << "\n----------------------------------------------------------------------\n"
				    << buildLog << '\n'
				    << "----------------------------------------------------------------------\n"
				    << "\n\n";
			}
			else
			    std::cerr << " <empty>\n";
		    }
		}
#endif
	    }
	    catch(cl::BuildError& err) {
		// Do we still need to do anything useful here?
		throw;
	    }

            /////////////////////////////////////////////////////////////////
	    // Save built program in the cache
	    /////////////////////////////////////////////////////////////////
	    if(home) {
		// Check for existent cache directories and create them if necessary
		struct stat statBuf;
		int err = 0;
		unsigned slashPos = 0;
		while(slashPos < cacheSubdir.size()) {
		    slashPos = cacheSubdir.find('/', slashPos + 1);
		    auto dir = std::string(home) + "/" + cacheSubdir.substr(0, slashPos);

		    if((::stat(dir.c_str(), &statBuf)) == -1) {
			if(errno == ENOENT)
			    err |= ::mkdir(dir.c_str(), 0755);
			else
			    err |= 1;	// If ::stat returned any other error than ENOENT, force a nonzero err so we show the appropriate error message (below)
		    }
		}

		// Get program binaries and write them to a file
		if(err == 0) {
		    // A program was built for every requested device. compiledPrograms is a vector of char* (one pointer per device)
		    auto compiledPrograms = program->getInfo<CL_PROGRAM_BINARIES>();
		    auto compiledProgramsLen = program->getInfo<CL_PROGRAM_BINARY_SIZES>();

		    // We only deal with one device for now. Save only first returned binary
		    std::ofstream f;
			f.open(cacheFile.c_str(), std::ios::out | std::ios::binary);
			if(f.is_open()) {
			    f.write(reinterpret_cast<char*>(compiledPrograms[0].data()), compiledProgramsLen[0]);
			    f.close();
			    CLAPP_CERR("Wrote kernel cache file " << cacheFile << "\n");
			}
			else {
			    std::cerr << "Couldn't write cache file " << cacheFile << "\n";
			}
		}
		else
		    std::cerr << "Error creating cache directories. Not generating kernel cache files\n";
	    }
	    else {
		// The user has already been warned if the HOME environment variable does not exist
	    }
	}
	else {
	    std::ostringstream s;
	    s << "Couldn't find needed source file [" << currentFile.first << "] (or any dependencies) or a valid cached version\n";
	    BTTHROW(CLError(CL_BUILD_PROGRAM_FAILURE, s.str()), "CLapp::loadKernels");
	}


	/////////////////////////////////////////////////////////////////
	// Create kernels from built program
	/////////////////////////////////////////////////////////////////
        std::vector<cl::Kernel> programKernels;
        program->createKernels(&programKernels);
        //programs.push_back(program);

        // Only add compiled kernels to our list if they don't exist already (warn the user otherwise)
        std::set<std::string> loadedKernels;
        loadedKernels.clear();
        for(auto&& i: programKernels) {
            std::string kernelName;
            i.getInfo(CL_KERNEL_FUNCTION_NAME, &kernelName);
            if(!kernels.count(kernelName)) {
                kernels[kernelName].kernel = i;
                loadedKernels.insert(kernelName);
		++totalLoadedKernels;
            }
            else
                CLAPP_CERR("Warning: a kernel named \"" << kernelName << "\" already exists! Will not overwrite it\n");
        }

#ifndef NDEBUG
	if(!loadedKernels.empty()) {
	    auto i = loadedKernels.begin();
	    CLAPP_CERR("Loaded and built kernels [" << *i);
	    ++i;
	    while(i != loadedKernels.end()) {
		CLAPP_CERR(',' << *i);
		++i;
	    }
	    CLAPP_CERR("]\n");
	}
	else
	    CLAPP_CERR("No kernels loaded\n");
#endif

        // Don't recompile this source file in subsequent calls to loadKernels
        kernelFiles[currentFile.first].loaded = true;
    }

    // After first execution of loadKernels, subsequent calls to addKernelFile will trigger a warning about possible queue stalls
    kernelsLoaded = true;

    // Measure kernel load/compilation times ends now
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsedTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()) / 1e9;
    std::cerr<<"Loaded "<< totalLoadedKernels <<" kernels in "<< elapsedTime <<" sec\n";

}

/**
 * @brief Shows info about OpenCL platforms and devices on standard output
 */
int CLapp::dumpInfo(void) {
    std::vector<cl::Platform> platforms;
    try {
	cl::Platform::get(&platforms);
    }
    catch(cl::Error& e) {
	if(e.err() == CL_PLATFORM_NOT_FOUND_KHR) {
	    std::cout << "Got 0 OpenCL platform(s). Nothing to show!\n\n";
	    return -1;
	}
	else throw CLError(e);
    }

    std::string stringValue;
    for(unsigned int i = 0; i < platforms.size(); i++) {
	std::cout << "Platform " << i << '\n';

	platforms[i].getInfo(CL_PLATFORM_PROFILE, &stringValue);
	std::cout << "    CL_PLATFORM_PROFILE=" << stringValue << '\n';
	platforms[i].getInfo(CL_PLATFORM_NAME, &stringValue);
	std::cout << "    CL_PLATFORM_NAME=" << stringValue << '\n';
	platforms[i].getInfo(CL_PLATFORM_VENDOR, &stringValue);
	std::cout << "    CL_PLATFORM_VENDOR=" << stringValue << '\n';
	platforms[i].getInfo(CL_PLATFORM_VERSION, &stringValue);
	std::cout << "    CL_PLATFORM_VERSION=" << stringValue << '\n';

	//Store OpenCL platform version
	std::string::size_type pos = stringValue.find(' ');
	std::string platformVersion = stringValue.substr(pos + 1,
				      stringValue.find(' ', pos + 1) - pos - 1);

	platforms[i].getInfo(CL_PLATFORM_EXTENSIONS, &stringValue);
	std::cout << "    CL_PLATFORM_EXTENSIONS=" << stringValue << '\n';
	std::cout << '\n';

	std::vector<cl::Device> devices;
	try {
	    platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	}
	catch(cl::Error& e) {
	    if(e.err() == CL_DEVICE_NOT_FOUND) {
		std::cout << "    Got 0 devices for platform " << i << "\n\n";
		continue;
	    }
	    else throw CLError(e);
	}
	std::cout << "    Got " << devices.size() << " device(s) for platform "
		  << i << "\n\n";

	for(unsigned int j = 0; j < devices.size(); j++) {
	    std::cout << "    Device " << j << '\n';

	    cl_device_type devType;
	    devices[j].getInfo(CL_DEVICE_TYPE, &devType);
	    std::cout << "        CL_DEVICE_TYPE=";
	    if(devType & CL_DEVICE_TYPE_DEFAULT)
		std::cout << "CL_DEVICE_TYPE_DEFAULT ";
	    if(devType & CL_DEVICE_TYPE_CPU)
		std::cout << "CL_DEVICE_TYPE_CPU ";
	    if(devType & CL_DEVICE_TYPE_GPU)
		std::cout << "CL_DEVICE_TYPE_GPU ";
	    if(devType & CL_DEVICE_TYPE_ACCELERATOR)
		std::cout << "CL_DEVICE_TYPE_ACCELERATOR ";
#ifdef CL_VERSION_1_2
	    if(devType & CL_DEVICE_TYPE_CUSTOM)
		std::cout << "CL_DEVICE_TYPE_CUSTOM ";
#endif //CL_VERSION_1_2
	    std::cout << '\n';

	    cl_uint cluintValue;
	    devices[j].getInfo(CL_DEVICE_VENDOR_ID, &cluintValue);
	    std::cout << "        CL_DEVICE_VENDOR_ID=0x" << std::hex
		      << cluintValue << std::dec << '\n';

	    devices[j].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &cluintValue);
	    std::cout << "        CL_DEVICE_MAX_COMPUTE_UNITS=" << cluintValue
		      << '\n';

	    unsigned int uintValue;
	    devices[j].getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &uintValue);
	    std::cout << "        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS="
		      << uintValue << '\n';

	    std::vector<size_t> sizetVector;
	    devices[j].getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &sizetVector);
	    std::cout << "        CL_DEVICE_MAX_WORK_ITEM_SIZES=";
	    for(unsigned int k = 0; k < uintValue; k++)
		std::cout << sizetVector[k] << ' ';
	    std::cout << '\n';

	    size_t sizetValue;
	    devices[j].getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &sizetValue);
	    std::cout << "        CL_DEVICE_MAX_WORK_GROUP_SIZE=" << sizetValue
		      << '\n';

	    devices[j].getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, &cluintValue);
	    std::cout << "        CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR="
		      << cluintValue << '\n';
	    devices[j].getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, &cluintValue);
	    std::cout << "        CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT="
		      << cluintValue << '\n';
	    devices[j].getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, &cluintValue);
	    std::cout << "        CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT="
		      << cluintValue << '\n';
	    devices[j].getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, &cluintValue);
	    std::cout << "        CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG="
		      << cluintValue << '\n';
	    devices[j].getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, &cluintValue);
	    std::cout << "        CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT="
		      << cluintValue << '\n';
	    devices[j].getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, &cluintValue);
	    std::cout << "        CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE="
		      << cluintValue << '\n';
	    devices[j].getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, &cluintValue);
	    std::cout << "        CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF="
		      << cluintValue << '\n';

	    devices[j].getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, &cluintValue);
	    std::cout << "        CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR="
		      << cluintValue << '\n';
	    devices[j].getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, &cluintValue);
	    std::cout << "        CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT="
		      << cluintValue << '\n';
	    devices[j].getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, &cluintValue);
	    std::cout << "        CL_DEVICE_NATIVE_VECTOR_WIDTH_INT="
		      << cluintValue << '\n';
	    devices[j].getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, &cluintValue);
	    std::cout << "        CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG="
		      << cluintValue << '\n';
	    devices[j].getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, &cluintValue);
	    std::cout << "        CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT="
		      << cluintValue << '\n';
	    devices[j].getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, &cluintValue);
	    std::cout << "        CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE="
		      << cluintValue << '\n';
	    devices[j].getInfo(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, &cluintValue);
	    std::cout << "        CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF="
		      << cluintValue << '\n';

	    devices[j].getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &cluintValue);
	    std::cout << "        CL_DEVICE_MAX_CLOCK_FREQUENCY=" << cluintValue
		      << " MHz\n";

	    devices[j].getInfo(CL_DEVICE_ADDRESS_BITS, &cluintValue);
	    std::cout << "        CL_DEVICE_ADDRESS_BITS=" << cluintValue
		      << '\n';

	    cl_ulong clulongValue;
	    devices[j].getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &clulongValue);
	    std::cout << "        CL_DEVICE_MAX_MEM_ALLOC_SIZE=" << clulongValue
		      << " bytes\n";

	    cl_bool clboolValue;
	    devices[j].getInfo(CL_DEVICE_IMAGE_SUPPORT, &clboolValue);
	    std::cout << "        CL_DEVICE_IMAGE_SUPPORT=" << clboolValue
		      << '\n';

	    devices[j].getInfo(CL_DEVICE_MAX_READ_IMAGE_ARGS, &cluintValue);
	    std::cout << "        CL_DEVICE_=MAX_READ_IMAGE_ARGS="
		      << cluintValue << '\n';

	    devices[j].getInfo(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, &cluintValue);
	    std::cout << "        CL_DEVICE_=MAX_WRITE_IMAGE_ARGS="
		      << cluintValue << '\n';

	    devices[j].getInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH, &sizetValue);
	    std::cout << "        CL_DEVICE_IMAGE2D_MAX_WIDTH=" << sizetValue
		      << '\n';

	    devices[j].getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &sizetValue);
	    std::cout << "        CL_DEVICE_IMAGE2D_MAX_HEIGHT=" << sizetValue
		      << '\n';

	    devices[j].getInfo(CL_DEVICE_IMAGE3D_MAX_WIDTH, &sizetValue);
	    std::cout << "        CL_DEVICE_IMAGE3D_MAX_WIDTH=" << sizetValue
		      << '\n';

	    devices[j].getInfo(CL_DEVICE_IMAGE3D_MAX_HEIGHT, &sizetValue);
	    std::cout << "        CL_DEVICE_IMAGE3D_MAX_HEIGHT=" << sizetValue
		      << '\n';

	    devices[j].getInfo(CL_DEVICE_IMAGE3D_MAX_DEPTH, &sizetValue);
	    std::cout << "        CL_DEVICE_IMAGE3D_MAX_DEPTH=" << sizetValue
		      << '\n';

	    devices[j].getInfo(CL_DEVICE_MAX_SAMPLERS, &sizetValue);
	    std::cout << "        CL_DEVICE_MAX_SAMPLERS=" << sizetValue
		      << '\n';

	    devices[j].getInfo(CL_DEVICE_MAX_PARAMETER_SIZE, &sizetValue);
	    std::cout << "        CL_DEVICE_MAX_PARAMETER_SIZE=" << sizetValue
		      << '\n';

	    devices[j].getInfo(CL_DEVICE_MEM_BASE_ADDR_ALIGN, &cluintValue);
	    std::cout << "        CL_DEVICE_MEM_BASE_ADDR_ALIGN=" << cluintValue
		      << '\n';

	    cl_device_fp_config cldevicefpconfigValue;
	    devices[j].getInfo(CL_DEVICE_SINGLE_FP_CONFIG, &cldevicefpconfigValue);
	    std::cout << "        CL_DEVICE_SINGLE_FP_CONFIG=";
	    if(cldevicefpconfigValue & CL_FP_DENORM)
		std::cout << "CL_FP_DENORM ";
	    if(cldevicefpconfigValue & CL_FP_INF_NAN)
		std::cout << "CL_FP_INF_NAN ";
	    if(cldevicefpconfigValue & CL_FP_ROUND_TO_NEAREST)
		std::cout << "CL_FP_ROUND_TO_NEAREST ";
	    if(cldevicefpconfigValue & CL_FP_ROUND_TO_ZERO)
		std::cout << "CL_FP_ROUND_TO_ZERO ";
	    if(cldevicefpconfigValue & CL_FP_ROUND_TO_INF)
		std::cout << "CL_FP_ROUND_TO_INF ";
	    if(cldevicefpconfigValue & CL_FP_FMA)
		std::cout << "CL_FP_FMA ";
#ifdef CL_VERSION_1_2
	    if(cldevicefpconfigValue & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)
		std::cout << "CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT ";
#endif //CL_VERSION_1_2
	    if(cldevicefpconfigValue & CL_FP_SOFT_FLOAT)
		std::cout << "CL_FP_SOFT_FLOAT ";
	    std::cout << '\n';

	    devices[j].getInfo(CL_DEVICE_DOUBLE_FP_CONFIG, &cldevicefpconfigValue);
	    std::cout << "        CL_DEVICE_DOUBLE_FP_CONFIG=";
	    if(cldevicefpconfigValue & CL_FP_DENORM)
		std::cout << "CL_FP_DENORM ";
	    if(cldevicefpconfigValue & CL_FP_INF_NAN)
		std::cout << "CL_FP_INF_NAN ";
	    if(cldevicefpconfigValue & CL_FP_ROUND_TO_NEAREST)
		std::cout << "CL_FP_ROUND_TO_NEAREST ";
	    if(cldevicefpconfigValue & CL_FP_ROUND_TO_ZERO)
		std::cout << "CL_FP_ROUND_TO_ZERO ";
	    if(cldevicefpconfigValue & CL_FP_ROUND_TO_INF)
		std::cout << "CL_FP_ROUND_TO_INF ";
	    if(cldevicefpconfigValue & CL_FP_FMA)
		std::cout << "CL_FP_FMA ";
#ifdef CL_VERSION_1_2
	    if(cldevicefpconfigValue & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)
		std::cout << "CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT ";
#endif //CL_VERSION_1_2
	    if(cldevicefpconfigValue & CL_FP_SOFT_FLOAT)
		std::cout << "CL_FP_SOFT_FLOAT ";
	    std::cout << '\n';

	    cl_device_mem_cache_type cldevicememcachetypeValue;
	    devices[j].getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, &cldevicememcachetypeValue);
	    std::cout << "        CL_DEVICE_GLOBAL_MEM_CACHE_TYPE=";
	    switch(cldevicememcachetypeValue) {
		case CL_NONE:
		    std::cout << "CL_NONE";
		    break;
		case CL_READ_ONLY_CACHE:
		    std::cout << "CL_READ_ONLY_CACHE";
		    break;
		case CL_READ_WRITE_CACHE:
		    std::cout << "CL_READ_WRITE_CACHE";
		    break;
		default:
		    std::cout << "[unknown (" << cldevicememcachetypeValue << ")]";
	    }
	    std::cout << '\n';

	    devices[j].getInfo(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, &cluintValue);
	    std::cout << "        CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE="
		      << cluintValue << " bytes\n";

	    devices[j].getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &clulongValue);
	    std::cout << "        CL_DEVICE_GLOBAL_MEM_CACHE_SIZE="
		      << clulongValue << " bytes\n";

	    devices[j].getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &clulongValue);
	    std::cout << "        CL_DEVICE_GLOBAL_MEM_SIZE=" << clulongValue
		      << " bytes\n";

	    devices[j].getInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
			       &clulongValue);
	    std::cout << "        CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE="
		      << clulongValue << " bytes\n";

	    devices[j].getInfo(CL_DEVICE_MAX_CONSTANT_ARGS, &cluintValue);
	    std::cout << "        CL_DEVICE_MAX_CONSTANT_ARGS=" << cluintValue
		      << '\n';

	    cl_device_local_mem_type cldevicelocalmemtypeValue;
	    devices[j].getInfo(CL_DEVICE_LOCAL_MEM_TYPE, &cldevicelocalmemtypeValue);
	    std::cout << "        CL_DEVICE_LOCAL_MEM_TYPE=";
	    switch(cldevicelocalmemtypeValue) {
		case CL_NONE:
		    std::cout << "CL_NONE";
		    break;
		case CL_LOCAL:
		    std::cout << "CL_LOCAL";
		    break;
		case CL_GLOBAL:
		    std::cout << "CL_GLOBAL";
		    break;
		default:
		    std::cout << "[unknown (" << cldevicelocalmemtypeValue << ")]";
		    break;
	    }
	    std::cout << '\n';

	    devices[j].getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &clulongValue);
	    std::cout << "        CL_DEVICE_LOCAL_MEM_SIZE=" << clulongValue
		      << " bytes\n";

	    devices[j].getInfo(CL_DEVICE_ERROR_CORRECTION_SUPPORT,
			       &clboolValue);
	    std::cout << "        CL_DEVICE_ERROR_CORRECTION_SUPPORT="
		      << clboolValue << '\n';

	    devices[j].getInfo(CL_DEVICE_HOST_UNIFIED_MEMORY, &clboolValue);
	    std::cout << "        CL_DEVICE_HOST_UNIFIED_MEMORY=" << clboolValue << '\n';

	    devices[j].getInfo(CL_DEVICE_PROFILING_TIMER_RESOLUTION, &sizetValue);
	    std::cout << "        CL_DEVICE_PROFILING_TIMER_RESOLUTION="
		      << sizetValue << " nanoseconds\n";

	    devices[j].getInfo(CL_DEVICE_ENDIAN_LITTLE, &clboolValue);
	    std::cout << "        CL_DEVICE_ENDIAN_LITTLE=" << clboolValue << '\n';

	    devices[j].getInfo(CL_DEVICE_AVAILABLE, &clboolValue);
	    std::cout << "        CL_DEVICE_AVAILABLE=" << clboolValue << '\n';

	    devices[j].getInfo(CL_DEVICE_COMPILER_AVAILABLE, &clboolValue);
	    std::cout << "        CL_DEVICE_COMPILER_AVAILABLE=" << clboolValue << '\n';

	    cl_device_exec_capabilities cldeviceexeccapabilitiesValue;
	    devices[j].getInfo(CL_DEVICE_EXECUTION_CAPABILITIES, &cldeviceexeccapabilitiesValue);
	    std::cout << "        CL_DEVICE_EXECUTION_CAPABILITIES=";
	    if(cldeviceexeccapabilitiesValue & CL_EXEC_KERNEL)
		std::cout << "CL_EXEC_KERNEL ";
	    if(cldeviceexeccapabilitiesValue & CL_EXEC_NATIVE_KERNEL)
		std::cout << "CL_EXEC_NATIVE_KERNEL ";
	    std::cout << '\n';

	    cl_command_queue_properties clcommandqueuepropertiesValue;
	    devices[j].getInfo(CL_DEVICE_QUEUE_PROPERTIES, &clcommandqueuepropertiesValue);
	    std::cout << "        CL_DEVICE_QUEUE_PROPERTIES=";
	    if(clcommandqueuepropertiesValue & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
		std::cout << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ";
	    if(clcommandqueuepropertiesValue & CL_QUEUE_PROFILING_ENABLE)
		std::cout << "CL_QUEUE_PROFILING_ENABLE ";
	    std::cout << '\n';

	    cl_platform_id clplatformidValue;
	    devices[j].getInfo(CL_DEVICE_PLATFORM, &clplatformidValue);
	    std::cout << "        CL_DEVICE_PLATFORM=" << clplatformidValue << '\n';

	    devices[j].getInfo(CL_DEVICE_NAME, &stringValue);
	    std::cout << "        CL_DEVICE_NAME=" << stringValue << '\n';

	    devices[j].getInfo(CL_DEVICE_VENDOR, &stringValue);
	    std::cout << "        CL_DEVICE_VENDOR=" << stringValue << '\n';

	    devices[j].getInfo(CL_DRIVER_VERSION, &stringValue);
	    std::cout << "        CL_DRIVER_VERSION=" << stringValue << '\n';

	    devices[j].getInfo(CL_DEVICE_PROFILE, &stringValue);
	    std::cout << "        CL_DEVICE_PROFILE=" << stringValue << '\n';

	    devices[j].getInfo(CL_DEVICE_OPENCL_C_VERSION, &stringValue);
	    std::cout << "        CL_DEVICE_OPENCL_C_VERSION=" << stringValue << '\n';

	    devices[j].getInfo(CL_DEVICE_EXTENSIONS, &stringValue);
	    std::cout << "        CL_DEVICE_EXTENSIONS=" << stringValue << '\n';

	    //Features defined in OpenCL 1.2
#ifdef CL_VERSION_1_2
	    unsigned pointPos = platformVersion.find('.');
	    unsigned platformVersionMajor = ::atoi(platformVersion.substr(0, pointPos).c_str());
	    unsigned platformVersionMinor = ::atoi(platformVersion.substr(pointPos + 1).c_str());

	    if((platformVersionMajor >= 2) ||
		    ((platformVersionMajor == 1) && (platformVersionMinor >= 2))) {
		std::cout << "\n        OpenCL 1.2 features:\n";
		devices[j].getInfo(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, &sizetValue);
		std::cout << "            CL_DEVICE_IMAGE_MAX_BUFFER_SIZE="
			  << sizetValue << '\n';

		devices[j].getInfo(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, &sizetValue);
		std::cout << "            CL_DEVICE_IMAGE_MAX_ARRAY_SIZE="
			  << sizetValue << '\n';

		devices[j].getInfo(CL_DEVICE_LINKER_AVAILABLE, &clboolValue);
		std::cout << "            CL_DEVICE_LINKER_AVAILABLE="
			  << clboolValue << '\n';

		devices[j].getInfo(CL_DEVICE_BUILT_IN_KERNELS, &stringValue);
		std::cout << "            CL_DEVICE_BUILT_IN_KERNELS="
			  << stringValue << '\n';

		devices[j].getInfo(CL_DEVICE_PRINTF_BUFFER_SIZE, &sizetValue);
		std::cout << "            CL_DEVICE_PRINTF_BUFFER_SIZE="
			  << sizetValue << '\n';

		devices[j].getInfo(CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, &clboolValue);
		std::cout << "            CL_DEVICE_PREFERRED_INTEROP_USER_SYNC="
			  << clboolValue << '\n';

		cl_device_id cldeviceidValue;
		devices[j].getInfo(CL_DEVICE_PARENT_DEVICE, &cldeviceidValue);
		std::cout << "            CL_DEVICE_PARENT_DEVICE="
			  << cldeviceidValue << '\n';

		devices[j].getInfo(CL_DEVICE_PARTITION_MAX_SUB_DEVICES, &cluintValue);
		std::cout << "            CL_DEVICE_PARTITION_MAX_SUB_DEVICES="
			  << cluintValue << '\n';

		std::vector<cl_device_partition_property> cldevicepartitionpropertyVector;
		devices[j].getInfo(CL_DEVICE_PARTITION_PROPERTIES, &cldevicepartitionpropertyVector);
		std::cout << "            CL_DEVICE_PARTITION_PROPERTIES=";
		for(std::vector<cl_device_partition_property>::const_iterator k = cldevicepartitionpropertyVector.begin(); k != cldevicepartitionpropertyVector.end(); ++k) {
		    switch(*k) {
			case CL_DEVICE_PARTITION_EQUALLY:
			    std::cout << "CL_DEVICE_PARTITION_EQUALLY ";
			    break;
			case CL_DEVICE_PARTITION_BY_COUNTS:
			    std::cout << "CL_DEVICE_PARTITION_BY_COUNTS ";
			    break;
			case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
			    std::cout << "CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN ";
			    break;
			default:
			    std::cout << "[unknown (" << *k << ")] ";
		    }
		}
		std::cout << '\n';

		cl_device_affinity_domain cldeviceaffinitydomainValue;
		devices[j].getInfo(CL_DEVICE_PARTITION_AFFINITY_DOMAIN, &cldeviceaffinitydomainValue);
		std::cout << "            CL_DEVICE_PARTITION_AFFINITY_DOMAIN=";
		if(cldeviceaffinitydomainValue & CL_DEVICE_AFFINITY_DOMAIN_NUMA)
		    std::cout << "CL_DEVICE_AFFINITY_DOMAIN_ ";
		if(cldeviceaffinitydomainValue & CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE)
		    std::cout << "CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE ";
		if(cldeviceaffinitydomainValue & CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE)
		    std::cout << "CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE ";
		if(cldeviceaffinitydomainValue & CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE)
		    std::cout << "CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE ";
		if(cldeviceaffinitydomainValue & CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE)
		    std::cout << "CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE ";
		if(cldeviceaffinitydomainValue & CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE)
		    std::cout << "CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE ";
		std::cout << '\n';

		devices[j].getInfo(CL_DEVICE_PARTITION_TYPE, &cldevicepartitionpropertyVector);
		std::cout << "            CL_DEVICE_PARTITION_TYPE=";
		for(std::vector<cl_device_partition_property>::const_iterator k = cldevicepartitionpropertyVector.begin(); k != cldevicepartitionpropertyVector.end(); ++k) {
		    switch(*k) {
			case CL_DEVICE_PARTITION_EQUALLY:
			    std::cout << "CL_DEVICE_PARTITION_EQUALLY ";
			    break;
			case CL_DEVICE_PARTITION_BY_COUNTS:
			    std::cout << "CL_DEVICE_PARTITION_BY_COUNTS ";
			    break;
			case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
			    std::cout << "CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN ";
			    break;
			case CL_DEVICE_AFFINITY_DOMAIN_NUMA:
			    std::cout << "CL_DEVICE_AFFINITY_DOMAIN_NUMA ";
			    break;
			case CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE:
			    std::cout << "CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE ";
			    break;
			case CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE:
			    std::cout << "CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE ";
			    break;
			case CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE:
			    std::cout << "CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE ";
			    break;
			case CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE:
			    std::cout << "CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE ";
			    break;
			default:
			    std::cout << "[unknown (" << *k << ")] ";
		    }
		}
		std::cout << '\n';

		devices[j].getInfo(CL_DEVICE_REFERENCE_COUNT, &cluintValue);
		std::cout << "            CL_DEVICE_REFERENCE_COUNT="
			  << cluintValue << '\n';
	    }
#endif //CL_VERSION_1_2

	    std::cout << '\n';


	} //for device
    } //for platform

    return EXIT_SUCCESS;
}

/**
 * @brief Gets message (text string) associated to error code
 * @param[in] err error code
 * @return the message associated to error code
 */
const char* CLapp::getOpenCLErrorCodeStr(cl_int err) {
    if(errStrings.find(err) != errStrings.end())
	return errStrings[err];
    else
	return "unknown error code";
}

const std::string CLapp::getOpenCLErrorInfoStr(CLError& err, const std::string& msg) {
    std::stringstream errorInfo;
    errorInfo << "Error: " << err.what() << " (CL Exception " << err.err() << ", "
	      << CLapp::getOpenCLErrorCodeStr(err.err()) << ")" << "\nat " << msg << std::endl;
      return errorInfo.str();
}

const std::string CLapp::getOpenCLErrorExtendedInfoStr(CLError& err, const std::string& msg, cl::Program program,
	cl::Device selected_device) {
    std::stringstream errorInfo;
    errorInfo << getOpenCLErrorInfoStr(err, msg);
    if((err.err() == CL_BUILD_PROGRAM_FAILURE) || (err.err() == CL_INVALID_KERNEL)) {
	errorInfo << "Extended info: ";
	std::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device);
	errorInfo << "Program Info: " << str << std::endl << std::flush;
    }
    return errorInfo.str();
}

/**
 * @brief Calculates a good (not necessarily optimal) workgroup size for a given device, according to its claimed preferences
 * @param[in] kernel kernel for which to calculate the local size
 * @param[in] device CL device where the kernel will be enqueued
 * @param[in] globalSize global work size
 * @return a (hopefully) good workgroup size
 */
cl::NDRange CLapp::calcLocalSize(const cl::Kernel& kernel, const cl::Device& device, const cl::NDRange& globalSize) {
    // First, see if the kernel programmer set a mandatory workgroup size
    std::array<size_t, 3> compiledSize = kernel.getWorkGroupInfo<CL_KERNEL_COMPILE_WORK_GROUP_SIZE>(device);
    if((compiledSize[0] != 0) || (compiledSize[1] != 0) || (compiledSize[2] != 0))
	return cl::NDRange(compiledSize[0], compiledSize[1], compiledSize[2]);

    size_t preferredMultiple = kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
    //size_t maxLocalSize=kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

    // Find a work size that will keep compute units as full as possible (we implicitly asume that preferredMultiple is the actual
    // number of processing elements in a hardware compute unit)
    std::array<size_t, 3> goodSize;
    size_t dims = globalSize.dimensions();
    for(size_t dim = 0; dim < dims; dim++)
	goodSize[dim] = preferredMultiple <= globalSize[dim] ? preferredMultiple: globalSize[dim];

    switch(dims) {
	case 1:
	    return cl::NDRange(goodSize[0]);
	case 2:
	    return cl::NDRange(goodSize[0], goodSize[1]);
	case 3:
	    return cl::NDRange(goodSize[0], goodSize[1], goodSize[2]);
	default:
	    BTTHROW(std::invalid_argument("CLapp::calcLocalSize: Impossible (>3) number of dimensions in NDRange"), "CLapp::calcLocalSize");
    }
}

/**
 * @brief Rounds up numToRound to multiple of baseNumber
 * @param[in] numToRound positive number to be rounded
 * @param[in] baseNumber value whose multiple nearest to numToRound is returned
 * @return rounded number (nearest multiple of baseNumber)
 */
cl_uint CLapp::roundUp(cl_uint numToRound, cl_uint baseNumber) {
    //assert(baseNumber);
    cl_uint remainder = numToRound % baseNumber;
    //CERR("remainder: " << remainder << std::endl);
    if(remainder == 0) {
	return numToRound;
    }
    else {
	return numToRound + baseNumber - remainder;
    }
}

/**
 * @brief Adds a new Data subclass object to the list of active data
 * (data used for algorithms executed on OpenCL selected device)
 * @param[in] pData pointer to an object subclass of Data class
 * @return data handle for the added object
 */
DataHandle CLapp::addData(std::shared_ptr<Data> pData, bool copyDataToDevice) {
    return addData(pData.get(), copyDataToDevice);
}

DataHandle CLapp::addData(Data* pData, bool copyDataToDevice) {
    const std::lock_guard<std::mutex> lock(dataMapMutex);

    // If Data has been previously added to pCLapp vector of Data, first we have to remove it from that vector.
    // This way, device memory and host memory mapped to device memory are recreated and data from
    // non-mapped host memory is copied again (non-mapped host memory data may have been updated since last
    // addData).
    if(pData->getHandle() != INVALIDDATAHANDLE) {
	delData(pData->getHandle());
	pData->setHandle(INVALIDDATAHANDLE);
    }
    //if(nextDataKey == 17) BTTHROW(std::invalid_argument("OJO CUIDAO"),"CLapp::addData()");
    DataHandle thisDataKey = nextDataKey++;
    dataMap[thisDataKey] = std::make_shared<DeviceDataProperties>(shared_from_this(), pData, copyDataToDevice);
    pData->setHandle(thisDataKey);
    return thisDataKey;
}
/**
 * @brief Checks if handle exists in the data map. Throw exception if not exists.
 * @param[in] handle handle for an existing object subclass of Data class to be checked
 * @param[in] specificMessage message for the exception
 * @exception ivalid_argument if data handle does not exist in the data map
 */
void CLapp::checkDataHandle(DataHandle handle, std::string specificMessage) {
	//const std::lock_guard<std::mutex> lock(dataMapMutex);
	uint elementCount;
    elementCount = dataMap.count(handle);
    if(elementCount != 1) {
	BTTHROW(std::invalid_argument("Invalid handle (" + std::to_string(handle) + "), " + specificMessage), "CLapp::checkDataHandle");
    }
}

/**
 * @brief Deletes data subclass object from the list of active data
 * (data used for algorithms executed on OpenCL selected device)
 * @param[in] handle handle for an existing object subclass of Data class
 */
void CLapp::delData(DataHandle handle) {
    const std::lock_guard<std::mutex> lock(dataMapMutex);

    checkDataHandle(handle, "data not deleted from CLapp");
    CLAPP_CERR("data map size before delData: " << dataMap.size() << std::endl);

    // Delete DeviceDataProperties object
    dataMap[handle] = nullptr;
    // Remove shared pointer  to Data object from process map
    dataMap.erase(handle);

    CLAPP_CERR("data map size after delData: " << dataMap.size() << std::endl);
}

/**
 * @brief Gets Data object from the list of active data elements using a handle
 * @param[in] handle data handle of the data object we want to get
 * @return object subclass of Data belonging to active data list
 * @exception std::out_of_range if handle is out of bounds
 */
//std::shared_ptr<Data> CLapp::getData(DataHandle handle) {
Data* CLapp::getData(DataHandle handle) {
	const std::lock_guard<std::mutex> lock(dataMapMutex);
    return dataMap.at(handle)->getData(); // at methods checks bounds; operator [], not
}

/**
 * Get Data object from the list of active data elements using a handle
 * @param[in] handle data handle of the data object we want to get
 * @return object subclass of Data belonging to active data list
 * (exception std::out_of_range is thrown if handle is out of bounds)
 */
std::shared_ptr<DeviceDataProperties> CLapp::getDeviceDataProperties(DataHandle handle) const {
	const std::lock_guard<std::mutex> lock(dataMapMutex);
	return dataMap.at(handle); // at methods checks bounds; operator [], not
}

/**
 * @brief Copies data stored in host memory to device memory (as buffers, images or both) for a data represented by a handle
 * @param[in] handle data handle of the data object
 */
void CLapp::host2Device(DataHandle handle, bool copyData) {
	const std::lock_guard<std::mutex> lock(dataMapMutex);
	checkDataHandle(handle, "host2Device aborted");
    CLAPP_CERR("data map size before delData: " << dataMap.size() << std::endl);
    dataMap[handle]->host2Device(copyData);
}

/**
 * @brief Copies data stored in device memory to host memory (from buffers or images) for a data represented by a handle
 * @param[in] pData pointer to data object
 * @param[in] queueFinish true if queue.finish() method must be called to guarantee that kernel execution has finished before
 * copying data back to host memory
 */
void CLapp::device2Host(std::shared_ptr<Data> pData, bool queueFinish) {
	const std::lock_guard<std::mutex> lock(dataMapMutex);
	DataHandle handle = pData->getHandle();
    checkDataHandle(handle, "device2Host aborted");
    CLAPP_CERR("data map size before delData: " << dataMap.size() << std::endl);
    dataMap[handle]->device2Host(queueFinish);
}

/**
 * @brief Copies data stored in device memory to host memory (from buffers or images) for a data represented by a handle
 * @param[in] handle data handle of the data object
 */
void CLapp::device2Host(DataHandle handle, bool queueFinish) {
	const std::lock_guard<std::mutex> lock(dataMapMutex);
	checkDataHandle(handle, "device2Host aborted");
    CLAPP_CERR("data map size before delData: " << dataMap.size() << std::endl);
    dataMap[handle]->device2Host(queueFinish);
}

/**
 * @brief Returns a string representing the device type of device with index i
 *
 * @param[in] i index of device
 * @return a string representing the device type of device with index i
 */
std::string CLapp::getDeviceTypeAsString(size_t i) {
    cl_device_type devType;
    devices[i].getInfo(CL_DEVICE_TYPE, &devType);
    std::string deviceTypeString;
    if(devType & CL_DEVICE_TYPE_DEFAULT)
	deviceTypeString =  "CL_DEVICE_TYPE_DEFAULT";
    if(devType & CL_DEVICE_TYPE_CPU)
	deviceTypeString =  "CL_DEVICE_TYPE_CPU";
    if(devType & CL_DEVICE_TYPE_GPU)
	deviceTypeString = "CL_DEVICE_TYPE_GPU";
    if(devType & CL_DEVICE_TYPE_ACCELERATOR)
	deviceTypeString = "CL_DEVICE_TYPE_ACCELERATOR";
    if(devType & CL_DEVICE_TYPE_CUSTOM)
	deviceTypeString = "CL_DEVICE_TYPE_CUSTOM";
#ifdef CL_VERSION_1_2
    if(devType & CL_DEVICE_TYPE_CUSTOM)
	deviceTypeString = "CL_DEVICE_TYPE_CUSTOM";
#endif //CL_VERSION_1_2
    return deviceTypeString;
}

/**
 * @brief Returns an InfoItems object containing information about hardware and software
 * of the selected OpenCL device
 *
 * @param[in] i index of device
 * @return an object of InfoItems class
 */
LPISupport::InfoItems CLapp::getHWSWInfo(size_t i) {
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    LPISupport::InfoItems infoItemsHWSW;
    infoItemsHWSW.addInfoItem("Host name", hostname);
    infoItemsHWSW.addInfoItem("Device name", devices[i].getInfo<CL_DEVICE_NAME>());
    infoItemsHWSW.addInfoItem("Device version", devices[i].getInfo<CL_DEVICE_VERSION>());
    infoItemsHWSW.addInfoItem("Device type", getDeviceTypeAsString());
    uint deviceMaxWorkItemDimensions = devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
    std::string maxWorkItemsSizes;
    for(uint j = 0; j < deviceMaxWorkItemDimensions; j++) {
	maxWorkItemsSizes.append(std::to_string((devices[i].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>())[j]));
	maxWorkItemsSizes.append(" ");
    }
    infoItemsHWSW.addInfoItem("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS", std::to_string(deviceMaxWorkItemDimensions));
    infoItemsHWSW.addInfoItem("CL_DEVICE_MAX_WORK_ITEM_SIZES", maxWorkItemsSizes);
    infoItemsHWSW.addInfoItem("CL_DEVICE_MAX_WORK_GROUP_SIZE", std::to_string(devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()));
    return infoItemsHWSW;
}

/**
 * @brief Returns the name of device with index <i>i</i>
 *
 * @param[in] i index of device
 * @return a string representing the name of the specified device
 */
std::string CLapp::getDeviceName(size_t i) {
    return devices[i].getInfo<CL_DEVICE_NAME>();
}

/**
 * @brief Returns the vendor of device with index <i>i</i>
 *
 * @param[in] i index of device
 * @return a string representing the vendor of the specified device
 */
std::string CLapp::getDeviceVendor(size_t i) {
    return devices[i].getInfo<CL_DEVICE_VENDOR>();
}

/**
    * @brief Gets OpenCL context
    * @return reference to the OpenCL context
    */
const cl::Context& CLapp::getContext() const {
    return context;
}

/**
    * @brief Gets OpenCL device from list of devices
    * @param[in] i position of device in the list of devices
    * @return reference to selected device
    */
const cl::Device& CLapp::getDevice(const size_t i) const {
    return devices[i];
}

/**
    * @brief Gets HIP device
    * @return reference to selected HIP device
    */
#ifdef HAVE_HIP
const hipDevice_t CLapp::getHIPDevice() const {
    return hipDevice;
}
#endif

/**
    * @brief Gets OpenCL command queue from list of command queues
    * @param[in] i position of queue in the list of queues
    * @return reference to selected queue
    */
cl::CommandQueue& CLapp::getCommandQueue(const size_t i) {
    return commandQueues[i];
}

/**
    * @brief Gets OpenCL program from list of programs
    * @param[in] i position of program in the list of programs
    * @return reference to selected program
    */
// const cl::Program& CLapp::getProgram(const size_t i) const {
//     return *programs[i];
// }

/**
 * @brief Returns maximum supported local work item size depending on selected global sizes and device features
 *
 * @param[in] globalSizes cl::NDRange value specifying global sizes
 * @return a value of type cl::NDRange containing the calculated local sizes
 */
cl::NDRange CLapp::getMaxLocalWorkItemSizes(cl::NDRange globalSizes) {
    uint localSizesArray[3];
    uint availableMaxGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    for(uint i = 0; i < globalSizes.dimensions(); i++) {
	if(globalSizes.get()[i] <= availableMaxGroupSize) {
	    localSizesArray[i] = globalSizes.get()[i];
	}
	else {
	    uint submultiple = globalSizes.get()[i];
	    while(submultiple > availableMaxGroupSize) {
		submultiple /= 2;
	    }
	    localSizesArray[i] = submultiple;
	}
	availableMaxGroupSize /= localSizesArray[i];
    }
    cl::NDRange localSizes {localSizesArray[0], localSizesArray[1], localSizesArray[2]};
    return localSizes;
}

/**
 * @brief Gets pointer to data of a all NDArrays of a Data object stored in device memory as a buffer
 * (pointer to object of class cl::Buffer).
 * @param[in] handle data handle of the data object
 * @return raw pointer to object of class cl::Buffer
 */
cl::Buffer* CLapp::getDeviceBuffer(DataHandle handle) {
    return this->getDeviceDataProperties(handle)->getDeviceBuffer();
}

/**
 * @brief Gets HIP pointer to data of a all NDArrays of a Data object stored in device memory as a buffer
 * (raw void* pointer).
 * @param[in] handle data handle of the data object
 * @return raw void* HIP pointer to device data
 */
void* CLapp::getHIPDeviceBuffer(DataHandle handle) {
    return this->getDeviceDataProperties(handle)->getHIPDeviceBuffer();
}

/**
 * @brief Gets pointer to data of a all NDArrays of a Data object stored in device memory as a buffer
 * (pointer to object of class cl::Buffer).
 * @param[in] handle data handle of the data object
 * @param[in] NDArrayIndex NDarray index for the vector of NDArrays
 * @return raw pointer to object of class cl::Buffer
 */
cl::Buffer* CLapp::getDeviceBuffer(DataHandle handle, dimIndexType NDArrayIndex) {
    return this->getDeviceDataProperties(handle)->getDeviceBuffer(NDArrayIndex);
}

/**
 * @brief Gets pointer to data of array of dimensions and strides and all NDArrays of a Data object stored in contiguous device memory as a buffer
 * (pointer to object of class cl::Buffer).
 * @param[in] handle data handle of the data object
 * @return raw pointer to object of class cl::Buffer
 */
cl::Buffer* CLapp::getCompleteDeviceBuffer(DataHandle handle) {
    return this->getDeviceDataProperties(handle)->getCompleteDeviceBuffer();
}

/**
 * @brief Gets pointer to data of a NDArray stored in host memory as a buffer
 * (pointer to object of class cl::Buffer).
 * @param[in] handle data handle of the data object
 * @param[in] NDArrayIndex NDarray index for the vector of NDArrays
 * @return raw pointer to object of class cl::Buffer
 */
void* CLapp::getHostBuffer(DataHandle handle, dimIndexType NDArrayIndex) {
    return this->getDeviceDataProperties(handle)->getHostBuffer(NDArrayIndex);
}

/**
 * @brief Gets pointer to spatial and temporal data dimensions and their strides stored in host memory as an array.
 * @param[in] handle data handle of the data object
 * @return generic pointer to array of spatial and temporal data dimensions and their strides
 */
const void* CLapp::getDataDimsAndStridesHostBuffer(DataHandle handle) {
    return this->getDeviceDataProperties(handle)->getDataDimsAndStridesHostBuffer();
}

/**
 * @brief Gets pointer to spatial and temporal data dimensions and their strides stored in contiguous device memory as a buffer
 * (pointer to object of class cl::Buffer).
 * @param[in] handle data handle of the data object
 * @return raw pointer to object of class cl::Buffer
 */
cl::Buffer* CLapp::getDataDimsAndStridesDeviceBuffer(DataHandle handle) {
    return this->getDeviceDataProperties(handle)->getDataDimsAndStridesDeviceBuffer();
}

/**
    * @brief Sets error text explantation from error code (integer)
    * @param[in] err error code
    * @param[out] errStr text message for the error code
    */
void CLapp::setOpenCLErrorCodeStr(const cl_int err, const char* errStr) {
    errStrings[err] = errStr;
}

/**
 * @brief shows kernel build errors on stderr.
 * @param[in] e Reference to exception object thrown by cl::build() or other similar methods
 */
void CLapp::dumpBuildError(cl::BuildError& e) {
    std::cerr << "CL program compilation failed!\n";
    const auto& buildLogs = e.getBuildLog();
    for(auto&& i : buildLogs) {
	std::cerr << "While compiling program for device \"" << i.first.getInfo<CL_DEVICE_NAME>() << "\":\n"
		  << "----------------------------------------------------------------------\n"
		  << i.second
		  << "\n----------------------------------------------------------------------\n\n";
    }
}

long CLapp::score(const cl::Device& device) {
    auto freq = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
    auto numCU = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    unsigned coresPerCU;
    unsigned clocksPerInstruction;

    if(device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
	auto vendorID = device.getInfo<CL_DEVICE_VENDOR_ID>();
	switch(vendorID) {
	    case 0x1002: // AMD
		coresPerCU = 64;
		clocksPerInstruction = 4; // Note: this applies to GCN chips only!!
		break;
	    case 0x10de: // nVIDIA
		coresPerCU = 32;
		clocksPerInstruction = 1;
		break;
	    default:
		coresPerCU = 1;
		clocksPerInstruction = 1;
		CLAPP_CERR("CLapp::score: Warning: unknown GPU vendor. Assuming 1 core per CU\n");
	}
    }
    else {
	coresPerCU = 1;
	clocksPerInstruction = 1;
    }

    long score = freq * numCU * coresPerCU / clocksPerInstruction;
    return score;
}

void CLapp::dumpDeviceData() const {
    std::cerr<<"Handle; #NDArrays; Size\n";

    //const std::lock_guard<std::mutex> lock(dataMapMutex);
    for(auto i: dataMap)
	std::cerr<<i.first<<": "<<i.second->getData()->getNDArrays()->size()<<"; "<<i.second->pDeviceBuffer->getInfo<CL_MEM_SIZE>()<<'\n';
}

} // namespace OpenCLIPER

#undef CLAPP_DEBUG
