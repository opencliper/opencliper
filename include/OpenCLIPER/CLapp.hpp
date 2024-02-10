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
#ifndef CLAPP_HPP
#define CLAPP_HPP

#include<OpenCLIPER/defs.hpp>

#if defined(__APPLE__) || defined(__MACOSX)
    #include<OpenCL/cl.hpp>
#else
    #ifdef HAVE_OPENCL_HPP
	#include<CL/opencl.hpp>
    #else
	#include<CL/cl2.hpp>
    #endif
#endif

#include<vector>
#include<map>
#include<atomic>
#include<mutex>
// For gethostname POSIX function
#include<stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>

#include<OpenCLIPER/DeviceDataProperties.hpp>
#include<LPISupport/Utils.hpp>
#include<LPISupport/InfoItems.hpp>

// Ensure AMD/nVidia constants/types to retrieve PCIe bus id of devices are defined
#ifndef CL_DEVICE_TOPOLOGY_AMD
    #define CL_DEVICE_TOPOLOGY_AMD 0x4037
#endif

#ifndef CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD
    #define CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD 1
#endif

#ifndef HAVE_CL_DEVICE_TOPOLOGY_AMD
typedef union {
    struct {
	cl_uint type;
	cl_uint data[5];
    } raw;
    struct {
	cl_uint type;
	cl_char unused[17];
	cl_char bus;
	cl_char device;
	cl_char function;
    } pcie;
} cl_device_topology_amd;
#endif

#ifndef CL_DEVICE_PCI_BUS_ID_NV
    #define CL_DEVICE_PCI_BUS_ID_NV 0x4008
#endif

#ifndef CL_DEVICE_PCI_SLOT_ID_NV
    #define CL_DEVICE_PCI_SLOT_ID_NV 0x4009
#endif

#ifdef HAVE_HIP
    #define __HIP_PLATFORM_HCC__
    #include <hip/hip_runtime_api.h>
#endif

// opencl.hpp includes the | operator but not &
// We need it to test for supported command queue properties
/*
inline cl::QueueProperties operator&(cl::QueueProperties lhs, cl::QueueProperties rhs) {
    return static_cast<cl::QueueProperties>(static_cast<cl_command_queue_properties>(lhs) & static_cast<cl_command_queue_properties>(rhs));
}
*/

namespace OpenCLIPER {

class Data;

// Class to report CL-like exceptions. Storage is done in an std::string instead of a char*, so that a copy of the what() message is kept within the class.
// This allows to construct variable error strings on the fly and use them in catch safely (CLError will lose the target to its char* once the stack is unwound and catch is reached)
class CLError: public cl::Error {
        private:
        cl_int err_;
        std::string errStr_;
    public:
        /*! \brief Create a new CL error exception for a given error code
         *  and corresponding message. This message is stored internally within an std::string.
         *
         *  \param err error code value.
         *
         *  \param errStr a descriptive string of the error cause.  If set, it
         *                will be returned by what().
         */
        CLError(cl_int err, std::string errStr = NULL): Error(err, errStr.c_str()), err_(err), errStr_(errStr) {}
        CLError(cl::Error& e): Error(e), err_(e.err()), errStr_(e.what()) {}

        ~CLError() {}

        /*! \brief Get error string associated with exception
         *
         * \return A const reference to the error message string.
         */
        virtual const char* what() const throw() { return errStr_.c_str(); }

        /*! \brief Get error code associated with exception
         *
         *  \return The error code.
         */
        cl_int err() const { return err_; }

};

/**
 * @brief Class representing an application with an OpenCL device bound
 *
 * This class stores:
 * * infomation about available and selected devices
 * * a list of Process subclasses objects (representing algorithms to be executed on selected device)
 * * a list of Data subclasses objects (used as input and/or output data for Process subclasses objects).
 */
class CLapp: public std::enable_shared_from_this<CLapp> {

	friend class Data;
	friend class XData;
	friend class KData;


    public:
	//----------------------------------------------
	// Classes
	//----------------------------------------------

	/// Enumerated type with the device type options
	enum DeviceType {
	    DEVICE_TYPE_ANY = CL_DEVICE_TYPE_ALL, ///< all device types allowed
	    DEVICE_TYPE_CPU = CL_DEVICE_TYPE_CPU, ///< only CPU-type devices allowed
	    DEVICE_TYPE_GPU = CL_DEVICE_TYPE_GPU ///< only GPU-type devices allowed
	};

	/// Options for OpenCL platform selection
	struct PlatformTraits {
	    /// platform name
	    std::string		     name;

	    /// platform vendor
	    std::string		     vendor;

            /// platform version
	    std::string		     version;

            /// required platform extensions
	    std::vector<std::string> extensions;
	};

	/// Options for OpenCL device selection
	struct DeviceTraits {
	    /// device type (CPU, GPU, etc.)
	    DeviceType			type;

	    /// device name
	    std::string			name;

	    /// device vendor
	    std::string			vendor;

            /// device version
	    std::string			version;

            /// required device extensions
	    std::vector<std::string>	extensions;

            /// device queue properties
	    cl::QueueProperties		queueProperties;

            /// Use HIP if available
	    bool			useHIP = false;

	    /**
	     * @brief Default constructor for struct fields initialization.
	     */
	    DeviceTraits(DeviceType t = DEVICE_TYPE_ANY, cl::QueueProperties p = cl::QueueProperties::None):
		type(t), queueProperties(p) {}
	};

	struct KernelProperties {
	    cl::Kernel kernel;
	    std::string sourceFile;

	    KernelProperties() {}
	    KernelProperties(const std::string& s): kernel(), sourceFile(s) {}
	};

	struct KernelFileProperties {
	    std::string sourcePath;
            bool loaded = false; // True if this file has alredy been loaded

	    KernelFileProperties(): loaded(false) {}
	    KernelFileProperties(const std::string& s): sourcePath(s), loaded(false) {}
	};

	//----------------------------------------------
	// Methods
	//----------------------------------------------

	// Creation/destruction
	static std::shared_ptr<CLapp> create();
	static std::shared_ptr<CLapp> create(const PlatformTraits& platformTraits, const DeviceTraits& deviceTraits);
	~CLapp();

	// Platform/device initialization
	void	init();
	void	init(const PlatformTraits& platformTraits, const DeviceTraits& deviceTraits, bool quiet = false);

	// Kernel management
	cl::Kernel&	getKernel(const size_t i = 0);
	cl::Kernel&	getKernel(const std::string& name);
	void		addKernelFile(const std::string& sourceFile);
	void		addKernelDir(const std::string& kernelDir);
	void		loadKernels(const char* compilerOptionsArg = nullptr);

	// Data management
	Data*					getData(DataHandle handle);
	std::shared_ptr<DeviceDataProperties>   getDeviceDataProperties(DataHandle handle) const;
	void					host2Device(DataHandle handle, bool copyData=true);
	void					device2Host(std::shared_ptr<Data> pData, bool queueFinish = true);
	void					device2Host(DataHandle handle, bool queueFinish = true);
	cl::Buffer*				getDeviceBuffer(DataHandle handle, dimIndexType NDArrayIndex);
	cl::Buffer*				getDeviceBuffer(DataHandle handle);
	void*					getHIPDeviceBuffer(DataHandle handle);
	cl::Buffer*				getCompleteDeviceBuffer(DataHandle handle);
	void* 					getHostBuffer(DataHandle handle, dimIndexType NDArrayIndex);
	const void*				getDataDimsAndStridesHostBuffer(DataHandle handle);
	cl::Buffer*				getDataDimsAndStridesDeviceBuffer(DataHandle handle);

	// Error management
	void				checkDataHandle(DataHandle handle, std::string specificMessage);
	static const char*		getOpenCLErrorCodeStr(const cl_int err);
	void				setOpenCLErrorCodeStr(const cl_int err, const char* errStr);
	static void			dumpBuildError(cl::BuildError& e);
	static const std::string	getOpenCLErrorInfoStr(CLError& err, const std::string& msg);
	static const std::string	getOpenCLErrorExtendedInfoStr(CLError& err, const std::string& msg, cl::Program program,
							       cl::Device selected_device);

	// Device-specific calculations
	static cl::NDRange	calcLocalSize(const cl::Kernel& kernel, const cl::Device& device, const cl::NDRange& globalSize);
	static 	cl_uint		roundUp(cl_uint numToRound, cl_uint baseNumber);
	cl::NDRange		getMaxLocalWorkItemSizes(cl::NDRange globalSizes);
	static long		score(const cl::Device& device);

	// General info
	static int		dumpInfo();
	std::string		getDeviceTypeAsString(size_t i = 0);
	LPISupport::InfoItems	getHWSWInfo(size_t i = 0);
	const cl::Device&	getDevice(const size_t i = 0) const;
	std::string		getDeviceName(size_t i = 0);
	std::string		getDeviceVendor(size_t i = 0);
	const			cl::Context& getContext() const;
	cl::CommandQueue&	getCommandQueue(const size_t i = 0);
	//const cl::Program&	getProgram(const size_t i = 0) const;
	void 			dumpDeviceData() const;
#ifdef HAVE_HIP
	const hipDevice_t	getHIPDevice() const;
#endif

    protected:
	// Data management
	DataHandle      addData(std::shared_ptr<Data> pData, bool copyDataToDevice);
	DataHandle      addData(Data* pData, bool copyDataToDevice);
	void            delData(DataHandle handle);

    private:
	/// @brief Default constructor (empty). Note: use create() methods instead of calling this (private) constructor.
	// The real work is done in the public create() methods
	CLapp(): nextDataKey(FIRSTVALIDDATAHANDLE) {}

	/// OpenCL platform
	cl::Platform			platform;

	/// OpenCL context
	cl::Context			context;

	/// List of OpenCL devices
	std::vector<cl::Device>		devices;

	/// List of "device strings". Used to generate hashes unique to each compiled CL program
	std::vector<std::string>	deviceStrings;

	/// HIP device equivalent to first OpenCL device. We should maintain a vector of pairs or so here...
#ifdef HAVE_HIP
	hipDevice_t                         hipDevice;
#endif

	/// List of OpenCL command queues
	std::vector<cl::CommandQueue>	commandQueues;

	/// List of programs
	//std::vector<std::shared_ptr<cl::Program>>	programs;

        /// List of kernel files
        typedef std::map<std::string, KernelFileProperties> KernelFileList;
	KernelFileList			kernelFiles;

        /// List of directories to look into for kernel files
	typedef std::set<std::string> KernelPathList;
	KernelPathList			kernelDirs;

	/// List of kernels
	typedef std::map<std::string, KernelProperties> KernelList;
	KernelList			kernels;

	/// True if loadKernels has returned successfully at least once
	bool				kernelsLoaded = false;

	/// Map with data handles as keys and smart shared pointers to DeviceDataProperties objects as values
	/// Access to dataMap must be protected by a mutex (std::map is not thread-safe)
	std::map<DataHandle, std::shared_ptr<DeviceDataProperties>>     dataMap;

	/// Current valid value for data keys (initially not valid)
	std::atomic<DataHandle>		nextDataKey;

	/// Error strings for CL error codes
	static std::map<const cl_int, const char*>	errStrings;
};

} //namespace OpenCLIPER
#endif // CLAPP_HPP
