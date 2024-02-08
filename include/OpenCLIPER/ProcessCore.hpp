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
// Avoiding compiler errors due to multiple include of header files
#ifndef PROCESSCORE_HPP
#define PROCESSCORE_HPP

#include <OpenCLIPER/defs.hpp>
#include <OpenCLIPER/Data.hpp>
#include <LPISupport/SampleCollection.hpp>
#include <LPISupport/InfoItems.hpp>

#include <iostream> // std::cout, std::fixed
#include <iomanip> // std::setprecision

// Needed for kernels to find includes outside "kernels" subdirectory
//#define KERNELCOMPILEOPTS "-I../include/"

namespace OpenCLIPER {

class CLapp;

/**
 * @brief Class representing common data and behaviour for all the processes in charge of operations on data.
 *
 */
class ProcessCore {
    public:
	/// Initialization parameters (to be redefined in the subclass, if needed)
	struct InitParameters {
	    /// @brief Destructor
	    virtual ~InitParameters() {}
	};

	/// Launch parameters (to be redefined in the subclass, if needed)
	struct LaunchParameters {
	    /// @brief Destructor
	    virtual ~LaunchParameters() {}
	};

	/// Profiling parameters
	struct ProfileParameters {
            /// Enable/disable host and kernel profiling for this process
            bool enable;
            
            /// Number of times to run this process' kernel when profiling
            unsigned long loops;
            
            /// @brief Constructors
            ProfileParameters(): enable(false), loops(1) {}
            ProfileParameters(bool e, unsigned long l): enable(e), loops(l) {}

	    /// @brief Destructor
	    virtual ~ProfileParameters() {}
	};

        /// Constructors
	ProcessCore(const std::shared_ptr<ProfileParameters>& pPP = nullptr);
	ProcessCore(const std::shared_ptr<Data>& pInputData, const std::shared_ptr<Data>& pOutputData, const std::shared_ptr<ProfileParameters>& pPP = nullptr);

	/// Destructor
	virtual ~ProcessCore();

        /**
	* @brief Returns a SampleCollection object containing values of host+device execution times
	* @returns object of SampleCollection class
	*/
	std::shared_ptr<LPISupport::SampleCollection> getSamplesGPU_CPUExecTime() {
	    return pSamplesGPU_CPUExecutionTime;
	}

	/**
	* @brief Returns a SampleCollection object containing values of device execution times
	* @returns object of SampleCollection class
	*/
	std::shared_ptr<LPISupport::SampleCollection> getSamplesGPUExecTime() {
	    return pSamplesGPUExecutionTime;
	}

	/**
	* @brief Returns pointer to init parameters
	* @returns smart shared pointer to InitParameters object
	*/
	virtual const std::shared_ptr<InitParameters> getInitParameters() const {
	    return pInitParameters;
	}

	/**
	* @brief Returns pointer to launch parameters
	* @returns smart shared pointer to LaunchParameters object
	*/
	virtual const std::shared_ptr<LaunchParameters> getLaunchParameters() const {
	    return pLaunchParameters;
	}

	void setCommandQueue(const cl::CommandQueue& cq) { queue = cq; }
	virtual void setApp(const std::shared_ptr<CLapp>& pCLapp) = 0;
	void setInput(std::shared_ptr<Data> pInputData);
	void setOutput(std::shared_ptr<Data> pOutputData);

	/**
	* @brief Sets initialization parameters for a Process object (this method can be redefined in a Process subclass).
	* @param[in] p structure with initialization parameters for a Process object
	*/
	virtual void setInitParameters(const std::shared_ptr<InitParameters>& p) {
	    pInitParameters = p;
	}

	/**
	* @brief Sets launch parameters for a Process object (this method can be redefined in a Process subclass).
	* @param[in] p structure with launch parameters of a Process object
	*/
	virtual void setLaunchParameters(const std::shared_ptr<LaunchParameters>& p) {
	    pLaunchParameters = p;
	}

	/**
	* @brief Method for Process object initialization (intialization is specific of Process subclasses).
	*/
	virtual void init() {
	    if(!getApp())
                BTTHROW(std::invalid_argument("Invalid CLapp pointer"), "ProcessCore::init");
	}

	/**
	* @brief Method that runs OpenCL kernel(s) associated to this process (subclasses must implement it)
	* @param[in] profileParameters parameters related to profiling of kernel execution
	*/
	virtual void launch() = 0;

	/**
	* @brief Gets infoItems class variable value
	*
	* @return the value of infoItems class variable
	*/
	LPISupport::InfoItems getInfoItems() {
	    return infoItems;
	}


    protected:
	/**
	* @brief Get value of shared pointer to CLapp process assigned to this object (must be implemented by derived classes)
	* @return shared pointer to CLapp object
	*/
	virtual const std::shared_ptr<CLapp> getApp() const = 0;

	std::shared_ptr<Data> getInput();

	/**
	* @brief Returns handle associated to input data
	* @return handle to input data
	*/
	DataHandle getInHandle() {
	    return inHandle;
	}

	std::shared_ptr<Data> getOutput();

	/**
	* @brief Returns handle associated to output data
	* @return handle to output data
	*/
	DataHandle getOutHandle() {
	    return outHandle;
	}

	void checkCommonLaunchParameters();
	void checkXDataLaunchParameters(SyncSource syncSource = SYNCSOURCEDEFAULT);
	void startProfiling();
	void stopProfiling();
	void startHostCodeProfiling();
	void stopHostCodeProfiling();
	void startKernelProfiling();
	void stopKernelProfiling();
	void buildKernelProfilingInfo();
	void getKernelGroupExecutionTimes(std::vector<cl::Event> eventList, std::string itemTitle, std::string totalsTitle);
	void addGlobalAndLocalWorkItemSizeInfo(cl::NDRange globalSizes, cl::NDRange localSizes);

	/// Smart shared pointer to init parameters
	std::shared_ptr<InitParameters> pInitParameters;

	/// Smart shared pointer to launch parameters
	std::shared_ptr<LaunchParameters> pLaunchParameters;

	/// Smart shared pointer to profile parameters
	std::shared_ptr<ProfileParameters> pProfileParameters;

	/// Wether selected queue supports kernel profiling
	bool profilingSupported = false;

        /// OpenCL Command queue
	cl::CommandQueue queue;

        /// This process' CL kernel
	cl::Kernel kernel;

	/// Name of the kernel function to be executed
	std::string kernelName;

	/// Storage for error strings
	std::string errStr;

        /// Vector with InfoItems data (list of pairs title, value storing profiling information).
	LPISupport::InfoItems infoItems;

        /// Vector with values of GPU execution times
	std::shared_ptr<LPISupport::SampleCollection> pSamplesGPUExecutionTime;

        /// Vector with values of GPU+CPU execution times
	std::shared_ptr<LPISupport::SampleCollection> pSamplesGPU_CPUExecutionTime;

        /// Vector with events for profiling kernel execuion times
	std::vector<cl::Event> eventsVector;


    private:
	/// Smart shared pointer to input Data object
	std::shared_ptr<Data> pInputData = nullptr;

        /// Handle associated to input data
	DataHandle inHandle = INVALIDDATAHANDLE;

	/// Smart shared pointer to output Data object
	std::shared_ptr<Data> pOutputData = nullptr;

        /// Handle associated to output data
	DataHandle outHandle = INVALIDDATAHANDLE;

	/// Event for kernel execution start (used for profiling kernel execution times).
	cl::Event start_ev;

        /// Event for kernel execution stop (used for profiling kernel execution times).
	cl::Event stop_ev;

        /// clock at CPU starting execution of host Process code
	std::chrono::high_resolution_clock::time_point beginCPUExecTime;
	
        /// clock at CPU ending execution of host Process code
	std::chrono::high_resolution_clock::time_point endCPUExecTime;
};
} /* namespace OpenCLIPER */
#endif // PROCESSCORE_HPP
