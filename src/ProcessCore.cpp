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
 * ProcessCore.cpp
 *
 *  Created on: 10 de nov. de 2016
 *      Author: manrod
 */
#include<OpenCLIPER/defs.hpp>
#include<OpenCLIPER/ProcessCore.hpp>
#include<OpenCLIPER/CLapp.hpp>


namespace OpenCLIPER {

/**
 * @brief Default constructor.
 *
 * Initializes collections storing device and device+host kernel execution time samples and binds CLapp object to this process
 *
 * @param[in] pCLapp pointer to CLapp object
 */
ProcessCore::ProcessCore(const std::shared_ptr<ProfileParameters>& pPP) {
    if(pPP)
        pProfileParameters = pPP;
    else
        pProfileParameters = std::make_shared<ProfileParameters>();

    pSamplesGPUExecutionTime = std::make_shared<LPISupport::SampleCollection>("Kernel execution time (s)");
    pSamplesGPU_CPUExecutionTime = std::make_shared<LPISupport::SampleCollection>("Total (host+device) execution time (s)");
}

/**
 * @brief Constructor for input and output objects to this process.
 *
 * Initializes collections storing device and device+host kernel execution time samples, and binds CLapp, input and output data objects
 * to this process.
 *
 * @param[in] pCLapp pointer to CLapp object
 * @param pInputData input data for the process
 * @param pOutData output data for the process
 */
ProcessCore::ProcessCore(const std::shared_ptr<Data>& pInputData, const std::shared_ptr<Data>& pOutData, const std::shared_ptr<ProfileParameters>& pPP) {
    if(pPP)
        pProfileParameters = pPP;
    else
        pProfileParameters = std::make_shared<ProfileParameters>();
    
    pSamplesGPUExecutionTime = std::make_shared<LPISupport::SampleCollection>("Kernel execution time (s)");
    pSamplesGPU_CPUExecutionTime = std::make_shared<LPISupport::SampleCollection>("Total (host+device) execution time (s)");
    setInput(pInputData);
    setOutput(pOutData);
}

/**
 * @brief Desctructor.
 *
 * Frees resources.
 *
 */
ProcessCore::~ProcessCore() {
    inHandle = INVALIDDATAHANDLE;
    outHandle = INVALIDDATAHANDLE;
    pInputData = nullptr;
    pOutputData = nullptr;
    pInitParameters = nullptr;
    pLaunchParameters = nullptr;
    pSamplesGPUExecutionTime = nullptr;
    pSamplesGPU_CPUExecutionTime = nullptr;
}

/**
 * @brief Returns a shared pointer to input data
 * @throw invalid_argument if pOutputData is null (nullptr)
 * @return shared pointer to input data
 */
std::shared_ptr<Data> ProcessCore::getInput() {
    if(pInputData == nullptr) {
	BTTHROW(std::invalid_argument("pInputData is null"), "ProcessCore::getInput");
    }
    return pInputData;
}

/**
 * @brief Returns a shared pointer to output data
 * @throw invalid_argument if pOutputData is null (nullptr)
 * @return shared pointer to output data
 */
std::shared_ptr<Data> ProcessCore::getOutput() {
    if(pOutputData == nullptr) {
	BTTHROW(std::invalid_argument("pOutputData is null"), "ProcessCore::getOutput");
    }
    return pOutputData;
}

/**
 * @brief ...
 *
 * @param pInputData ...
 * @return void
 */
void ProcessCore::setInput(std::shared_ptr<Data> pInputData) {
    this->pInputData = pInputData;
    if(pInputData != nullptr) {
	this->inHandle = pInputData->getHandle();
    }
}

/**
 * @brief ...
 *
 * @param pOutputData ...
 * @return void
 */
void ProcessCore::setOutput(std::shared_ptr<Data> pOutputData) {
    //CERR("ProcessCore::setOutput, arg pOutputData: " << pOutputData << std::endl);
    //CERR("ProcessCore::setOutput, this->pOutputData before assignment: " << this->pOutputData << std::endl << std::endl);
    this->pOutputData = pOutputData;
    if(pOutputData != nullptr) {
	this->outHandle = pOutputData->getHandle();
    }
}

/**
 *@brief  Method for testing common errors before launching kernel (null pointers to CLapp, inputData or outputData objects)
 */
void ProcessCore::checkCommonLaunchParameters() {
    if(getApp() == nullptr) {
	BTTHROW(std::invalid_argument(
	    "CLapp not assigned or not valid, launch aborted"), "ProcessCore::checkCommonLaunchParameters");
    }
    if(pInputData == nullptr) {
	BTTHROW(std::invalid_argument("inputData not assigned, launch aborted"), "ProcessCore::checkCommonLaunchParameters");
    }
    if(pOutputData == nullptr) {
	BTTHROW(std::invalid_argument("OutputData not assigned, launch aborted"), "ProcessCore::checkCommonLaunchParameters");
    }
    // Delete old profiling information
    infoItems.clear();
}

/**
 * @brief Method for testing if inputData object has its data stored also on device memory (otherwise, copy is requested)
 * @param[in] syncSource format used for storing data in device memory (buffers, images or both)
 */
void ProcessCore::checkXDataLaunchParameters(SyncSource syncSource) {
    if(pInputData->getDeviceBuffer() == nullptr) {
    }
}

/**
 * @brief Starts host and device code profiling if selected device supports profiling and it is enabled
 */
void ProcessCore::startProfiling() {
    startHostCodeProfiling();
    startKernelProfiling();
}

/**
 * @brief Stops host and device code profiling
 */
void ProcessCore::stopProfiling() {
    stopKernelProfiling();
    stopHostCodeProfiling();
}

/**
 * @brief Starts host code profiling if selected device supports profiling and it is enabled
 */
void ProcessCore::startHostCodeProfiling() {
    if(pProfileParameters->enable && profilingSupported) {
	beginCPUExecTime = std::chrono::high_resolution_clock::now();
    }
}

/**
 * @brief Stops host code profiling
 */
void ProcessCore::stopHostCodeProfiling() {
    if(pProfileParameters->enable && profilingSupported) {
	endCPUExecTime = std::chrono::high_resolution_clock::now();
	TIME_DIFF_TYPE elapsedTime =
	    (std::chrono::duration_cast<std::chrono::nanoseconds>(endCPUExecTime - beginCPUExecTime).count()) / 1e9;
	pSamplesGPU_CPUExecutionTime->appendSample(elapsedTime);
    }
}

/**
 * @brief Starts kernel profiling if selected device supports profiling and it is enabled
 */
void ProcessCore::startKernelProfiling() {
    if(pProfileParameters->enable && profilingSupported) {
	getApp()->getCommandQueue().enqueueMarkerWithWaitList(NULL, &start_ev);
    }
}

/**
 * @brief Stops kernel profiling
 */
void ProcessCore::stopKernelProfiling() {
    if(pProfileParameters->enable && profilingSupported) {
	getApp()->getCommandQueue().enqueueMarkerWithWaitList(NULL, &stop_ev);
	stop_ev.wait();
	cl_ulong ev_start_time = (cl_ulong) 0;
	cl_ulong ev_stop_time = (cl_ulong) 0;
	ev_start_time = start_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	ev_stop_time = stop_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	double elapsedTInSeg = (ev_stop_time - ev_start_time) / 1e9;
	pSamplesGPUExecutionTime->appendSample(elapsedTInSeg);
    }
}

/**
 * @brief Store kernel execution times for several kernel executions in a SampleCollection class variable.
 *
 * Start and stop time for every kernel execution is got from events associated to kernels, and the diference between these two times
 * is stored as a value in the collection of kernel execution time values (pSamplesGPUExecutionTime collection).
supported by device).
 */
void ProcessCore::buildKernelProfilingInfo() {
    queue.finish();
    if(pProfileParameters->enable && profilingSupported) {
	for(unsigned int i = 0; i < eventsVector.size(); i++) {
        double startTime, stopTime, elapsedTimeInSeg;
	    stopTime = eventsVector.at(i).getProfilingInfo<CL_PROFILING_COMMAND_END>();
	    startTime = eventsVector.at(i).getProfilingInfo<CL_PROFILING_COMMAND_START>();
	    elapsedTimeInSeg = (stopTime - startTime)  / 1e9; // start and stop times in nanoseconds
	    pSamplesGPUExecutionTime->appendSample(elapsedTimeInSeg);
	}
    }
}

/**
 * @brief Calculates total execution time of a group of kernels (since profiling has been started to profiling has been ended) and stores
 * this information in a variable of type InfoItems
 * @param[in] kernelsExecEventList vector of events related to kernels execution
 * @param[in] itemTitle title of item of information for one kernel
 * @param[in] totalsTitle title for the total elapsed time
 */
void ProcessCore::getKernelGroupExecutionTimes(std::vector<cl::Event> kernelsExecEventList,
	std::string itemTitle, std::string totalsTitle) {
    if(profilingSupported) {
	double totalElapsedTimeInSeg = 0.0;
	for(unsigned int i = 0; i < kernelsExecEventList.size(); i++) {
	    double startTime, stopTime, elapsedTimeInSeg;
	    stopTime = kernelsExecEventList.at(i).getProfilingInfo<CL_PROFILING_COMMAND_END>();
	    startTime = kernelsExecEventList.at(i).getProfilingInfo<CL_PROFILING_COMMAND_START>();
	    elapsedTimeInSeg = (stopTime - startTime)  / 1e9; // start and stop times in nanoseconds
	    totalElapsedTimeInSeg += elapsedTimeInSeg;
	    infoItems.addInfoItem(itemTitle + " " + std::to_string(i) + " start time", std::to_string(startTime));
	    infoItems.addInfoItem(itemTitle + " " + std::to_string(i) + " stop time ", std::to_string(stopTime));
	    std::ostringstream outputstream;
	    outputstream << std::fixed << std::setprecision(PROFILINGTIMESPRECISION) << elapsedTimeInSeg;
	    infoItems.addInfoItem(itemTitle + " " + std::to_string(i) + " execution time (s)", outputstream.str());
	}
	if(kernelsExecEventList.size() > 0) {
	    std::ostringstream outputstream;
	    outputstream << std::fixed << std::setprecision(PROFILINGTIMESPRECISION) << totalElapsedTimeInSeg;
	    infoItems.addInfoItem(totalsTitle + " execution time (s)", outputstream.str());
	}
    }
}

/**
 * @brief Adds info about global and local kernel sizes to infoItems class variable
 *
 * @param[in] globalSizes kernel global sizes
 * @param[in] localSizes kernel local sizes
 */
void ProcessCore::addGlobalAndLocalWorkItemSizeInfo(cl::NDRange globalSizes, cl::NDRange localSizes) {
    if(pProfileParameters->enable) {
	std::string globalSizeString = "globalSize[";
	std::string endBracketString = "]";
	for(cl_uint i = 0; i < globalSizes.dimensions(); i++) {
	    infoItems.addInfoItem(
		globalSizeString + std::to_string(i) + endBracketString, std::to_string(globalSizes.get()[i]));
	}
	std::string localSizeString = "localSize[";
	if(localSizes.dimensions() == 0) {
	    for(cl_uint i = 0; i < getApp()->getDevice().getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>(); i++) {
		infoItems.addInfoItem(localSizeString + std::to_string(i) + endBracketString, std::to_string(0));
	    }
	}
	for(cl_uint i = 0; i < localSizes.dimensions(); i++) {
	    infoItems.addInfoItem(localSizeString + std::to_string(i) + endBracketString, std::to_string(localSizes.get()[i]));
	}
    }
}
} /* namespace OpenCLIPER */
