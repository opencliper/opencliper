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
 * Process.cpp
 *
 *  Created on: 10 de nov. de 2016
 *      Author: manrod
 */
#include<OpenCLIPER/Process.hpp>
#include<OpenCLIPER/CLapp.hpp>

// Uncomment to show class-specific debug messages
#define PROCESS_DEBUG

#if !defined NDEBUG && defined CLAPP_DEBUG
    #define PROCESS_CERR(x) CERR(x)
#else
    #define PROCESS_CERR(x)
    #undef PROCESS_DEBUG
#endif

namespace OpenCLIPER {

Process::Process(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP): ProcessCore(pPP) {
    setApp(pCLapp);
    queue = pCLapp->getCommandQueue();
}

Process::Process(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data>& pInputData, const std::shared_ptr<Data>& pOutputData,
                 const std::shared_ptr<ProfileParameters>& pPP): ProcessCore(pInputData, pOutputData, pPP) {

    setApp(pCLapp);
    queue = pCLapp->getCommandQueue();
}

/**
* @brief Get value of shared pointer to CLapp process assigned to this object
* @return shared pointer to CLapp object
*/
const std::shared_ptr<CLapp> Process::getApp() const {
    if(pCLapp != nullptr)
        return pCLapp;
    else
        BTTHROW(std::runtime_error("getApp() called before setApp()"), "Process::getApp()");
}


/**
 * @brief Binds CLapp object to this Process object.
 *
 * This method is automatically called from CLapp object when a process object is added to the CLapp object
 * using CLapp::addProcess method.
 * @param[in] pCLapp smart shared pointer to CLapp object
 */
void Process::setApp(const std::shared_ptr<CLapp>& pCLapp) {
    if(!pCLapp) {
    	std::stringstream strstr;
    	strstr << "Invalid CLapp pointer: " << pCLapp.get();
        BTTHROW(std::invalid_argument(strstr.str()), "Process::setApp");
    }
    this->pCLapp = pCLapp;

    cl_command_queue_properties queue_properties;
    queue_properties = pCLapp->getDevice().getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
    if(queue_properties & CL_QUEUE_PROFILING_ENABLE)
	profilingSupported = true;
    else
	profilingSupported = false;
}

const std::string Process::getKernelFile() const {
    // A process may not need a kernel (e.g. if it just calls other processes). Let them fall back to this method instead of forcing them to reimplement it
    return "";
}

} // namespace OpenCLIPER

#undef PROCESS_DEBUG
