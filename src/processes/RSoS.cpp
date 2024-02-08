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
#include <OpenCLIPER/processes/RSoS.hpp>
#include <OpenCLIPER/CLapp.hpp>
//#include <OpenCLIPER/OpenCLIPER_devil.hpp>
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/KData.hpp>
#include <LPISupport/InfoItems.hpp>

#include <iostream>

#define KERNELCOMPILEOPTS "-I../include/"
//#define KERNELCOMPILEOPTS "-cl-std=CL2.0 -I../include/ -g"
#define CLASSNAME "OpenCLIPER::XImageSum"

// Uncomment to show class-specific debug messages
// #define RSOS_DEBUG

#if !defined NDEBUG && defined RSOS_DEBUG
    #define RSOS_CERR(x) CERR(x)
#else
    #define RSOS_CERR(x)
    #undef RSOS_DEBUG
#endif

namespace OpenCLIPER {

void RSoS::init() {
    kernel = getApp()->getKernel("rss_kernel");
}


void RSoS::launch() {
    checkCommonLaunchParameters();

    infoItems.addInfoItem("Title", "RSoS info");
    BEGIN_TIME(beginTime);
    startKernelProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Buffer* pInputBuffer;
	cl::Buffer* pOutputBuffer;
	cl_ulong globalWorkSize = 1;
	cl::Event event;

	RSOS_CERR("pInputData->getData()->size: " << getInput()->getData()->size() << std::endl);

	if(getInput()->getData()->size() == 0) {
	    BTTHROW(std::invalid_argument(std::string(CLASSNAME) + std::string("::launch: inputData size is 0")), "RSoS::launch");
	}

	// First NDArray device buffer is a pointer to a contiguous device memory for all the NDArrays
	pInputBuffer = getInput()->getDeviceBuffer();
	pOutputBuffer = getOutput()->getDeviceBuffer();

	RSOS_CERR("Starting kernel ... " << std::endl);

	kernel.setArg(0, *pInputBuffer);
	kernel.setArg(1, *pOutputBuffer);
	cl::NDRange globalSizes = cl::NDRange(NDARRAYWIDTH(getInput()->getData()->at(0)), NDARRAYHEIGHT(getInput()->getData()->at(0)),
					      (std::dynamic_pointer_cast<KData>(getInput()))->getDynDimsTotalSize());
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSizes,  cl::NullRange, NULL, &event);
	stopKernelProfiling();
	if(pProfileParameters->enable) {
	    END_TIME(endTime);
	    TIME_DIFF_TYPE elapsedTime;
	    TIME_DIFF(elapsedTime, beginTime, endTime);
	    std::ostringstream ostream;
	    ostream << std::fixed << std::setprecision(PROFILINGTIMESPRECISION) << elapsedTime;
	    infoItems.addInfoItem("Number of work items", std::to_string(globalWorkSize));
	    infoItems.addInfoItem("Total (host+device) RSoS time (s)", ostream.str());
	    if(profilingSupported) {
		getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::RSoS::launch kernel",
					     "OpenCLIPER::RSoS::launch group of kernels");
	    }
	}

	RSOS_CERR("RSoS finished." << std::endl);
    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "RSoS::launch");
    }
}
} /* namespace OpenCLIPER */
