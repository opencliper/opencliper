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
/*! ControlPointsLocation.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/creabspline/ControlPointsLocation.hpp>

// Uncomment to show class-specific debug messages
//#define CONTROLPOINTSLOCATION_DEBUG

#if !defined NDEBUG && defined CLAPP_DEBUG
    #define CONTROLPOINTSLOCATION_CERR(x) CERR(x)
#else
    #define CONTROLPOINTSLOCATION_CERR(x)
    #undef CONTROLPOINTSLOCATION_DEBUG
#endif

namespace OpenCLIPER {

void ControlPointsLocation::init() {
    kernel = getApp()->getKernel("controlPointsLocation");
}

void ControlPointsLocation::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* puData = getOutput()->getDeviceBuffer();

	const std::vector<cl_uint> dataSize = *(getOutput()->getNDArray(0)->getDims());
	cl::NDRange globalWorkSize = cl::NDRange(dataSize[0], dataSize[1]);

	kernel.setArg(0, *puData);
	kernel.setArg(1, pLP->C00);
	kernel.setArg(2, pLP->C10);
	kernel.setArg(3, pLP->c);
	kernel.setArg(4, pLP->Dp);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	//queue.finish();
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::ControlPointsLocation::launch kernel", "OpenCLIPER::ControlPointsLocation::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "ControlPointsLocation::launch");
    }

}
}

#undef CONTROLPOINTSLOCATION_DEBUG
