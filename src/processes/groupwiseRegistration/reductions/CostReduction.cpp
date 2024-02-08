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
 * CostReduction.cpp
 *
 *  Created on: Oct 15, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/reductions/CostReduction.hpp>

// Uncomment to show class-specific debug messages
//#define COSTREDUCTION_DEBUG

#if !defined NDEBUG && defined COSTREDUCTION_DEBUG
    #define COSTREDUCTION_CERR(x) CERR(x)
#else
    #define COSTREDUCTION_CERR(x)
    #undef COSTREDUCTION_DEBUG
#endif

namespace OpenCLIPER {

void CostReduction::init() {
    kernel = getApp()->getKernel("costReduction");
}

void CostReduction::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* VData = getInput()->getDeviceBuffer();
	const cl::Buffer* HData = getOutput()->getDeviceBuffer();

	const std::vector<cl_uint> dataSize = *(getInput()->getNDArray(0)->getDims());
	cl::NDRange globalWorkSize = cl::NDRange(dataSize[0] * dataSize[1]);

	kernel.setArg(0, *HData);
	kernel.setArg(1, *VData);
	kernel.setArg(2, *(pLP->maskData)->getDeviceBuffer());
	kernel.setArg(3, pLP->iter);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::CostReduction::launch kernel", "OpenCLIPER::CostReduction::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "CostReduction::launch");
    }
}
}

#undef COSTREDUCTION_DEBUG
