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
 * Reduction5Dto4D.cpp
 *
 *  Created on: Oct 15, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/reductions/ReductionSum.hpp>

// Uncomment to show class-specific debug messages
//#define REDUCTIONSUM_DEBUG

#if !defined NDEBUG && defined REDUCTIONSUM_DEBUG
    #define REDUCTIONSUM_CERR(x) CERR(x)
#else
    #define REDUCTIONSUM_CERR(x)
    #undef REDUCTIONSUM_DEBUG
#endif

namespace OpenCLIPER {

void ReductionSum::init() {
    kernel = getApp()->getKernel("reductionsum");
}

void ReductionSum::launch() {

    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* input = getInput()->getDeviceBuffer();
	const cl::Buffer* output = getOutput()->getDeviceBuffer();

	const std::vector<cl_uint> dataSize = *(getOutput()->getNDArray(0)->getDims());

	int multi = 1;
	for(int n : dataSize)
	    multi *= n;

	cl::NDRange globalWorkSize = cl::NDRange(multi);

	kernel.setArg(0, *output);
	kernel.setArg(1, *input);
	kernel.setArg(2, pLP->dim2sum);


	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::ReductionSum::launch kernel", "OpenCLIPER::ReductionSum::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "ReductionSum::launch");
    }
}
}

#undef REDUCTIONSUM_DEBUG
