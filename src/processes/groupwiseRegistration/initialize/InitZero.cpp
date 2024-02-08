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
 * InitZero.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/initialize/InitZero.hpp>

// Uncomment to show class-specific debug messages
//#define INITZERO_DEBUG

#if !defined NDEBUG && defined INITZERO_DEBUG
    #define INITZERO_CERR(x) CERR(x)
#else
    #define INITZERO_CERR(x)
    #undef INITZERO_DEBUG
#endif

namespace OpenCLIPER {

void InitZero::init() {
    kernel = getApp()->getKernel("initZero");
}

void InitZero::launch() {
    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	uint numNDArrays = getOutput()->getNumNDArrays();

	for(uint i = 0; i < numNDArrays; i++) {
	    const cl::Buffer* object = getOutput()->getDeviceBuffer(i);
	    const std::vector<cl_uint> dataSize = *(getOutput()->getNDArray(i)->getDims());

	    int multi = 1;
	    for(int n : dataSize)
		multi *= n;

	    cl::NDRange globalWorkSize = cl::NDRange(multi);

	    kernel.setArg(0, *object);

	    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	    kernelsExecEventList.push_back(event);
	    //queue.finish();
	}
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::InitZero::launch kernel", "OpenCLIPER::InitZero::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "InitZero::launch");
    }
}

}

#undef INITZERO_DEBUG
