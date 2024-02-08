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
 * Gradient.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/Gradient.hpp>

// Uncomment to show class-specific debug messages
//#define GRADIENT_DEBUG

#if !defined NDEBUG && defined GRADIENT_DEBUG
    #define GRADIENT_CERR(x) CERR(x)
#else
    #define GRADIENT_CERR(x)
    #undef GRADIENT_DEBUG
#endif

namespace OpenCLIPER {

void Gradient::init() {
    kernel = getApp()->getKernel("gradient");
}

void Gradient::launch() {
    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* inputData = getInput()->getDeviceBuffer(); // Input image or video
	const cl::Buffer* d1Data = getOutput()->getDeviceBuffer(d::d1); // Horizontal
	const cl::Buffer* d2Data = getOutput()->getDeviceBuffer(d::d2); // Vertical

	const std::vector<cl_uint> dataSize = *(getInput()->getNDArray(0)->getDims());
	cl::NDRange globalWorkSize = cl::NDRange(dataSize[0], dataSize[1], getInput()->getNumNDArrays());

	kernel.setArg(0, *d1Data);  // Horizontal
	kernel.setArg(1, *d2Data);  // Vertical
	kernel.setArg(2, *inputData);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::Gradient::launch kernel", "OpenCLIPER::Gradient::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "Gradient::launch");
    }
}
}

#undef GRADIENT_DEBUG
