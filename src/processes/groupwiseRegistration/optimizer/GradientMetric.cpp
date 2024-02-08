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
 * GradientMetric.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/GradientMetric.hpp>

// Uncomment to show class-specific debug messages
//#define GRADIENTMETRIC_DEBUG

#if !defined NDEBUG && defined GRADIENTMETRIC_DEBUG
    #define GRADIENTMETRIC_CERR(x) CERR(x)
#else
    #define GRADIENTMETRIC_CERR(x)
    #undef GRADIENTMETRIC_DEBUG
#endif

namespace OpenCLIPER {

void GradientMetric::init() {
    kernel = getApp()->getKernel("gradientMetric");
}

void GradientMetric::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* inputData = getInput()->getDeviceBuffer();
	const cl::Buffer* dyData = getOutput()->getDeviceBuffer();

	const std::vector<cl_uint> dataSize = *(getInput()->getNDArray(0)->getDims());
	uint numFrames = getInput()->getNumNDArrays();
	cl::NDRange globalWorkSize = cl::NDRange(dataSize[0], dataSize[1], numFrames);

	float factor = 2.0 / numFrames;

	kernel.setArg(0, *dyData);
	kernel.setArg(1, *inputData);
	kernel.setArg(2, factor);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::GradientMetric::launch kernel", "OpenCLIPER::GradientMetric::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "GradientMetric::launch");
    }
}

}

#undef GRADIENTMETRIC_DEBUG
