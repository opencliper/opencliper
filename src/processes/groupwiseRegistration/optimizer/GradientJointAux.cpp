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
 * GradientJointAux.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/GradientJointAux.hpp>

// Uncomment to show class-specific debug messages
//#define GRADIENTJOINTAUX_DEBUG

#if !defined NDEBUG && defined GRADIENTJOINTAUX_DEBUG
    #define GRADIENTJOINTAUX_CERR(x) CERR(x)
#else
    #define GRADIENTJOINTAUX_CERR(x)
    #undef GRADIENTJOINTAUX_DEBUG
#endif

namespace OpenCLIPER {

void GradientJointAux::init() {
    kernel = getApp()->getKernel("gradientJointAux");
}

void GradientJointAux::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* aux2Data = getInput()->getDeviceBuffer();
	const cl::Buffer* AuxData = getOutput()->getDeviceBuffer();

	const std::vector<cl_uint> dataSize = *(getOutput()->getNDArray(0)->getDims());

	cl::NDRange globalWorkSize = cl::NDRange(dataSize[0] * dataSize[1], dataSize[2], dataSize[3] * dataSize[4]);

	kernel.setArg(0, *AuxData);
	kernel.setArg(1, *aux2Data);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::GradientJointAux::launch kernel", "OpenCLIPER::GradientJointAux::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "GradientJointAux::launch");
    }
}

}

#undef GRADIENTJOINTAUX_DEBUG
