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
 * Reduction7Dto6D.cpp
 *
 *  Created on: Oct 15, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/reductions/Reduction7Dto6D.hpp>

// Uncomment to show class-specific debug messages
//#define REDUCTION7DTO6D_DEBUG

#if !defined NDEBUG && defined REDUCTION7DTO6D_DEBUG
    #define REDUCTION7DTO6D_CERR(x) CERR(x)
#else
    #define REDUCTION7DTO6D_CERR(x)
    #undef REDUCTION7DTO6D_DEBUG
#endif

namespace OpenCLIPER {

void Reduction7Dto6D::init() {
    kernel = getApp()->getKernel("reduction7Dto6D");
}

void Reduction7Dto6D::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* auxData = getOutput()->getDeviceBuffer();

	const cl::Buffer* dthetaxData = getInput()->getDeviceBuffer(dtheta::dthetax);
	const cl::Buffer* dthetax2Data = getInput()->getDeviceBuffer(dtheta::dthetax2);
	const cl::Buffer* dthetaxyData = getInput()->getDeviceBuffer(dtheta::dthetaxy);
	const cl::Buffer* dthetatData = getInput()->getDeviceBuffer(dtheta::dthetat);
	const cl::Buffer* dthetat2Data = getInput()->getDeviceBuffer(dtheta::dthetat2);

	const std::vector<cl_uint> dataSize = *(getInput()->getNDArray(dtheta::dthetax)->getDims());
	cl::NDRange globalWorkSize = cl::NDRange(dataSize[0] * dataSize[1] * dataSize[2] * dataSize[3] * dataSize[4] * dataSize[5]);

	kernel.setArg(0, *auxData);
	kernel.setArg(1, *dthetaxData);
	kernel.setArg(2, *dthetax2Data);
	kernel.setArg(3, *dthetaxyData);
	kernel.setArg(4, *dthetatData);
	kernel.setArg(5, *dthetat2Data);
	kernel.setArg(6, pLP->lambda0);
	kernel.setArg(7, pLP->lambda1);
	kernel.setArg(8, pLP->lambda2);
	kernel.setArg(9, pLP->lambda3);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::Reduction7Dto6D::launch kernel", "OpenCLIPER::Reduction7Dto6D::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "Reduction7Dto6D::launch");
    }
}
}

#undef REDUCTION7DTO6D_DEBUG
