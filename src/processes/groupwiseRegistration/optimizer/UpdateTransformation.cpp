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
 * UpdateTransformation.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/UpdateTransformation.hpp>

// Uncomment to show class-specific debug messages
//#define UPDATETRANSFORMATION_DEBUG

#if !defined NDEBUG && defined UPDATETRANSFORMATION_DEBUG
    #define UPDATETRANSFORMATION_CERR(x) CERR(x)
#else
    #define UPDATETRANSFORMATION_CERR(x)
    #undef UPDATETRANSFORMATION_DEBUG
#endif

namespace OpenCLIPER {

void UpdateTransformation::init() {
    kernel = getApp()->getKernel("updateTransformation");
}

void UpdateTransformation::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* DifData = (pLP->DifData)->getDeviceBuffer();

	const cl::Buffer* TData = getOutput()->getDeviceBuffer();
	const cl::Buffer* dHData = getInput()->getDeviceBuffer();

	const std::vector<cl_uint> dataSize = *(getInput()->getNDArray(0)->getDims());
	cl::NDRange globalWorkSize = cl::NDRange(dataSize[0], dataSize[1], dataSize[2] * dataSize[3]);

	kernel.setArg(0, *TData);
	kernel.setArg(1, *DifData);
	kernel.setArg(2, *dHData);
	kernel.setArg(3, pLP->Wnobj);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::UpdateTransformation::launch kernel", "OpenCLIPER::UpdateTransformation::launch group of kernels");

    }
    catch(cl::Error& err) {
	BTTHROW(CLError(err), "XImageSum::launch");
    }
    catch(std::exception& err) {
	BTTHROW(err, "UpdateTransformation::launch");
    }
}
}

#undef UPDATETRANSFORMATION_DEBUG
