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
 * Cost6D.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/Cost6D.hpp>

// Uncomment to show class-specific debug messages
//#define COST6D_DEBUG

#if !defined NDEBUG && defined COST6D_DEBUG
    #define COST6D_CERR(x) CERR(x)
#else
    #define COST6D_CERR(x)
    #undef COST6D_DEBUG
#endif

namespace OpenCLIPER {

void Cost6D::init() {
    kernel = getApp()->getKernel("cost6D");
}

void Cost6D::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* cost2Data = getInput()->getDeviceBuffer();
	const cl::Buffer* dHData = getOutput()->getDeviceBuffer();

	const std::vector<cl_uint> dataSize = *(getOutput()->getNDArray(0)->getDims());
	cl::NDRange globalWorkSize = cl::NDRange(dataSize[2], dataSize[3]);

	kernel.setArg(0, *dHData);
	kernel.setArg(1, *cost2Data);
	kernel.setArg(2, pLP->l);
	kernel.setArg(3, pLP->k);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::Cost6D::launch kernel", "OpenCLIPER::Cost6D::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "Cost6D::launch");
    }
}

}

#undef COST6D_DEBUG
