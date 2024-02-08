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
/*! CreateBB.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */

#include <OpenCLIPER/processes/groupwiseRegistration/creabspline/CreateBB.hpp>

// Uncomment to show class-specific debug messages
//#define CREATEBB_DEBUG

#if !defined NDEBUG && defined CREATEBB_DEBUG
    #define CREATEBB_CERR(x) CERR(x)
#else
    #define CREATEBB_CERR(x)
    #undef CREATEBB_DEBUG
#endif

namespace OpenCLIPER {

void CreateBB::init() {
    kernel = getApp()->getKernel("createBB");
}

void CreateBB::launch() {
    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* BBAuxData = getInput()->getDeviceBuffer();

	const cl::Buffer* auxBBData = getOutput()->getDeviceBuffer(auxBBx::auxBB);
	const cl::Buffer* auxBB1Data = getOutput()->getDeviceBuffer(auxBBx::auxBB1);
	const cl::Buffer* auxBB2Data = getOutput()->getDeviceBuffer(auxBBx::auxBB2);
	const cl::Buffer* auxBB11Data = getOutput()->getDeviceBuffer(auxBBx::auxBB11);

	// Need the dimensions of the largest auxBB buffer (it could be auxBB1 or auxBB2)
	const std::vector<cl_uint> dataSize = *(getOutput()->getNDArray(auxBBx::auxBB1)->getDims());
	cl::NDRange globalWorkSize = cl::NDRange(dataSize[0], dataSize[1], dataSize[2] * dataSize[3]);

	kernel.setArg(0, *auxBBData);
	kernel.setArg(1, *auxBB1Data);
	kernel.setArg(2, *auxBB2Data);
	kernel.setArg(3, *auxBB11Data);
	kernel.setArg(4, *BBAuxData);
	kernel.setArg(5, dataSize[2]);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	//queue.finish();
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::CreateBB::launch kernel", "OpenCLIPER::CreateBB::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "CreateBB::launch");
    }
}
}

#undef CREATEBB_DEBUG
