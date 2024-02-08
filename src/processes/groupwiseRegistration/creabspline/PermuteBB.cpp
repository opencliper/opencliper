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
 * PermuteBB.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/creabspline/PermuteBB.hpp>

// Uncomment to show class-specific debug messages
//#define PERMUTEBB_DEBUG

#if !defined NDEBUG && defined PERMUTEBB_DEBUG
    #define PERMUTEBB_CERR(x) CERR(x)
#else
    #define PERMUTEBB_CERR(x)
    #undef PERMUTEBB_DEBUG
#endif

namespace OpenCLIPER {

void PermuteBB::init() {
    kernel = getApp()->getKernel("permuteBB");
}

void PermuteBB::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* auxBBData = getInput()->getDeviceBuffer(auxBBx::auxBB);
	const cl::Buffer* auxBB1Data = getInput()->getDeviceBuffer(auxBBx::auxBB1);
	const cl::Buffer* auxBB2Data = getInput()->getDeviceBuffer(auxBBx::auxBB2);
	const cl::Buffer* auxBB11Data = getInput()->getDeviceBuffer(auxBBx::auxBB11);

	const cl::Buffer* BBData = getOutput()->getDeviceBuffer(BBx::BB);
	const cl::Buffer* BB1Data = getOutput()->getDeviceBuffer(BBx::BB1);
	const cl::Buffer* BB2Data = getOutput()->getDeviceBuffer(BBx::BB2);
	const cl::Buffer* BB11Data = getOutput()->getDeviceBuffer(BBx::BB11);

	const std::vector<cl_uint> dataSize = *(getOutput()->getNDArray(BBx::BB1)->getDims());
	cl::NDRange globalWorkSize = cl::NDRange(dataSize[0] * dataSize[1], dataSize[2] * dataSize[3], dataSize[4] * dataSize[5]);

	kernel.setArg(0, *BBData);
	kernel.setArg(1, *BB1Data);
	kernel.setArg(2, *BB2Data);
	kernel.setArg(3, *BB11Data);
	kernel.setArg(4, *auxBBData);
	kernel.setArg(5, *auxBB1Data);
	kernel.setArg(6, *auxBB2Data);
	kernel.setArg(7, *auxBB11Data);
	kernel.setArg(8, pLP->Dpobj);
	kernel.setArg(9, dataSize[0]);
	kernel.setArg(10, dataSize[2]);
	kernel.setArg(11, dataSize[4]);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	//queue.finish();
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::PermuteBB::launch kernel", "OpenCLIPER::PermuteBB::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "PermuteBB::launch");
    }
}
}

#undef PERMUTEBB_DEBUG
