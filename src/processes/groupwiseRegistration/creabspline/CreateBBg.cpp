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
 * CreateBBg.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/creabspline/CreateBBg.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <iostream>

// Uncomment to show class-specific debug messages
//#define CREATEBBG_DEBUG

#if !defined NDEBUG && defined CREATEBBG_DEBUG
    #define CREATEBBG_CERR(x) CERR(x)
#else
    #define CREATEBBG_CERR(x)
    #undef CREATEBBG_DEBUG
#endif

namespace OpenCLIPER {

void CreateBBg::init() {
    kernel = getApp()->getKernel("createBBg");
}

void CreateBBg::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* BBgData = (pLP->BBgallData)->getDeviceBuffer();

	const cl::Buffer* BBAux1Data = getInput()->getDeviceBuffer();
	const cl::Buffer* auxBB1gData = getOutput()->getDeviceBuffer(auxBBxg::auxBB1g);
	const cl::Buffer* auxBB2gData = getOutput()->getDeviceBuffer(auxBBxg::auxBB2g);
	const cl::Buffer* auxBB11gData = getOutput()->getDeviceBuffer(auxBBxg::auxBB11g);

	const std::vector<cl_uint> dataSize = *(getOutput()->getNDArray(auxBBxg::auxBB1g)->getDims());
	cl::NDRange globalWorkSize = cl::NDRange(dataSize[0], dataSize[1], dataSize[2] * dataSize[3]);

	kernel.setArg(0, *BBgData);
	kernel.setArg(1, *auxBB1gData);
	kernel.setArg(2, *auxBB2gData);
	kernel.setArg(3, *auxBB11gData);
	kernel.setArg(4, *BBAux1Data);
	kernel.setArg(5, pLP->Dp);
	kernel.setArg(6, dataSize[2]);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	//queue.finish();
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::CreateBBg::launch kernel", "OpenCLIPER::CreateBBg::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "CreateBBg::launch");
    }

}
}

#undef CREATEBBG_DEBUG
