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
 * PermuteBBg.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/creabspline/PermuteBBg.hpp>

// Uncomment to show class-specific debug messages
//#define PERMUTEBBG_DEBUG

#if !defined NDEBUG && defined PERMUTEBBG_DEBUG
    #define PERMUTEBBG_CERR(x) CERR(x)
#else
    #define PERMUTEBBG_CERR(x)
    #undef PERMUTEBBG_DEBUG
#endif

namespace OpenCLIPER {

void PermuteBBg::init() {
    kernel = getApp()->getKernel("permuteBBg");
}

void PermuteBBg::launch() {
    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* aux1BB1gData = getInput()->getDeviceBuffer(aux1BBxg::aux1BB1g);
	const cl::Buffer* aux1BB2gData = getInput()->getDeviceBuffer(aux1BBxg::aux1BB2g);
	const cl::Buffer* aux1BB11gData = getInput()->getDeviceBuffer(aux1BBxg::aux1BB11g);

	const cl::Buffer* BB1gData = getOutput()->getDeviceBuffer(BBxg::BB1g);
	const cl::Buffer* BB2gData = getOutput()->getDeviceBuffer(BBxg::BB2g);
	const cl::Buffer* BB11gData = getOutput()->getDeviceBuffer(BBxg::BB11g);

	// Need the dimensions of the largest BBxg buffer (it could be BB1g or BB2g)
	const std::vector<cl_uint> dataSize = *(getOutput()->getNDArray(BBxg::BB1g)->getDims());
	cl::NDRange globalWorkSize = cl::NDRange(dataSize[0] * dataSize[1], dataSize[2] * dataSize[3], dataSize[4] * dataSize[5]);

	kernel.setArg(0, *BB1gData);
	kernel.setArg(1, *BB2gData);
	kernel.setArg(2, *BB11gData);
	kernel.setArg(3, *aux1BB1gData);
	kernel.setArg(4, *aux1BB2gData);
	kernel.setArg(5, *aux1BB11gData);
	kernel.setArg(6, dataSize[0]);
	kernel.setArg(7, dataSize[2]);
	kernel.setArg(8, dataSize[4]);
	kernel.setArg(9, dataSize[6]);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::PermuteBBg::launch kernel", "OpenCLIPER::PermuteBBg::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "PermuteBBg::launch");
    }
}
}

#undef PERMUTEBBG_DEBUG
