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
 * PermuteTAux.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/PermuteTAux.hpp>

// Uncomment to show class-specific debug messages
//#define PERMUTETAUX_DEBUG

#if !defined NDEBUG && defined PERMUTETAUX_DEBUG
    #define PERMUTETAUX_CERR(x) CERR(x)
#else
    #define PERMUTETAUX_CERR(x)
    #undef PERMUTETAUX_DEBUG
#endif

namespace OpenCLIPER {

void PermuteTAux::init() {
    kernel = getApp()->getKernel("permuteTAux");
}

void PermuteTAux::launch() {
    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* TAuxPermData = getOutput()->getDeviceBuffer(TAux::permuted);
	const cl::Buffer* TAuxData = getInput()->getDeviceBuffer(TAux::original);

	const std::vector<cl_uint> dataSize = *(getInput()->getNDArray(TAux::original)->getDims());
	cl::NDRange globalWorkSize = cl::NDRange(dataSize[1] * dataSize[0], dataSize[2] * dataSize[3], dataSize[4] * dataSize[5]);

	kernel.setArg(0, *TAuxPermData);
	kernel.setArg(1, *TAuxData);
	kernel.setArg(2, dataSize[1]);
	kernel.setArg(3, dataSize[2]);
	kernel.setArg(4, dataSize[4]);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::PermuteTAux::launch kernel", "OpenCLIPER::PermuteTAux::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "PermuteTAux::launch");
    }
}
}

#undef PERMUTETAUX_DEBUG
