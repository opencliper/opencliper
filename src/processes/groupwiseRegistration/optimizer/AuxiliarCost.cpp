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
 * AuxiliarCost.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/AuxiliarCost.hpp>

// Uncomment to show class-specific debug messages
//#define AUXILIARCOST_DEBUG

#if !defined NDEBUG && defined AUXILIARCOST_DEBUG
    #define AUXILIARCOST_CERR(x) CERR(x)
#else
    #define AUXILIARCOST_CERR(x)
    #undef AUXILIARCOST_DEBUG
#endif

namespace OpenCLIPER {

void AuxiliarCost::init() {
    kernel = getApp()->getKernel("auxiliarCost");
}

void AuxiliarCost::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* XAuxData = (pLP->XauxData)->getDeviceBuffer();
	const cl::Buffer* cost1Data = getOutput()->getDeviceBuffer();

	const cl::Buffer* dVpermutedData = getInput()->getDeviceBuffer(1);

	const std::vector<cl_uint> dataSize = *(getInput()->getNDArray(0)->getDims());
	cl::NDRange globalWorkSize = cl::NDRange(dataSize[0] * dataSize[1], dataSize[2], dataSize[5]);

	kernel.setArg(0, *cost1Data);
	kernel.setArg(1, *dVpermutedData);
	kernel.setArg(2, *XAuxData);
	kernel.setArg(3, pLP->l);
	kernel.setArg(4, pLP->k);
	kernel.setArg(5, dataSize[0]);
	kernel.setArg(6, dataSize[3]);
	kernel.setArg(7, dataSize[4]);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::AuxiliarCost::launch kernel", "OpenCLIPER::AuxiliarCost::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "AuxiliarCost::launch");
    }
}
}

#undef AUXILIARCOST_DEBUG
