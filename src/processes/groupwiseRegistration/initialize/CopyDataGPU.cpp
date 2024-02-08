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
 * CopyDataGPU.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/initialize/CopyDataGPU.hpp>

// Uncomment to show class-specific debug messages
//#define COPYDATAGPU_DEBUG

#if !defined NDEBUG && defined COPYDATAGPU_DEBUG
    #define COPYDATAGPU_CERR(x) CERR(x)
#else
    #define COPYDATAGPU_CERR(x)
    #undef COPYDATAGPU_DEBUG
#endif

namespace OpenCLIPER {

void CopyDataGPU::init() {
    kernel = getApp()->getKernel("copyDataGPU");
}

void CopyDataGPU::launch() {
    checkCommonLaunchParameters();
    try {
	const cl::Buffer* inputData = getInput()->getDeviceBuffer();
	const cl::Buffer* outputData = getOutput()->getDeviceBuffer();

	cl_uint dataSize = NDARRAYWIDTH(getInput()->getNDArray(0)) * NDARRAYHEIGHT(getInput()->getNDArray(0)) * NDARRAYDEPTH(getInput()->getNDArray(0)) * getInput()->getDynDimsTotalSize();
	cl::NDRange globalWorkSize = cl::NDRange(dataSize);

	kernel.setArg(0, *outputData);
	kernel.setArg(1, *inputData);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, NULL);
    }
    catch(cl::Error& err) {
	BTTHROW(CLError(err), "CopyDataGPU::launch");
    }
}
} //namespace OpenCLIPER

#undef COPYDATAGPU_DEBUG
