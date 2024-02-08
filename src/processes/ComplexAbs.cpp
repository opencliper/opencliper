/* Copyright (C) 2018 Federico Simmross Wattenberg,
 *                    Manuel Rodríguez Cayetano,
 *                    Javier Royuela del Val,
 *                    Elena Martín González,
 *                    Elisa Moya Sáez,
 *                    Marcos Martín Fernández and
 *                    Carlos Alberola López
 *                    Emilio López-Ales
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
 * ComplexAbs.cpp
 *
 *  Created on: 15 de oct. de 2021
 *      Author: Emilio López-Ales
 */

#include <OpenCLIPER/processes/ComplexAbs.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <libgen.h>

namespace OpenCLIPER {


void ComplexAbs::init() {
	kernel = getApp()->getKernel("complexAbs");
}

void ComplexAbs::launch() {
	startProfiling();
	try {
		std::vector<cl::Event> kernelsExecEventList;
		cl::Event event;

		const cl::Buffer* pInputBuffer = getInput()->getDeviceBuffer();
		const cl::Buffer* pOutputBuffer = getOutput()->getDeviceBuffer();

		cl::NDRange globalWorkSize = cl::NDRange(NDARRAYWIDTH(getInput()->getNDArray(0)), NDARRAYHEIGHT(getInput()->getNDArray(0)), NDARRAYDEPTH(getInput()->getNDArray(0)));

		kernel.setArg(0, *pInputBuffer);
		kernel.setArg(1, *pOutputBuffer);

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
		kernelsExecEventList.push_back(event);

		queue.finish();

		stopProfiling();
		if(pProfileParameters->enable)
			getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::ComplexAbs::launch kernel", "OpenCLIPER::ComplexAbs::launch group of kernels");
	}
	catch(cl::Error& err) {
		BTTHROW(CLError(err), "ComplexAbs::launch");
	}
}

}
