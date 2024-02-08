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
 * TemporalTV.cpp
 *
 *  Created on: 20 de nov. de 2017
 *      Author: Elisa Moya-Saez
 * 
 *  Modified on: 29 de oct. de 2021
 *      Author: Emilio López-Ales
 */

#include <OpenCLIPER/processes/nesta/TemporalTV.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <libgen.h>

namespace OpenCLIPER {

void TemporalTV::init() {
	auto pIP = std::dynamic_pointer_cast<InitParameters>(pInitParameters);

	if(pIP->dir == FORWARD)
		kernel = getApp()->getKernel("operator_tTV");
	else if(pIP->dir == ADJOINT)
		kernel = getApp()->getKernel("operator_tTVadj");

}

void TemporalTV::launch() {
	startProfiling();
	try {
		std::vector<cl::Event> kernelsExecEventList;
		cl::Event event;

		const cl::Buffer* pInputBuffer = getInput()->getDeviceBuffer();
		const cl::Buffer* pOutputBuffer = getOutput()->getDeviceBuffer();
        
		cl_uint numFrames = getInput()->getDynDimsTotalSize();

		cl::NDRange globalWorkSize = cl::NDRange(NDARRAYWIDTH(getInput()->getNDArray(0)), NDARRAYHEIGHT(getInput()->getNDArray(0)), NDARRAYDEPTH(getInput()->getNDArray(0))); //Added slices

		kernel.setArg(0, *pInputBuffer);
		kernel.setArg(1, *pOutputBuffer);
		kernel.setArg(2, numFrames);

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
		kernelsExecEventList.push_back(event);

		stopProfiling();
		if(pProfileParameters->enable)
			getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::TemporalTV::launch kernel", "OpenCLIPER::TemporalTV::launch group of kernels");
	}
	catch(cl::Error& err) {
		BTTHROW(CLError(err), "TemporalTV::launch");
	}
}
}
