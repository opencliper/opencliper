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
 * GradientWithSmoothTerms.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/GradientWithSmoothTerms.hpp>

// Uncomment to show class-specific debug messages
//#define GRADIENTWITHSMOOTHTERMS_DEBUG

#if !defined NDEBUG && defined GRADIENTWITHSMOOTHTERMS_DEBUG
    #define GRADIENTWITHSMOOTHTERMS_CERR(x) CERR(x)
#else
    #define GRADIENTWITHSMOOTHTERMS_CERR(x)
    #undef GRADIENTWITHSMOOTHTERMS_DEBUG
#endif

namespace OpenCLIPER {

void GradientWithSmoothTerms::init() {
    kernel = getApp()->getKernel("gradientJoint");
}

void GradientWithSmoothTerms::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* BBgData = (pLP->BBgallData)->getDeviceBuffer(BBgx::BBg);
	const cl::Buffer* dyData = (pLP->dyData)->getDeviceBuffer();
	const cl::Buffer* dxData = (pLP->dxData)->getDeviceBuffer();

	const cl::Buffer* dVData = getOutput()->getDeviceBuffer(dV::original);
	const cl::Buffer* AuxData = getInput()->getDeviceBuffer();

	const std::vector<cl_uint> sized = *((pLP->dxData)->getNDArray(0)->getDims());
	const std::vector<cl_uint> sizeBBg = *((pLP->BBgallData)->getNDArray(BBgx::BB1g)->getDims());
	const std::vector<cl_uint> sizedV = *(getOutput()->getNDArray(0)->getDims());

	cl::NDRange globalWorkSize = cl::NDRange(sizedV[0] * sizedV[1], sizedV[2] * sizedV[3], sizedV[4] * sizedV[5]);

	kernel.setArg(0, *dVData);
	kernel.setArg(1, *dyData);
	kernel.setArg(2, *dxData);
	kernel.setArg(3, *BBgData);
	kernel.setArg(4, *AuxData);
	kernel.setArg(5, *(pLP->coefg)->getDeviceBuffer());
	kernel.setArg(6, sizeBBg[0]);
	kernel.setArg(7, sizeBBg[2]);
	kernel.setArg(8, sizeBBg[4]);
	kernel.setArg(9, sized[0]);
	kernel.setArg(10, sized[1]);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::GradientWithSmoothTerms::launch kernel", "OpenCLIPER::GradientWithSmoothTerms::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "GradientWithSmoothTerms::launch");
    }
}

}

#undef GRADIENTWITHSMOOTHTERMS_DEBUG
