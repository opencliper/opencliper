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
 * GradientRegularization.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/GradientRegularization.hpp>

// Uncomment to show class-specific debug messages
//#define GRADIENTREGULARIZATION_DEBUG

#if !defined NDEBUG && defined GRADIENTREGULARIZATION_DEBUG
    #define GRADIENTREGULARIZATION_CERR(x) CERR(x)
#else
    #define GRADIENTREGULARIZATION_CERR(x)
    #undef GRADIENTREGULARIZATION_DEBUG
#endif

namespace OpenCLIPER {

void GradientRegularization::init() {
    kernel = getApp()->getKernel("gradientRegularization");
}

void GradientRegularization::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* BBgData = (pLP->BBgallData)->getDeviceBuffer(BBgx::BBg);
	const cl::Buffer* BB1gData = (pLP->BBgallData)->getDeviceBuffer(BBgx::BB1g);
	const cl::Buffer* BB2gData = (pLP->BBgallData)->getDeviceBuffer(BBgx::BB2g);
	const cl::Buffer* BB11gData = (pLP->BBgallData)->getDeviceBuffer(BBgx::BB11g);

	const cl::Buffer* dthetaxData = getOutput()->getDeviceBuffer(dtheta::dthetax);
	const cl::Buffer* dthetax2Data = getOutput()->getDeviceBuffer(dtheta::dthetax2);
	const cl::Buffer* dthetaxyData = getOutput()->getDeviceBuffer(dtheta::dthetaxy);
	const cl::Buffer* dthetatData = getOutput()->getDeviceBuffer(dtheta::dthetat);
	const cl::Buffer* dthetat2Data = getOutput()->getDeviceBuffer(dtheta::dthetat2);

	const cl::Buffer* dtauxData = getInput()->getDeviceBuffer(dtau::dtaux);
	const cl::Buffer* dtaux2Data = getInput()->getDeviceBuffer(dtau::dtaux2);
	const cl::Buffer* dtauxyData = getInput()->getDeviceBuffer(dtau::dtauxy);
	const cl::Buffer* dtautData = getInput()->getDeviceBuffer(dtau::dtaut);
	const cl::Buffer* dtaut2Data = getInput()->getDeviceBuffer(dtau::dtaut2);

	const std::vector<cl_uint> sizeBBg = *((pLP->BBgallData)->getNDArray(BBgx::BB1g)->getDims());
	const std::vector<cl_uint> sizedtau = *(getInput()->getNDArray(dtau::dtaux)->getDims());

	cl::NDRange globalWorkSize = cl::NDRange(sizeBBg[0] * sizeBBg[1], sizeBBg[2] * sizeBBg[3], sizeBBg[4] * sizeBBg[5]);

	kernel.setArg(0, *dthetaxData);
	kernel.setArg(1, *dthetax2Data);
	kernel.setArg(2, *dthetaxyData);
	kernel.setArg(3, *dthetatData);
	kernel.setArg(4, *dthetat2Data);
	kernel.setArg(5, *dtauxData);
	kernel.setArg(6, *dtaux2Data);
	kernel.setArg(7, *dtauxyData);
	kernel.setArg(8, *dtautData);
	kernel.setArg(9, *dtaut2Data);
	kernel.setArg(10, *(pLP->coefg)->getDeviceBuffer());
	kernel.setArg(11, *BBgData);
	kernel.setArg(12, *BB1gData);
	kernel.setArg(13, *BB2gData);
	kernel.setArg(14, *BB11gData);
	kernel.setArg(15, sizeBBg[0]);
	kernel.setArg(16, sizeBBg[2]);
	kernel.setArg(17, sizeBBg[4]);
	kernel.setArg(18, sizedtau[0]);
	kernel.setArg(19, sizedtau[1]);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::GradientRegularization::launch kernel", "OpenCLIPER::GradientRegularization::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "GradientRegularization::launch");
    }
}
}

#undef GRADIENTREGULARIZATION_DEBUG
