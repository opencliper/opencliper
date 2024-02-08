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
 * Regularization.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/Regularization.hpp>

// Uncomment to show class-specific debug messages
//#define REGULARIZATION_DEBUG

#if !defined NDEBUG && defined REGULARIZATION_DEBUG
    #define REGULARIZATION_CERR(x) CERR(x)
#else
    #define REGULARIZATION_CERR(x)
    #undef REGULARIZATION_DEBUG
#endif

namespace OpenCLIPER {

void Regularization::init() {
    auto pIP = std::dynamic_pointer_cast<InitParameters>(pInitParameters);
    kernel = getApp()->getKernel("regularization");

    std::vector<dimIndexType>* r1 = new std::vector<dimIndexType>(pIP->longr1); // Indice filas bound_box
    for(uint i = 0; i < pIP->longr1; i++) {
	r1->at(i) = (pIP->bound_box->at(0) - 1) + i;
    }
    std::vector<dimIndexType>* pDims = new std::vector<dimIndexType>({pIP->longr1});
    NDArray* r1NDArray = new ConcreteNDArray<dimIndexType>(pDims, r1);
    r1Data = std::make_shared<XData>(getApp(), r1NDArray, TYPEID_INDEX);


    std::vector<dimIndexType>* r2 = new std::vector<dimIndexType>(pIP->longr2); // Indice columnas bound_box
    for(uint i = 0; i < pIP->longr2; i++) {
	r2->at(i) = (pIP->bound_box->at(1) - 1) + i;
    }
    delete(pDims); pDims = new std::vector<dimIndexType>({pIP->longr2});
    NDArray* r2NDArray = new ConcreteNDArray<dimIndexType>(pDims, r2);
    r2Data = std::make_shared<XData>(getApp(), r2NDArray, TYPEID_INDEX);
}

void Regularization::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* BBData = (pLP->BBallData)->getDeviceBuffer(BBx::BB);
	const cl::Buffer* BB1Data = (pLP->BBallData)->getDeviceBuffer(BBx::BB1);
	const cl::Buffer* BB2Data = (pLP->BBallData)->getDeviceBuffer(BBx::BB2);
	const cl::Buffer* BB11Data = (pLP->BBallData)->getDeviceBuffer(BBx::BB11);

	const cl::Buffer* dtauxData = getOutput()->getDeviceBuffer(dtau::dtaux);
	const cl::Buffer* dtaux2Data = getOutput()->getDeviceBuffer(dtau::dtaux2);
	const cl::Buffer* dtauxyData = getOutput()->getDeviceBuffer(dtau::dtauxy);
	const cl::Buffer* dtautData = getOutput()->getDeviceBuffer(dtau::dtaut);
	const cl::Buffer* dtaut2Data = getOutput()->getDeviceBuffer(dtau::dtaut2);

	const cl::Buffer* TAuxPermData = getInput()->getDeviceBuffer(TAux::permuted);
	const cl::Buffer* TAuxShift1Data = getInput()->getDeviceBuffer(TAux::shift1);
	const cl::Buffer* TAuxShift2Data = getInput()->getDeviceBuffer(TAux::shift2);

	const std::vector<cl_uint> sizeTAux = *(getInput()->getNDArray(TAux::original)->getDims());
	const std::vector<cl_uint> sizedtau = *(getOutput()->getNDArray(dtau::dtaux)->getDims());
	cl::NDRange globalWorkSize = cl::NDRange(sizeTAux[1], sizeTAux[0], sizedtau[2] * sizedtau[3]);

	kernel.setArg(0, *dtauxData);
	kernel.setArg(1, *dtaux2Data);
	kernel.setArg(2, *dtauxyData);
	kernel.setArg(3, *dtautData);
	kernel.setArg(4, *dtaut2Data);
	kernel.setArg(5, *TAuxShift1Data);
	kernel.setArg(6, *TAuxShift2Data);
	kernel.setArg(7, *TAuxPermData);
	kernel.setArg(8, *BBData);
	kernel.setArg(9, *BB1Data);
	kernel.setArg(10, *BB2Data);
	kernel.setArg(11, *BB11Data);
	kernel.setArg(12, *r1Data->getDeviceBuffer());
	kernel.setArg(13, *r2Data->getDeviceBuffer());
	kernel.setArg(14, sizedtau[0]);
	kernel.setArg(15, sizedtau[1]);
	kernel.setArg(16, sizedtau[2]);
	kernel.setArg(17, sizeTAux[4]);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::Regularization::launch kernel", "OpenCLIPER::Regularization::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "Regularization::launch");
    }
}
}

#undef REGULARIZATION_DEBUG

