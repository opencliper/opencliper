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
 * SumLambdaMetric.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/SumLambdaMetric.hpp>

// Uncomment to show class-specific debug messages
//#define SUMLAMBDAMETRIC_DEBUG

#if !defined NDEBUG && defined SUMLAMBDAMETRIC_DEBUG
    #define SUMLAMBDAMETRIC_CERR(x) CERR(x)
#else
    #define SUMLAMBDAMETRIC_CERR(x)
    #undef SUMLAMBDAMETRIC_DEBUG
#endif

namespace OpenCLIPER {

void SumLambdaMetric::init() {
    auto pIP = std::dynamic_pointer_cast<InitParameters>(pInitParameters);
    kernel = getApp()->getKernel("sumLambdaMetric");

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

void SumLambdaMetric::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event; 

	const cl::Buffer* VData = getOutput()->getDeviceBuffer();

	const cl::Buffer* dtauxData = getInput()->getDeviceBuffer(dtau::dtaux);
	const cl::Buffer* dtaux2Data = getInput()->getDeviceBuffer(dtau::dtaux2);
	const cl::Buffer* dtauxyData = getInput()->getDeviceBuffer(dtau::dtauxy);
	const cl::Buffer* dtautData = getInput()->getDeviceBuffer(dtau::dtaut);
	const cl::Buffer* dtaut2Data = getInput()->getDeviceBuffer(dtau::dtaut2);

	const std::vector<cl_uint> sizeV = *(getOutput()->getNDArray(0)->getDims());
	const std::vector<cl_uint> sizedtau = *(getInput()->getNDArray(dtau::dtaux)->getDims());

	cl::NDRange globalWorkSize = cl::NDRange(r1Data->getSpatialDimSize(0,0), r2Data->getSpatialDimSize(0,0));

	kernel.setArg(0,  *VData);
	kernel.setArg(1,  sizedtau[2]);
	kernel.setArg(2,  sizedtau[3]);
	kernel.setArg(3,  *dtauxData);
	kernel.setArg(4,  *dtaux2Data);
	kernel.setArg(5,  *dtauxyData);
	kernel.setArg(6,  *dtautData);
	kernel.setArg(7,  *dtaut2Data);
	kernel.setArg(8,  *r1Data->getDeviceBuffer());
	kernel.setArg(9,  *r2Data->getDeviceBuffer());
	kernel.setArg(10, pLP->term0);
	kernel.setArg(11, pLP->term1);
	kernel.setArg(12, pLP->term2);
	kernel.setArg(13, pLP->term3);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::SumLambdaMetric::launch kernel", "OpenCLIPER::SumLambdaMetric::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "SumLambdaMetric::launch");
    }
}

}

#undef SUMLAMBDAMETRIC_DEBUG
