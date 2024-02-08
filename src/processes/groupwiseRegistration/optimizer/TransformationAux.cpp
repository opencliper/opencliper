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
 * TransformationAux.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/TransformationAux.hpp>


// Uncomment to show class-specific debug messages
//#define TRANSFORMATIONAUX_DEBUG

#if !defined NDEBUG && defined TRANSFORMATIONAUX_DEBUG
    #define TRANSFORMATIONAUX_CERR(x) CERR(x)
#else
    #define TRANSFORMATIONAUX_CERR(x)
    #undef TRANSFORMATIONAUX_DEBUG
#endif

namespace OpenCLIPER {

void TransformationAux::init() {
    auto pIP = std::dynamic_pointer_cast<InitParameters>(pInitParameters);
    kernel = getApp()->getKernel("transformationAux");
    
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

void TransformationAux::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* TData = getInput()->getDeviceBuffer();
	const cl::Buffer* TAuxData = getOutput()->getDeviceBuffer();

	const std::vector<cl_uint> dataSize = *(getOutput()->getNDArray(0)->getDims());
	const std::vector<cl_uint> sizeT = *(getInput()->getNDArray(0)->getDims());

	cl::NDRange globalWorkSize = cl::NDRange(dataSize[0] * dataSize[1], dataSize[2] * dataSize[3], dataSize[4] * dataSize[5]);

	kernel.setArg(0,  *TAuxData);
	kernel.setArg(1,  *TData);
	kernel.setArg(2,  *(pLP->coef)->getDeviceBuffer());
	kernel.setArg(3,  *r1Data->getDeviceBuffer());
	kernel.setArg(4,  *r2Data->getDeviceBuffer());
	kernel.setArg(5,  dataSize[0]);
	kernel.setArg(6,  dataSize[2]);
	kernel.setArg(7,  dataSize[4]);
	kernel.setArg(8,  (pLP->coef)->getSpatialDimSize(0,0));
	kernel.setArg(9,  sizeT[0]);
	kernel.setArg(10, sizeT[1]);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::TransformationAux::launch kernel", "OpenCLIPER::TransformationAux::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "TransformationAux::launch");
    }
}
}

#undef TRANSFORMATIONAUX_DEBUG
