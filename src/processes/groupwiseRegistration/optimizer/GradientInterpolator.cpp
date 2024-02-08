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
 * GradientInterpolator.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/GradientInterpolator.hpp>

// Uncomment to show class-specific debug messages
//#define GRADIENTINTERPOLATOR_DEBUG

#if !defined NDEBUG && defined GRADIENTINTERPOLATOR_DEBUG
    #define GRADIENTINTERPOLATOR_CERR(x) CERR(x)
#else
    #define GRADIENTINTERPOLATOR_CERR(x)
    #undef GRADIENTINTERPOLATOR_DEBUG
#endif

namespace OpenCLIPER {

void GradientInterpolator::init() {
    auto pIP = std::dynamic_pointer_cast<InitParameters>(pInitParameters);
    kernel = getApp()->getKernel("gradientInterpolator");


    std::vector<dimIndexType>* r1margin = new std::vector<dimIndexType>(pIP->longr1margin); // Indice filas bound_box
    for(dimIndexType i = 0; i < pIP->longr1margin; i++) {
	r1margin->at(i) = (pIP->bound_box->at(1) - pIP->margin->at(0) - 1) + i - 1;
    }
    std::vector<dimIndexType>* pDims = new std::vector<dimIndexType>({pIP->longr1margin});
    NDArray* r1marginNDArray = new ConcreteNDArray<dimIndexType>(pDims, r1margin);
    r1marginData = std::make_shared<XData>(getApp(), r1marginNDArray, TYPEID_INDEX);

    std::vector<dimIndexType>* r2margin = new std::vector<dimIndexType>(pIP->longr2margin); // Indice columnas bound_box
    for(dimIndexType i = 0; i < pIP->longr2margin; i++) {
	r2margin->at(i) = (pIP->bound_box->at(0) - pIP->margin->at(1) - 1) + i - 1;
    }
    delete(pDims); pDims = new std::vector<dimIndexType>({pIP->longr2margin});
    NDArray* r2marginNDArray = new ConcreteNDArray<dimIndexType>(pDims, r2margin);
    r2marginData = std::make_shared<XData>(getApp(), r2marginNDArray, TYPEID_INDEX);
    
}

void GradientInterpolator::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* d1Data = (pLP->dallData)->getDeviceBuffer(d::d1);
	const cl::Buffer* d2Data = (pLP->dallData)->getDeviceBuffer(d::d2);
	const cl::Buffer* xnData = getInput()->getDeviceBuffer();
	const cl::Buffer* dxData = getOutput()->getDeviceBuffer();

	const std::vector<cl_uint> dataSize = *(getOutput()->getNDArray(0)->getDims());    

	cl::NDRange globalWorkSize = cl::NDRange(r1marginData->getSpatialDimSize(0,0), r2marginData->getSpatialDimSize(0,0), dataSize[2]);

	kernel.setArg(0, *dxData);
	kernel.setArg(1, *d1Data);
	kernel.setArg(2, *d2Data);
	kernel.setArg(3, *xnData);
	kernel.setArg(4, *(pLP->xData)->getDeviceBuffer());
	kernel.setArg(5, *r1marginData->getDeviceBuffer());
	kernel.setArg(6, *r2marginData->getDeviceBuffer());

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::GradientInterpolator::launch kernel", "OpenCLIPER::GradientInterpolator::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "GradientInterpolator::launch");
    }
}

}

#undef GRADIENTINTERPOLATOR_DEBUG
