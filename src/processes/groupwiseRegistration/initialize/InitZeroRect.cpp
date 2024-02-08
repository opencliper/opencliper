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
 * InitZeroRect.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/initialize/InitZeroRect.hpp>

// Uncomment to show class-specific debug messages
//#define INITZERORECT_DEBUG

#if !defined NDEBUG && defined INITZERORECT_DEBUG
    #define INITZERORECT_CERR(x) CERR(x)
#else
    #define INITZERORECT_CERR(x)
    #undef INITZERORECT_DEBUG
#endif

namespace OpenCLIPER {

void InitZeroRect::init() {
    kernel = getApp()->getKernel("initZeroRect");
}

void InitZeroRect::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    infoItems.addInfoItem("Title", "InitZeroRect info");

    cl_uint width = NDARRAYWIDTH(getInput()->getNDArray(0));
    cl_uint height = NDARRAYHEIGHT(getInput()->getNDArray(0));
    cl_uint numFrames = getInput()->getDynDimsTotalSize();

    int margin[3] = {(int)ceil(height / 40), (int)ceil(width / 40), (int)ceil(numFrames / 40)};

   dimIndexType longr1margin = (pLP->bound_box->at(3) + margin[0] + 1) - (pLP->bound_box->at(1) - margin[0] - 1) + 1;
    dimIndexType longr2margin = (pLP->bound_box->at(2) + margin[1] + 1) - (pLP->bound_box->at(0) - margin[1] - 1) + 1;
    
    
    std::vector<dimIndexType>* r1margin = new std::vector<dimIndexType>(longr1margin); // Indice filas bound_box
    for(dimIndexType i = 0; i < longr1margin; i++) {
	r1margin->at(i) = (pLP->bound_box->at(1) - margin[0] - 1) + i - 1;
    }
    std::vector<dimIndexType>* pDims = new std::vector<dimIndexType>({longr1margin});
    NDArray* r1marginNDArray = new ConcreteNDArray<dimIndexType>(pDims, r1margin);
    r1marginData = std::make_shared<XData>(getApp(), r1marginNDArray, TYPEID_INDEX);

    std::vector<dimIndexType>* r2margin = new std::vector<dimIndexType>(longr2margin); // Indice columnas bound_box
    for(dimIndexType i = 0; i < longr2margin; i++) {
	r2margin->at(i) = (pLP->bound_box->at(0) - margin[1] - 1) + i - 1;
    }
    delete(pDims); pDims = new std::vector<dimIndexType>({longr2margin});
    NDArray* r2marginNDArray = new ConcreteNDArray<dimIndexType>(pDims, r2margin);
    r2marginData = std::make_shared<XData>(getApp(), r2marginNDArray, TYPEID_INDEX);

    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* outputData = getOutput()->getDeviceBuffer();

	cl::NDRange globalWorkSize = cl::NDRange(longr1margin, longr2margin, getInput()->getDynDimsTotalSize());

	kernel.setArg(0, *outputData);
	kernel.setArg(1, *r1marginData->getDeviceBuffer());
	kernel.setArg(2, *r2marginData->getDeviceBuffer());

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::InitZeroRect::launch kernel", "OpenCLIPER::InitZeroRect::launch group of kernels");


    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "InitZeroRect::launch");
    }
}

}
