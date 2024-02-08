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
 * DataNormalization.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: Elena Martin Gonzalez
 */
#include <OpenCLIPER/processes/groupwiseRegistration/initialize/DataNormalization.hpp>

// Uncomment to show class-specific debug messages
//#define DATANORMALIZATION_DEBUG

#if !defined NDEBUG && defined DATANORMALIZATION_DEBUG
    #define DATANORMALIZATION_CERR(x) CERR(x)
#else
    #define DATANORMALIZATION_CERR(x)
    #undef DATANORMALIZATION_DEBUG
#endif

namespace OpenCLIPER {

void DataNormalization::init() {
    kernel = getApp()->getKernel("dataNormalization");
}

void DataNormalization::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    checkCommonLaunchParameters();
    startProfiling();
    try {
	std::vector<cl::Event> kernelsExecEventList;
	cl::Event event;

	const cl::Buffer* inputData = getInput()->getDeviceBuffer();
	const cl::Buffer* outputData = getOutput()->getDeviceBuffer();
        
        uint multi = 1;
        if (getInput()->getAllSizesEqual() == 1){
            multi = getInput()->getNumNDArrays();
            for(uint bb=0; bb < getInput()->getNumSpatialDims(); bb++)
                multi *= getInput()->getSpatialDimSize(bb, 0);
        }

	cl::NDRange globalWorkSize = cl::NDRange(multi);

	kernel.setArg(0, *outputData);
	kernel.setArg(1, *inputData);
	kernel.setArg(2, pLP->oldmax);
	kernel.setArg(3, pLP->newmax);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
	kernelsExecEventList.push_back(event);
	stopProfiling();
	if(pProfileParameters->enable)
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::DataNormalization::launch kernel", "OpenCLIPER::DataNormalization::launch group of kernels");

    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "DataNormalization::launch");
    }
}
}

#undef DATANORMALIZATION_DEBUG
