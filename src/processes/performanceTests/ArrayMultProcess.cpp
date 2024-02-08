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
#include <OpenCLIPER/processes/performanceTests/ArrayMultProcess.hpp>
#define CLASSNAME "OpenCLIPER::ArrayMultProcess"

namespace OpenCLIPER {
void ArrayMultProcess::init() {
    kernel = getApp()->getKernel("arrayMult_kernel");
}

void ArrayMultProcess::launch() {
    checkCommonLaunchParameters();
    // Set input and output OpenCL buffers on device memory
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);
    cl::Buffer* pInBufA = getInput()->getDeviceBuffer();
    DataHandle inBufBDataHandle = pLP->inHandleB;
    if(inBufBDataHandle == INVALIDDATAHANDLE) {
	BTTHROW(std::invalid_argument(std::string(CLASSNAME) + std::string("::launch: non-existing second array")), "ArrayMultProcess::launch");
    }

    try {
	cl::Buffer* pInBufB = getApp()->getData(inBufBDataHandle)->getDeviceBuffer();
	cl::Buffer* pOutBuf = getOutput()->getDeviceBuffer();

	// Set kernel parameters
	kernel.setArg(0, *pInBufA);
	kernel.setArg(1, *pInBufB);
	kernel.setArg(2, *pOutBuf);
	kernel.setArg(3, pLP->RowsA);
	kernel.setArg(4, pLP->ColsA);
	kernel.setArg(5, pLP->ColsB);

	dimIndexType height = NDARRAYHEIGHT(getInput()->getNDArray(0));
	dimIndexType width = NDARRAYWIDTH(getInput()->getNDArray(0));

	//cl::NDRange globalSizes = cl::NDRange(height, width);
	cl::NDRange globalSizes = cl::NDRange(height * width);
	cl::NDRange localSizes;
	int blockSize = pLP->blockSize;
	switch(blockSize) {
	    case -1:
		localSizes = getApp()->getMaxLocalWorkItemSizes(globalSizes);
		break;
	    case 0:
		localSizes = cl::NDRange();
		break;
	    default:
		localSizes = cl::NDRange(blockSize);
	}
	std::cerr << "globalSizes: " << globalSizes[0] << ", " << globalSizes[1] << std::endl;
	std::cerr << "localsizes: " << localSizes[0] << ", " << localSizes[1] << std::endl;
	std::cerr << "globalSizes: " << globalSizes[0] << ", " << globalSizes[1] << std::endl;
	std::cerr << "localsizes: " << localSizes[0] << ", " << localSizes[1] << std::endl;
	// Execute kernel
	//startProfiling();
	eventsVector.resize(pProfileParameters->loops);
	for(unsigned long i = 0; i < pProfileParameters->loops; i++) {
	    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSizes, cl::NDRange(), NULL, &eventsVector.at(i));
	}
	buildKernelProfilingInfo();
	//stopProfiling();
    } catch(cl::Error& err) {
    	BTTHROW(CLError(err), "ArrayAddProcess::launch");
    }
}
}
