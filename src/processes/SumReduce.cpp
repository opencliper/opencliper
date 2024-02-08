/* Copyright (C) 2018 Federico Simmross Wattenberg,
 *                    Manuel Rodrí­guez Cayetano,
 *                    Javier Royuela del Val,
 *                    Elena Martín González,
 *                    Elisa Moya Sáez,
 *                    Marcos MartÃ­n Fernández and
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
 * SumReduce.cpp
 *
 *  Created on: 7 de mar. de 2019
 *      Author: fedsim
 */

#include <OpenCLIPER/processes/SumReduce.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <OpenCLIPER/hostKernelFunctions.hpp>
#include <LPISupport/InfoItems.hpp>
#include <OpenCLIPER/XData.hpp>

namespace OpenCLIPER {

void SumReduce::init() {
    auto pIP = std::dynamic_pointer_cast<InitParameters>(pInitParameters);
    if(!pIP) pIP = std::unique_ptr<InitParameters>(new InitParameters());

    if(!getInput())
	BTTHROW(CLError(CL_INVALID_MEM_OBJECT, "init() called before setInputData()"), "SumReduce::init");

    if(!getOutput())
	BTTHROW(CLError(CL_INVALID_MEM_OBJECT, "init() called before setOutputData()"), "SumReduce::init");

    if(!getInput()->getAllSizesEqual())
	BTTHROW(std::invalid_argument("SumReduce for variable-size data objects is not implemented at this time"), "SumReduce::init");

    if(getInput()->getElementDataType() != TYPEID_REAL)
        BTTHROW(std::invalid_argument("SumReduce is only implemented for real data type at this time"), "SumReduce::init");

    kernel = getApp()->getKernel("reduce_sum");

    dimIndexType nSpatialDims = getInput()->getNumSpatialDims();
    dimIndexType nTotalDims = nSpatialDims + (getInput()->getNumCoils() >= 2 ? 1 : 0) + getInput()->getNumTemporalDims();

    // Calculate batch sizes and distances
    batchSize = 1;
    for(unsigned i = nSpatialDims; i < nTotalDims; i++)
	batchSize *= getInput()->getDimSize(i, 0);

    //getDimStride will return batchDistance=0 if there are spatial dimensions only
    batchDistance = getInput()->getDimStride(nSpatialDims, 0);

    realGlobalSize = getInput()->getNDArrayTotalSize(0);
    localSize = CLapp::calcLocalSize(kernel, getApp()->getDevice(), realGlobalSize);
    nWorkgroups = (realGlobalSize - 1) / localSize[0] + 1;
    globalSize = cl::NDRange(localSize[0] * nWorkgroups); // globalSize must be multiple of localSize in OpenCL<2.0

    partialOutputs = std::make_shared<XData> (getApp(), nWorkgroups, TYPEID_REAL);
    partialOutputs2 = std::make_shared<XData> (getApp(), (nWorkgroups > localSize[0]) ? (nWorkgroups / localSize[0]) : 1, TYPEID_REAL);
}

void SumReduce::launch() {
    cl::Buffer* inBuffer = getInput()->getDeviceBuffer();
    cl::Buffer* outBuffer = getOutput()->getDeviceBuffer();

    cl::Buffer* scratchGlobalBuffer = partialOutputs->getDeviceBuffer();	// scratch buffers in global memory
    cl::Buffer* scratchGlobalBuffer2 = partialOutputs2->getDeviceBuffer();	// alternately used as input/output

    // First iteration: reduce the whole input data to <nWorkgroups> elements
    // ----------------------------------------------------------------------
    kernel.setArg(0, *inBuffer);					// input (global memory)
    kernel.setArg(2, cl::Local(sizeof(realType) * localSize[0]));	// scratch buffer in local memory
    kernel.setArg(3, batchSize);
    kernel.setArg(4, batchDistance);
    kernel.setArg(5, realGlobalSize);

    if(nWorkgroups>=2)
        kernel.setArg(1, *scratchGlobalBuffer);	    // if more than one iteration is needed, set output to a temporary scratch buffer
    else
        kernel.setArg(1, *outBuffer);       // if only one iteration needed, set output to final output buffer

    // Launch reduction
    try {
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, NULL);
    } catch(cl::Error& e) {
	BTTHROW(CLError(e.err(), getApp()->getOpenCLErrorCodeStr(e.err())), "SumReduce::launch");
    }

    // Successive iterations: continue reducing until there is one element left
    // ------------------------------------------------------------------------
    if(nWorkgroups>=2) {
	cl_uint realCurrentGlobalSize = nWorkgroups;
        cl::NDRange currentGlobalSize;
        cl::NDRange currentLocalSize;
        size_t currentNWorkgroups;

        kernel.setArg(3, 1); // there is just one batch for successive iterations

        while(realCurrentGlobalSize > 1) {
	    currentLocalSize = CLapp::calcLocalSize(kernel, getApp()->getDevice(), realCurrentGlobalSize);
            currentNWorkgroups = (realCurrentGlobalSize - 1) / currentLocalSize[0] + 1;
	    currentGlobalSize = cl::NDRange(currentLocalSize[0] * currentNWorkgroups);

            // set last output buffer as kernel input
            kernel.setArg(0, *scratchGlobalBuffer);

            // set final output buffer as kernel output if this is the last iteration, or a temporary buffer otherwise
            if(currentNWorkgroups == 1)
                kernel.setArg(1, *outBuffer);
            else
                kernel.setArg(1, *scratchGlobalBuffer2);

            // Launch reduction
            try {
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, currentGlobalSize, currentLocalSize, NULL, NULL);
	    }
	    catch(cl::Error& e) {
                BTTHROW(CLError(e.err(), getApp()->getOpenCLErrorCodeStr(e.err())), "SumReduce::launch");
	    }

            // set input size for next iteration
            realCurrentGlobalSize = currentNWorkgroups;

            // Swap input/output scratch buffers
            cl::Buffer* aux = scratchGlobalBuffer;
            scratchGlobalBuffer = scratchGlobalBuffer2;
            scratchGlobalBuffer2 = aux;
        }
    }
}

} /* namespace OpenCLIPER */
