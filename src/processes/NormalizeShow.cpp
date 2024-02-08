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
 * NormalizeShow.cpp
 *
 *  Created on: 18 de mar. de 2019
 *      Author: fedsim
 */

#include <OpenCLIPER/processes/NormalizeShow.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <OpenCLIPER/hostKernelFunctions.hpp>
#include <LPISupport/InfoItems.hpp>

namespace OpenCLIPER {

void NormalizeShow::init() {
    auto pIP = std::dynamic_pointer_cast<InitParameters>(pInitParameters);
    if(!pIP) pIP = std::unique_ptr<InitParameters>(new InitParameters());

    if(!getInput())
	BTTHROW(CLError(CL_INVALID_MEM_OBJECT, "init() called before setInputData()"), "NormalizeShow::init");

    if(!getOutput())
	BTTHROW(CLError(CL_INVALID_MEM_OBJECT, "init() called before setOutputData()"), "NormalizeShow::init");

    if(getInput()->getData()->size() != getOutput()->getData()->size())
	BTTHROW(CLError(CL_INVALID_MEM_OBJECT, "inputData and outputData must have the same number of images"), "NormalizeShow::init");

    if(!getInput()->getAllSizesEqual())
	BTTHROW(std::invalid_argument("NormalizeShow for variable-size data objects is not implemented at this time"), "NormalizeShow::init");

    kernel = getApp()->getKernel("normalize_show");

    dimIndexType nSpatialDims = getInput()->getNumSpatialDims();
    dimIndexType nTotalDims = nSpatialDims + (getInput()->getNumCoils() >= 2 ? 1 : 0) + getInput()->getNumTemporalDims();

    batchSize = 1;
    for(unsigned i = nSpatialDims; i < nTotalDims; i++)
	batchSize *= getInput()->getDimSize(i, 0);

    //getDimStride will return batchDistance=0 if there are spatial dimensions only
    batchDistance = getInput()->getDimStride(nSpatialDims, 0);
}

void NormalizeShow::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);
    if(!(pLP->sum))
	BTTHROW(std::invalid_argument("mandatory launch parameter 'sum' not set"), "NormalizeShow::launch");

    cl::Buffer* inputData = getInput()->getDeviceBuffer();
    cl::Buffer* outputData = getOutput()->getDeviceBuffer();

    cl::NDRange globalSizes = cl::NDRange(getInput()->getNDArrayTotalSize(0));
    cl::NDRange localSizes = cl::NDRange();

    kernel.setArg(0, *pLP->sum->getDeviceBuffer());
    kernel.setArg(1, *inputData);
    kernel.setArg(2, *outputData);
    kernel.setArg(3, batchSize);
    kernel.setArg(4, batchDistance);

    cl_int err;
    if((err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSizes, localSizes, NULL, NULL)) != CL_SUCCESS)
	BTTHROW(CLError(err, getApp()->getOpenCLErrorCodeStr(err)), "NormalizeShow::launch");
}

} /* namespace OpenCLIPER */
