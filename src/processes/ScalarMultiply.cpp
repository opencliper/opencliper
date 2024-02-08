/* Copyright (C) 2018 Federico Simmross Wattenberg,
 *                    Manuel Rodríguez Cayetano,
 *                    Javier Royuela del Val,
 *                    Elena Martí­n González,
 *                    Elisa Moya Sáez,
 *                    Marcos Martí­n Fernández and
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
 * scalarMultiply.cpp
 *
 *  Created on: 7 de mar. de 2019
 *      Author: fedsim
 */

#include <OpenCLIPER/processes/ScalarMultiply.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <OpenCLIPER/hostKernelFunctions.hpp>
#include <LPISupport/InfoItems.hpp>
#include <OpenCLIPER/XData.hpp>

namespace OpenCLIPER {

void ScalarMultiply::init() {
    //auto pIP=std::dynamic_pointer_cast<InitParameters>(pInitParameters);
    //if(!pIP) pIP=std::unique_ptr<InitParameters>(new InitParameters());

    if(!getInput())
	BTTHROW(CLError(CL_INVALID_MEM_OBJECT, "init() called before setInputData()"), "ScalarMultiply::init");

    if(!getOutput())
	BTTHROW(CLError(CL_INVALID_MEM_OBJECT, "init() called before setOutputData()"), "ScalarMultiply::init");

    if(getInput()->getData()->size() != getOutput()->getData()->size())
	BTTHROW(CLError(CL_INVALID_MEM_OBJECT, "inputData and outputData must have the same number of images"), "ScalarMultiply::init");

    if(!getInput()->getAllSizesEqual())
	BTTHROW(std::invalid_argument("ScalarMultiply for variable-size data objects is not implemented at this time"), "ScalarMultiply::init");
    
    auto elementDataType = getInput()->getElementDataType();
    if(elementDataType != TYPEID_COMPLEX && elementDataType != TYPEID_REAL)
        BTTHROW(std::invalid_argument("ScalarMultiply is only implemented for complex and real data types at this time"), "ScalarMultiply::init");

    kernel = getApp()->getKernel("scalarMultiply");

    dimIndexType nSpatialDims = getInput()->getNumSpatialDims();
    dimIndexType nTotalDims = nSpatialDims + (getInput()->getNumCoils() >= 2 ? 1 : 0) + getInput()->getNumTemporalDims();

    // Calculate batch sizes and distances
    batchSize = 1;
    for(unsigned i = nSpatialDims; i < nTotalDims; i++)
	batchSize *= getDimSize(static_cast<const cl_uint*>(getInput()->getHostBuffer()), i, 0);

    //getDimStride will return batchDistance=0 if there are spatial dimensions only
    batchDistance = getDimStride(static_cast<const cl_uint*>(getInput()->getHostBuffer()), nSpatialDims, 0);

    // Caution: this only works for complex and real element types!
    globalSize = cl::NDRange(NDArray::getElementSize(getInput()->getElementDataType()) / sizeof(realType) * getInput()->getNDArrayTotalSize(0));
}

void ScalarMultiply::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);
    if(!pLP) pLP = std::unique_ptr<LaunchParameters>(new LaunchParameters());

    cl::Buffer* inBuffer = getInput()->getDeviceBuffer();
    cl::Buffer* outBuffer = getOutput()->getDeviceBuffer();

    kernel.setArg(0, pLP->factor);
    kernel.setArg(1, *inBuffer);
    kernel.setArg(2, *outBuffer);
    kernel.setArg(3, batchSize);
    kernel.setArg(4, batchDistance);

    cl_int err;
    if((err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, cl::NDRange(), NULL, NULL)) != CL_SUCCESS)
	BTTHROW(CLError(err, getApp()->getOpenCLErrorCodeStr(err)), "ScalarMultiply::launch");
}

} /* namespace OpenCLIPER */
