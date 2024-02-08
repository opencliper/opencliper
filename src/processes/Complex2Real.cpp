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
 * Complex2Real.cpp
 *
 *  Created on: 14 de ene. de 2019
 *      Author: fedsim
 */

#include <OpenCLIPER/processes/Complex2Real.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <OpenCLIPER/hostKernelFunctions.hpp>
#include <LPISupport/InfoItems.hpp>

namespace OpenCLIPER {

void Complex2Real::init() {
    auto pIP = std::dynamic_pointer_cast<InitParameters>(pInitParameters);
    if(!pIP) pIP = std::unique_ptr<InitParameters>(new InitParameters());

    if(!getInput())
	BTTHROW(CLError(CL_INVALID_MEM_OBJECT, "OpenCLIPER::Complex2Real::init(): init() called before setInputData()"), "Complex2Real::init");

    if(!getOutput())
	BTTHROW(CLError(CL_INVALID_MEM_OBJECT, "OpenCLIPER::Complex2Real::init(): init() called before setOutputData()"), "Complex2Real::init");

    if(getInput()->getData()->size() != getOutput()->getData()->size())
	BTTHROW(CLError(CL_INVALID_MEM_OBJECT, "OpenCLIPER::Complex2Real::launch(): inputData and outputData must have the same number of images"), "Complex2Real::init");

    if(!getInput()->getAllSizesEqual())
	BTTHROW(std::invalid_argument("OpenCLIPER::Complex2Real::init(): Complex2Real for variable-size data objects is not implemented at this time"), "Complex2Real::init");

    switch(pIP->convType) {
	case ComplexPart::REAL:
	    kernel = getApp()->getKernel("complex2real_real");
	    break;
	case ComplexPart::IMAG:
	    kernel = getApp()->getKernel("complex2real_imag");
	    break;
	case ComplexPart::ABS:
	    kernel = getApp()->getKernel("complex2real_abs");
	    break;
	case ComplexPart::ARG:
	    kernel = getApp()->getKernel("complex2real_arg");
	    break;
	default:
	    BTTHROW(std::invalid_argument("OpenCLIPER::Complex2Real::init(): unknown conversion type requested"), "Complex2Real::init");
    }

    dimIndexType nSpatialDims = getInput()->getNumSpatialDims();
    dimIndexType nTotalDims = nSpatialDims + (getInput()->getNumCoils() >= 2 ? 1 : 0) + getInput()->getNumTemporalDims();

    batchSize = 1;
    for(unsigned i = nSpatialDims; i < nTotalDims; i++)
	batchSize *= getInput()->getDimSize(i, 0);

    // getDimStride will return batchDistance=0 if there are spatial dimensions only
    // Caution: strides between NDArrays for input and output may be different if NDArrays are smaller than device's alignment size
    inBatchDistance = getInput()->getDimStride(nSpatialDims,0);
    outBatchDistance = getOutput()->getDimStride(nSpatialDims,0);

}

void Complex2Real::launch() {
    cl::Buffer* inputData = getInput()->getDeviceBuffer();
    cl::Buffer* outputData = getOutput()->getDeviceBuffer();

    cl::NDRange globalSizes = cl::NDRange(getInput()->getNDArrayTotalSize(0));
    cl::NDRange localSizes = cl::NDRange();

    kernel.setArg(0, *inputData);
    kernel.setArg(1, *outputData);
    kernel.setArg(2, batchSize);
    kernel.setArg(3, inBatchDistance);
    kernel.setArg(4, outBatchDistance);

    cl_int err;
    if((err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSizes, localSizes, NULL, NULL)) != CL_SUCCESS)
	BTTHROW(CLError(err, getApp()->getOpenCLErrorCodeStr(err)), "Complex2Real::launch");
}
} /* namespace OpenCLIPER */
