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

#include <OpenCLIPER/processes/MemSet.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <OpenCLIPER/hostKernelFunctions.hpp>
#include <LPISupport/InfoItems.hpp>
#include <OpenCLIPER/XData.hpp>

namespace OpenCLIPER {

void MemSet::init() {
    //auto pIP=std::dynamic_pointer_cast<InitParameters>(pInitParameters);
    //if(!pIP) pIP=std::unique_ptr<InitParameters>(new InitParameters());

    if(!getOutput())
	BTTHROW(cl::Error(CL_INVALID_MEM_OBJECT, "OpenCLIPER::MemSet::init(): init() called before setOutputData()"), "MemSet::init");

    if(!getOutput()->getAllSizesEqual())
	BTTHROW(std::invalid_argument("OpenCLIPER::MemSet::init(): MemSet for variable-size data objects is not implemented at this time"), "MemSet::init");


    // Number of elements of Data is number of NDArrays multiplied by number of elements of
    // every NDArray
    std::cerr << "Number of NDArrays: " << getOutput()->getNumNDArrays() << std::endl;
    std::cerr << "Size of every NDArray: " << getOutput()->getNDArray(0)->size() << std::endl;
    globalSize = getOutput()->getNumNDArrays() * getOutput()->getNDArrayTotalSize(0);
    std::cerr << "Global size: " << globalSize << std::endl;
}

void MemSet::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);
    if(!pLP) pLP = std::unique_ptr<LaunchParameters>(new LaunchParameters());

    if (getOutput()->getElementDataType() == TYPEID_COMPLEX) {
	kernel = getApp()->getKernel("memset_complex");
	kernel.setArg(1, pLP->value.complex);
    } else if (getOutput()->getElementDataType() == TYPEID_REAL) {
	kernel = getApp()->getKernel("memset_real");
	kernel.setArg(1, pLP->value.real);
    } else if (getOutput()->getElementDataType() == TYPEID_INDEX) {
	kernel = getApp()->getKernel("memset_uint");
	kernel.setArg(1, pLP->value.dimIndex);
    } else {
	BTTHROW(std::invalid_argument("OpenCLIPER::MemSet::launch(): unsupported element data type"), "MemSet::launch");
    }
    cl::Buffer* outBuffer;

    outBuffer = getOutput()->getDeviceBuffer();
    kernel.setArg(0, *outBuffer);

    //cl_int err;
   // if((err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(), NULL, NULL)) != CL_SUCCESS)
   //	BTTHROW(cl::Error(err, getApp()->getOpenCLErrorCodeStr(err)), "MemSet::launch");

    for(uint i = 0; i < getOutput()->getNumNDArrays(); i++) {
	outBuffer = getOutput()->getDeviceBuffer(i);
	cl_uint dataSize = getOutput()->getNDArray(i)->size();

	cl::NDRange globalWorkSize = cl::NDRange(dataSize);

	kernel.setArg(0, *outBuffer);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, NULL);
	//queue.finish();
    }

}

} /* namespace OpenCLIPER */
