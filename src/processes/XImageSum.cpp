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
 * XimageSum.cpp
 *
 *  Modified on: 29 de oct. de 2021
 *      Author: Emilio López-Ales
 */
#include <OpenCLIPER/processes/XImageSum.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/KData.hpp>
#include <LPISupport/InfoItems.hpp>

#define KERNELCOMPILEOPTS "-I../include/"
//#define KERNELCOMPILEOPTS "-cl-std=CL2.0 -I../include/ -g"

// Uncomment to show class-specific debug messages
//#define XIMAGESUM_DEBUG

#if !defined NDEBUG && defined XIMAGESUM_DEBUG
    #define XIMAGESUM_CERR(x) CERR(x)
#else
    #define XIMAGESUM_CERR(x)
    #undef XIMAGESUM_DEBUG
#endif

namespace OpenCLIPER {

/**
    * @brief Method for kernel initialization.
    *
    * It gets a reference to an OpenCL kernel previously loaded and compiled.
    *
    */
void XImageSum::init() {
	kernel = getApp()->getKernel("xImageSum_kernel");
}


/**
 * @brief Launches a kernel.
 *
 * It sets kernel execution parameters and requests kernel execution 1 or serveral times (according to field of
 * profileParamters struct). If profiling is enabled (according to field of profileParameters struct),
 * kernel execution times are stored.
 * @param[in] profileParameters profiling configuration
 */
void XImageSum::launch() {
	checkCommonLaunchParameters();
	try {
		cl::Buffer* pInputBuffer = getInput()->getDeviceBuffer();
		cl::Buffer* pOutputBuffer = getOutput()->getDeviceBuffer();

		kernel.setArg(0, *pInputBuffer);
		kernel.setArg(1, *pOutputBuffer);

		cl::NDRange globalSize = cl::NDRange(NDARRAYWIDTH(getInput()->getData()->at(0)) * NDARRAYHEIGHT(getInput()->getData()->at(0)) * NDARRAYDEPTH(getInput()->getData()->at(0)));

		cl::NDRange localSize = cl::NDRange();
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, NULL);
	}
	catch(cl::Error& err) {
		BTTHROW(CLError(err), "XImageSum::launch");
	}
}
} /* namespace OpenCLIPER */
#undef XIMAGESUM_DEBUG

