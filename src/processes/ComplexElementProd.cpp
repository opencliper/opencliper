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
 * ComplexElementProd.cpp
 *
 *  Modified on: 29 de oct. de 2021
 *      Author: Emilio López-Ales
 */
#include <OpenCLIPER/processes/ComplexElementProd.hpp>
#include <OpenCLIPER/CLapp.hpp>
//#include <OpenCLIPER/Data.hpp>
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/KData.hpp>
#include <OpenCLIPER/SensitivityMapsData.hpp>
#include <LPISupport/InfoItems.hpp>
#include <iostream>

// Uncomment to show class-specific debug messages
//#define COMPLEXELEMENTPROD_DEBUG

#if !defined NDEBUG && defined COMPLEXELEMENTPROD_DEBUG
    #define COMPLEXELEMENTPROD_CERR(x) CERR(x)
#else
    #define COMPLEXELEMENTPROD_CERR(x)
    #undef COMPLEXELEMENTPROD_DEBUG
#endif

namespace OpenCLIPER {

void ComplexElementProd::init() {
	kernel = getApp()->getKernel("complexElementProd_kernel");
}

void ComplexElementProd::launch() {
	auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

	checkCommonLaunchParameters();

	try {
		bool inputIsKData = false, outputIsKData = false;
		cl::Event event;

		std::shared_ptr<Data> pTypedInputData, pTypedOutputData;
		pTypedInputData = std::dynamic_pointer_cast<KData>(getInput());
		if(pTypedInputData != nullptr) {
			inputIsKData = true;
		}
		else {
			pTypedInputData = std::dynamic_pointer_cast<XData>(getInput());
			if(pTypedInputData != nullptr) {
				inputIsKData = false;
			}
			else {
				BTTHROW(std::invalid_argument("inputData should be of type KData or XData"), "ComplexElementProd::launch");
			}
		}
		pTypedOutputData = std::dynamic_pointer_cast<KData>(getOutput());
		if(pTypedOutputData != nullptr) {
			outputIsKData = true;
		}
		else {
			pTypedOutputData = std::dynamic_pointer_cast<XData>(getOutput());
			if(pTypedOutputData != nullptr) {
				outputIsKData = false;
			}
			else {
				BTTHROW(std::invalid_argument("ComplexElementProd::launch: outputData should be of type KData or XData"),"ComplexElementProd::launch");
			}
		}
		if((inputIsKData == false) && (outputIsKData == false)) {
			BTTHROW(std::invalid_argument("ComplexElementProd::launch: input or output data should be of type KData (including valid Sensitivity Maps)"),"ComplexElementProd::launch");
		}

		if(getInput()->getData()->size() == 0) {
			BTTHROW(std::invalid_argument("ComplexElementProd::launch: inputData size is 0"), "ComplexElementProd::launch");
		}

		if(pLP->sensitivityMapsData == nullptr) {
			BTTHROW(std::invalid_argument("ComplexElementProd::launch: non-existing SensitivityMaps"), "ComplexElementProd::launch");
		}

		cl::Buffer* pInputBuffer = getInput()->getDeviceBuffer();
		cl::Buffer* pSensitivityMapsBuffer = pLP->sensitivityMapsData->getDeviceBuffer();
		cl::Buffer* pOutputBuffer = getOutput()->getDeviceBuffer();

		// Mask to invert the sign of a float if we need to conjugate sensitivity maps
		cl_uint conjugateMask = pLP->conjugateSensMap ? 0x80000000 : 0;

		kernel.setArg(0, *pInputBuffer);
		kernel.setArg(1, *pSensitivityMapsBuffer);
		kernel.setArg(2, *pOutputBuffer);
		kernel.setArg(3, conjugateMask);

		cl::NDRange globalSizes = {NDARRAYWIDTH(getInput()->getData()->at(0))* NDARRAYHEIGHT(getInput()->getData()->at(0))* NDARRAYDEPTH(getInput()->getData()->at(0))};

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSizes, cl::NDRange(), NULL, NULL);
	}
	catch(cl::Error& err) {
		BTTHROW(CLError(err), "ComplexElementProd::launch");
	}
}
} /* namespace OpenCLIPER */
#undef COMPLEXELEMENTPROD_DEBUG
