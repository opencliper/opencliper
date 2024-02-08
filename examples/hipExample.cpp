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
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/ProgramConfig.hpp>
#include <rocfft.h>
#include <string>

using namespace OpenCLIPER;
int main(int argc, char* argv[]) {
    std::shared_ptr<CLapp> pCLapp;

    try {
	// Set platform/device preferences according to command line options
	ProgramConfig* pProgramConfig = new ProgramConfig(argc, argv);
	auto pConfigTraits = std::dynamic_pointer_cast<ProgramConfig::ConfigTraits>(
				 pProgramConfig->getConfigTraits());

	auto platformTraits = pConfigTraits->platformTraits;
	auto deviceTraits = pConfigTraits->deviceTraits;
	deviceTraits.useHIP = true;

	// Create the app
	pCLapp = CLapp::create(platformTraits, deviceTraits);

	// Create input data
	auto pData = new std::vector<complexType>({1, 2, 3, 4, 5});
	auto pNDArray = NDArray::createNDArray(pData);
	auto pIn = std::make_shared<XData>(pCLapp, pNDArray, TYPEID_COMPLEX);

	// Create the FFT plan
	rocfft_plan plan = NULL;
	size_t fftLength = pIn->getNDArray(0)->size();
	rocfft_plan_create(&plan, rocfft_placement_inplace, rocfft_transform_type_complex_forward, rocfft_precision_single, 1, &fftLength, 1, NULL);

	// Execute plan, wait for execution to finish, destroy plan
	void* hipBuffer = pIn->getHIPDeviceBuffer();
	rocfft_execute(plan, &hipBuffer, NULL, NULL);
	hipDeviceSynchronize();
	rocfft_plan_destroy(plan);

	// Step 6: save output data
	pIn->matlabSave("output.mat");
	std::cout << pIn->hostBufferToString("FFT", 0) << '\n';

	// Step 7: clean up
	pIn = nullptr;
	pCLapp = nullptr;
    }
    catch(cl::BuildError& e) {
	CLapp::dumpBuildError(e);
    }
    catch(CLError& e) {
	std::cerr << CLapp::getOpenCLErrorInfoStr(e, argv[0]);
    }
    catch(std::exception& e) {
	LPISupport::Utils::showExceptionInfo(e, argv[0]);
    }
}
