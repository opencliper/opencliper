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
#include <OpenCLIPER/defs.hpp>
#include <OpenCLIPER/ProgramConfig.hpp>
#include <OpenCLIPER/KData.hpp>
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/defs.hpp>
#include <OpenCLIPER/ProgramConfig.hpp>
#include <OpenCLIPER/KData.hpp>
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/processes/examples/SimpleMRIRecon.hpp>
#include <OpenCLIPER/buildconfig.hpp>
#include <LPISupport/ProgramConfig.hpp>

using namespace OpenCLIPER;
int main(int argc, char* argv[]) {

    try {
	// Get a new OpenCLIPER app and select CPU as the computing device
	CLapp::PlatformTraits platformTraits;
	CLapp::DeviceTraits deviceTraits;

	ProgramConfig* pProgramConfig = new ProgramConfig(argc, argv);
	auto pConfigTraits = std::dynamic_pointer_cast<ProgramConfig::ConfigTraits>(
				 pProgramConfig->getConfigTraits());

	deviceTraits = pConfigTraits->deviceTraits;
	platformTraits = pConfigTraits->platformTraits;

	auto pCLapp = CLapp::create(platformTraits, deviceTraits);

	// Load input data from Matlab file and register it in our CL app
	// (data is sent to the computing device automatically)
	auto pInputKData = std::make_shared<KData>(pCLapp, DATA_DIR "/MRIdata.mat");

	// Create output with suitable size and register it in our CL app
	// (data is sent to the computing device automatically)
	auto pOutputXData = std::make_shared<XData>(pCLapp, pInputKData);

	// Create new process and load needed kernels
	auto pProcess = Process::create<SimpleMRIRecon>(pCLapp, pInputKData, pOutputXData);
	pCLapp->loadKernels();

	// Initialize & launch process
	pProcess->init();
	pProcess->launch();

	// Get data back from computing device (automatically called by save and matlabSave)
	// Save output data
	auto outputData = std::dynamic_pointer_cast<XData>(pOutputXData);
	outputData->save("XData", "png");
	outputData->matlabSave("outputFrames.mat");
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
