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
#include <OpenCLIPER/processes/examples/SimpleMRIReconSOS.hpp>
#include <OpenCLIPER/buildconfig.hpp>


using namespace OpenCLIPER;

int main(int argc, char* argv[]) {

    try {
	ProgramConfig* pProgramConfig = new ProgramConfig(argc, argv);
	auto pConfigTraits = std::dynamic_pointer_cast<ProgramConfig::ConfigTraits>(
				 pProgramConfig->getConfigTraits());

	// Step 1: get a new OpenCLIPER app and initialize computing device
	CLapp::PlatformTraits platformTraits = pConfigTraits->platformTraits;
	CLapp::DeviceTraits deviceTraits = pConfigTraits->deviceTraits;

	auto pCLapp = CLapp::create(platformTraits, deviceTraits);

	// Step 2: load input data from Matlab file
	auto pInputKData = std::make_shared<KData>(pCLapp, DATA_DIR "/MRIdataSOS.mat");

	// Step 3: create output of same size as input
	auto pOutputXData = std::make_shared<XData>(pCLapp, pInputKData);

	// Step 4: create new process and set its input/output
	auto pProcess = Process::create<SimpleMRIReconSOS>(pCLapp);
	pProcess->setInput(pInputKData);
	pProcess->setOutput(pOutputXData);

	// Step 5: initialize process
	pProcess->init();

	// Step 6: launch process
	pProcess->launch();

	// Step 7 get data back from computing device (automatically called by matlabSave)
	// and store output data into matlab file
	pOutputXData->matlabSave("outputFramesSOS.mat");
	pOutputXData->show();
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
