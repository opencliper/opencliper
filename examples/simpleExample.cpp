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

#include <iostream>
#include <string>
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/processes/examples/Negate.hpp>
#include <OpenCLIPER/ProgramConfig.hpp>
#include <OpenCLIPER/buildconfig.hpp>


using namespace OpenCLIPER;
int main(int argc, char* argv[]) {
    std::shared_ptr<CLapp> pCLapp;

    try {
	// Step 0: get a new OpenCLIPER app, initialize computing device and load OpenCL kernel(s)
	CLapp::PlatformTraits platformTraits;
	CLapp::DeviceTraits deviceTraits;

	ProgramConfig* pProgramConfig = new ProgramConfig(argc, argv);
	auto pConfigTraits = std::dynamic_pointer_cast<ProgramConfig::ConfigTraits>(
				 pProgramConfig->getConfigTraits());

	deviceTraits = pConfigTraits->deviceTraits;
	platformTraits = pConfigTraits->platformTraits;

	pCLapp = CLapp::create(platformTraits, deviceTraits);

	// Step 1: load input data
	auto pIn = std::make_shared<XData>(pCLapp, DATA_DIR "/Cameraman.tif", TYPEID_REAL);

	// Step 2: create output with same size as input
	auto pOut = std::make_shared<XData>(pCLapp, pIn, false);

	// Step 3: create new process, bound to our CL app and set its input/output data sets
        auto pProcess = Process::create<Negate>(pCLapp, pIn, pOut);
	pCLapp->loadKernels();

	// Step 4: initialize & launch process
	pProcess->init();
	pProcess->launch();

	// get data back from computing device (automatically done by save)
	// Step 5: save and show output data
	pOut->save("simpleExample_output.png");
	pOut->show();
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
