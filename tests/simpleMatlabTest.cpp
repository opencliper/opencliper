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
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/processes/examples/Negate.hpp>
#include <OpenCLIPER/processes/MemSet.hpp>
#include <iostream>
#include <string>
#include <OpenCLIPER/ProgramConfig.hpp>
#include <OpenCLIPER/buildconfig.hpp>


using namespace OpenCLIPER;
int main(int argc, char* argv[]) {

    try {
	// Step 1: get a new OpenCLIPER app
	CLapp::PlatformTraits platformTraits;
	CLapp::DeviceTraits deviceTraits;

	ProgramConfig* pProgramConfig = new ProgramConfig(argc, argv);
	auto pConfigTraits = std::dynamic_pointer_cast<ProgramConfig::ConfigTraits>(
				 pProgramConfig->getConfigTraits());

	deviceTraits = pConfigTraits->deviceTraits;
	platformTraits = pConfigTraits->platformTraits;


	auto pCLapp = CLapp::create(platformTraits, deviceTraits);

	// width = 2, height = 3, numFrames = 4, numCoils = 2
	std::shared_ptr<KData> genKData = std::shared_ptr<KData>(KData::genTestKData(pCLapp, 2, 3, 4, 2));
	genKData->matlabSave("outputGenKData.mat");

	std::cerr << "Testing memset process, setting GenKData values to 3+8j" << std::endl;
	auto pMemSet = Process::create<MemSet>(pCLapp);
	pMemSet->setOutput(genKData);

	auto launchParms = std::make_shared<MemSet::LaunchParameters>();
	launchParms->value.complex = {3,8};
	pMemSet->setLaunchParameters(launchParms);
	pMemSet->init();
	pMemSet->launch();
	genKData->device2Host();
	std::cerr << "genKData->getNCoils(): " << genKData->getNCoils() << std::endl;
	std::cerr << "genKData->getDynDimsTotalSize(): " << genKData->getDynDimsTotalSize() << std::endl;
	for (dimIndexType i=0; i < genKData->getNumNDArrays(); i++) {
	    std::cerr << genKData->hostBufferToString("hostBuffer(" + std::to_string(i) + ")", i) << std::endl;
	}

	genKData->matlabSave("outputGenKDataMemSet.mat");

	std::shared_ptr<XData> pIn(new XData(pCLapp, {std::string(DATA_DIR "/Cameraman.tif")}, TYPEID_REAL));
	pIn->save("Cameraman_loadedAndSaved.png");
	pIn->matlabSave("Cameraman.mat");
	std::shared_ptr<XData> pIn2(new XData(pCLapp, std::string("Cameraman.mat"), TYPEID_COMPLEX));
	//pCLapp->device2Host(pIn2, SyncSource::BUFFER_ONLY);
	pIn2->save("Cameraman_from_matlab.png");
	pCLapp = nullptr;
    }
    catch(CLError& e) {
	std::cerr << CLapp::getOpenCLErrorInfoStr(e, argv[0]);
    }
    catch(std::exception& e) {
	LPISupport::Utils::showExceptionInfo(e, argv[0]);
    }
}
