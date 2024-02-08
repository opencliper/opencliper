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

#include <OpenCLIPER/buildconfig.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <OpenCLIPER/KData.hpp>
#include <LPISupport/Utils.hpp>
#include <LPISupport/ProgramConfig.hpp>

using namespace OpenCLIPER;
int main(int argc, char* argv[]) {
    try {
	CLapp::PlatformTraits platformTraits;
	CLapp::DeviceTraits deviceTraits;

	auto pCLapp = CLapp::create(platformTraits, deviceTraits);
	// Load input data from Matlab file
	std::shared_ptr<KData> pInputKData= std::make_shared<KData>(pCLapp, DATA_DIR "/MRIdata.mat");

	for(unsigned int i = 0; i < pInputKData->getData()->size(); i++) {
	    std::cout << "KData: " << i << std::endl;
	    std::cout << pInputKData->getData()->at(i)->hostDataToString("KData: " + i);
	}
    }
    catch(std::exception& e) {
	LPISupport::Utils::showExceptionInfo(e, argv[0]);
    }
}
