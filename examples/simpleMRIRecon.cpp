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
#include <OpenCLIPER/processes/examples/SimpleMRIRecon.hpp>
#include <OpenCLIPER/buildconfig.hpp>
#include <LPISupport/ProgramConfig.hpp>


using namespace OpenCLIPER;
int main(int argc, char* argv[]) {

    try {
	// Get a new OpenCLIPER app, selecting the CPU as the computing device
	CLapp::PlatformTraits platformTraits;
	CLapp::DeviceTraits deviceTraits;

	ProgramConfig* pProgramConfig = new ProgramConfig(argc, argv);
	auto pConfigTraits = std::dynamic_pointer_cast<ProgramConfig::ConfigTraits>(
				 pProgramConfig->getConfigTraits());

	deviceTraits = pConfigTraits->deviceTraits;
	platformTraits = pConfigTraits->platformTraits;

	auto pCLapp = CLapp::create(platformTraits, deviceTraits);

	// Load OpenCL kernel(s) (only internal kernels are used here)

        // -----------------------------------------------------------------------------
        // 1. Fully-sampled data
        // -----------------------------------------------------------------------------

	// Load input data from Matlab file
        std::cerr << "Loading fully-sampled data...\n";
	auto pInputKData = std::make_shared<KData>(pCLapp, DATA_DIR "/MRIdataSOS.mat");

	// Create output with suitable size
	auto pOutputXData= std::make_shared<XData>(pCLapp, pInputKData);

        // Show input data
        Data::ShowParms sp;
        sp.title="Input: fully sampled K-space data";
        pInputKData->show(&sp);

	// Create new process, set its input/output data sets
	// and bind it to our CL app
	auto pProcess = Process::create<SimpleMRIRecon>(pCLapp);
	pProcess->setInput(pInputKData);
	pProcess->setOutput(pOutputXData);

	// Initialize & launch process
	pProcess->init();
	pProcess->launch();

	// Show and save result
        sp.title = "Output";
	pOutputXData->show(&sp);

        pOutputXData->matlabSave("recon_fullySampled.mat");

        // -----------------------------------------------------------------------------
        // 2. Half-scanned data
        // -----------------------------------------------------------------------------

        std::cerr << "Loading half-scanned data...\n";
	pInputKData = std::make_shared<KData>(pCLapp, DATA_DIR "/halfscan/171116_slc7.mat");
	pOutputXData= std::make_shared<XData>(pCLapp, pInputKData);

        sp.title="Input: half-scanned K-space data";
        pInputKData->show(&sp);

	pProcess->setInput(pInputKData);
	pProcess->setOutput(pOutputXData);
	pProcess->init();
	pProcess->launch();

        sp.title = "Output";
	pOutputXData->show(&sp);

        pOutputXData->matlabSave("recon_halfScan.mat");
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
