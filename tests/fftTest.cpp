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
#include <LPISupport/Utils.hpp>
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/KData.hpp>
#include <OpenCLIPER/processes/FFT.hpp>
#include <iostream>
#include <string>
#include <OpenCLIPER/ProgramConfig.hpp>
#include <OpenCLIPER/buildconfig.hpp>
#include <LPISupport/Timer.hpp>


using namespace OpenCLIPER;
int main(int argc, char* argv[]) {
    std::shared_ptr<CLapp> pCLapp;

    try {
	// Step 0: get a new OpenCLIPER app, initialize computing device and load OpenCL kernel(s)
	CLapp::PlatformTraits platformTraits;
	CLapp::DeviceTraits deviceTraits;
        Data::ShowParms sp;

	ProgramConfig* pProgramConfig = new ProgramConfig(argc, argv);
	auto pConfigTraits = std::dynamic_pointer_cast<ProgramConfig::ConfigTraits>(
				 pProgramConfig->getConfigTraits());

	deviceTraits = pConfigTraits->deviceTraits;
	platformTraits = pConfigTraits->platformTraits;

	bool showImages = pConfigTraits->showImagesOrVideos;
	bool showTimes = pConfigTraits->showTimes;
	if(!showImages && !showTimes) {
	    std::cerr << "No -s or -c options specified. Showing both images and elapsed times\n";
	    showImages = true;
	    showTimes = true;
	}
	
	pCLapp = CLapp::create(platformTraits, deviceTraits);

	// Load some K-Space data video
        std::cerr << "Loading input K-space data...\n";
	auto pIn = std::make_shared<KData>(pCLapp, DATA_DIR "/MRIdata.mat");
	if(showImages) {
	    sp.title = "Input data";
	    pIn->show(&sp);
	}

	// Create empty output buffer
	auto pOut = std::make_shared<KData>(pCLapp, pIn);

	// Create an FFT process and initialize it from input data
        std::cerr << "Creating FFT process...\n";
	auto fft = Process::create<FFT>(pCLapp);
	fft->setInput(pIn);
	fft->setOutput(pOut);
	fft->init();

	if(showImages) {
	    // Inverse transform input data and show it
	    auto launchParms = std::make_shared<FFT::LaunchParameters>(FFT::Direction::BACKWARD);
	    std::cerr << "Performing 2D Inverse Fourier Transform...\n";
	    fft->setLaunchParameters(launchParms);
	    fft->launch();
	    sp.title = "2D Inverse Fourier Transform";
	    pOut->show(&sp);

	    // Reinitialize FFT to transform along the row dimension only
	    auto initParms = std::make_shared<FFT::InitParameters>(0);
	    fft->setInitParameters(initParms);
	    fft->init();

	    // Inverse transform input data and show it
	    std::cerr << "Performing 1D Inverse Fourier Transform (row direction)...\n";
	    fft->launch();
	    sp.title = "1D Inverse Fourier Transform (row direction)";
	    pOut->show(&sp);

	    // Reinitialize FFT to transform along the column dimension only
	    initParms->dim = 1;
	    fft->setInitParameters(initParms);
	    fft->init();

	    // Inverse transform input data and show it
	    std::cerr << "Performing 1D Inverse Fourier Transform (column direction)...\n";
	    fft->launch();
	    sp.title = "1D Inverse Fourier Transform (column direction)";
	    pOut->show(&sp);
	}

	if(showTimes) {
	    std::cerr << "Measuring Performance: 2D Inverse Fourier Transform...\n";
	    auto initParms = std::make_shared<FFT::InitParameters>();
	    fft->setInitParameters(initParms);
	    fft->init();
	    LPISupport::Timer timer;
	    unsigned nExperiments = 10;
	    unsigned nIters = 1000;
	    for(unsigned i = 0; i < nExperiments; i++) {
		timer.start();
		for(unsigned j = 0; j < nIters; j++)
		    fft->launch();
		pCLapp->getCommandQueue().finish();
		std::cerr << nIters <<" transforms in " << timer.get() << " s\n";
	    }
	}
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
