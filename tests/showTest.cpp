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
// #include <OpenCLIPER/processes/FFT.hpp>
#include <iostream>
#include <string>
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

	// Create uninitialized 2-dimensional buffer and dump it on the terminal.
	{
        std::cerr<<"Dumping uninitialized data (don't trust it'll contain zeros):\n\n";
	auto pIn = std::make_shared<XData>(pCLapp, 2, 5, TYPEID_COMPLEX);
	std::cout<<pIn->hostBufferToString("Uninitialized data", 0);
        }

	// Create 2D data from immediate values and show them (dimensions as separate vector)
        {
        std::cerr<<"Showing 2D data...\n";
	auto pData = new std::vector<complexType>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
	auto pDims = new std::vector<dimIndexType>({5, 2});
	auto pIn = std::make_shared<XData>(pCLapp, pDims, pData);
        pIn->show();
        }

        // 3D data from a single vector and inline dimensions. Shown as a mosaic of 2D slices
        {
        std::cerr<<"Showing 3D static data...\n";
	auto pData = new std::vector<complexType>({0, 1, 2, 3, 4,
                                                   5, 6, 7, 8, 9,

                                                   10, 11, 12, 13, 14,
                                                   15, 16, 17, 18, 19,

                                                   20, 21, 22, 23, 24,
                                                   25, 26, 27, 28, 29});
        auto pIn = std::make_shared<XData>(pCLapp, 5, 2, 3, pData);
        pIn->show();
        }

	// 3 frames of 2D data from immediate values. Shown as a video
        {
        std::cerr<<"Showing 3D dynamic data...\n";
	auto pData1 = new std::vector<complexType>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
	auto pData2 = new std::vector<complexType>({10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
	auto pData3 = new std::vector<complexType>({20, 21, 22, 23, 24, 25, 26, 27, 28, 29});
        auto pData = new std::vector<std::vector<complexType>*>({pData1, pData2, pData3});

        auto pSpatialDims = new std::vector<dimIndexType>({5, 2});
	auto pTempDims = new std::vector<dimIndexType>({3});
	auto pIn = std::make_shared<XData>(pCLapp, pSpatialDims, pTempDims, pData);

        // Show at 3 frames per second
        Data::ShowParms sp; sp.fps=3;
        pIn->show(&sp);
        }

        // Load a simple image and show it
	{
        std::cerr<<"Showing a simple image...\n";
	auto pIn = std::make_shared<XData>(pCLapp, DATA_DIR "/Cameraman.tif", TYPEID_REAL);
	pIn->show();
	}

	// Load a video that contains complex data and show it
	{
        std::cerr<<"Showing a video of complex-type data...\n";
	auto pIn = std::make_shared<XData>(pCLapp, std::string(DATA_DIR "/heartVideo.mat"), TYPEID_COMPLEX);
	pIn->show();
	}

	// Load some K-Space data video and show it
	{
        std::cerr<<"Showing a video of complex-type K-space data. Note: large file; please be patient...\n";
	auto pIn = std::make_shared<KData>(pCLapp, DATA_DIR "/MRIdata.mat");
	pIn->show();
	}

	// Load some 3D+2t X-Space data video and show it
	{
        std::cerr<<"Showing a video of complex-type 5D data. Note: HUGE file; may trigger out-of-memory errors...\n";
	auto pIn = std::make_shared<XData>(pCLapp, DATA_DIR "/phantom3D_5fresp_20fcard.mat");

        // Show frames along the second temporal dimension
        Data::ShowParms sp; sp.timeDimension=1;
	pIn->show(&sp);
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
