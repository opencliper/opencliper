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
#include <OpenCLIPER/ProgramConfig.hpp>
#include <OpenCLIPER/buildconfig.hpp>
#include <LPISupport/Timer.hpp>
#include <clblast.h>


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

	auto pIn = std::make_shared<XData>(pCLapp, DATA_DIR "/Cameraman.tif", TYPEID_REAL);

	auto pKernelData = new std::vector<realType>({1, 1, 1, 1, 1,
						      1, 1, 1, 1, 1,
						      1, 1, 1, 1, 1,
						      1, 1, 1, 1, 1,
						      1, 1, 1, 1, 1});

	//auto pKernelData = new std::vector<realType>({0.25, 0.5, 0.25,
	//					      0.5 , 1  , 0.5 ,
	//					      0.25, 0.5, 0.25});

	//auto pKernelData = new std::vector<realType>({0, 0, 0, 0, 0,
	//					      0, 0, 0, 0, 0,
	//					      0, 0, 1, 0, 0,
	//					      0, 0, 0, 0, 0,
	//					      0, 0, 0, 0, 0});
	auto pConvKernel = std::make_shared<XData>(pCLapp, 5, 5 , pKernelData);

	auto pOut = std::make_shared<XData>(pCLapp, pIn->getSpatialDimSize(0,0), pIn->getSpatialDimSize(1,0), TYPEID_REAL);

	LPISupport::Timer timer;
	for(auto i=0; i<10000; i++) {
	    auto ret = clblast::Convgemm<float>(clblast::KernelMode::kCrossCorrelation,
			1, //channels
			pIn->getSpatialDimSize(1,0), //image height
			pIn->getSpatialDimSize(0,0), //image width
			pConvKernel->getSpatialDimSize(1,0), //kernel height
			pConvKernel->getSpatialDimSize(0,0), //kernel width
			pConvKernel->getSpatialDimSize(1,0)/2, //pad height
			pConvKernel->getSpatialDimSize(0,0)/2, //pad width
			1, //pIn->getSpatialDimStride(1,0), // stride along height
			1, //pIn->getSpatialDimStride(0,0), // stride along width
			1, // dilation_h
			1, // dilation_w
			1, // num_kernels
			1, // batch_count,
			(*(pIn->getDeviceBuffer()))(), // image buffer
			0, // image offset
			(*(pConvKernel->getDeviceBuffer()))(), // kernel buffer
			0, // kernel offset
			(*(pOut->getDeviceBuffer()))(), // output buffer
			0, // result offset
			&(pCLapp->getCommandQueue())(), // command queue
			NULL); // event
	}
	pCLapp->getCommandQueue().finish();
	std::cout<<timer.get()<<"segundos\n";

	//pOut->save("convTest_output.png");
	//pIn->show();
	//pConvKernel->show();
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
