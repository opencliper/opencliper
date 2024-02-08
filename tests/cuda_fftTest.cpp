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

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime_api.h>


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

	pCLapp = CLapp::create(platformTraits, deviceTraits);

	// Load some K-Space data video
        std::cerr << "Loading input K-space data...\n";
	auto pIn = std::make_shared<KData>(pCLapp, DATA_DIR "/MRIdata.mat");

        std::cerr << "Measuring Performance: 2D Inverse Fourier Transform...\n";

	// CUDA code starts...
	cufftHandle plan;
	cufftComplex *inDeviceData;
	cufftComplex *outDeviceData;
        int nx = pIn->getSpatialDimSize(0,0);
        int ny = pIn->getSpatialDimSize(1,0);
        auto nBatches = pIn->getNumNDArrays();
        int nDims = 2;
	std::vector<int> dims = {nx, ny};

	cudaMalloc((void**)&inDeviceData, sizeof(cufftComplex)*nx*ny*nBatches);
	cudaMalloc((void**)&outDeviceData, sizeof(cufftComplex)*nx*ny*nBatches);
        cudaMemcpy(inDeviceData, pIn->getHostBuffer(), sizeof(cufftComplex)*nx*ny*nBatches,cudaMemcpyHostToDevice);
	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		exit(1);
	}

	/* Create a 3D FFT plan. */
	if (cufftPlanMany(&plan, nDims, dims.data(),
					NULL, 1, nx*ny, // *inembed, istride, idist
					NULL, 1, nx*ny, // *onembed, ostride, odist
					CUFFT_C2C, nBatches) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		exit(1);
	}

	/* Use the CUFFT plan to transform the signal in place. */
	LPISupport::Timer timer;
	unsigned nExperiments = 10;
	unsigned nIters = 1000;
	for(unsigned i = 0; i < nExperiments; i++) {
	    timer.start();
	    for(unsigned j = 0; j < nIters; j++)
		if(cufftExecC2C(plan, inDeviceData, outDeviceData, CUFFT_INVERSE) != CUFFT_SUCCESS){
		    fprintf(stderr, "CUFFT error: ExecC2C Backward failed");
		    exit(1);
		}
	}

	if (cudaDeviceSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		exit(1);
	}
        std::cerr << nIters <<" transforms in " << timer.get() << " s\n";
	cufftDestroy(plan);
	cudaFree(inDeviceData);
	cudaFree(outDeviceData);
	// CUDA code ends

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
