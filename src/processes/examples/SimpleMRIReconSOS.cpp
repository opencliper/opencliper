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
#include <OpenCLIPER/KData.hpp>
#include <OpenCLIPER/processes/examples/SimpleMRIReconSOS.hpp>
#include <OpenCLIPER/processes/FFT.hpp>
#include <OpenCLIPER/processes/RSoS.hpp>
#include <LPISupport/InfoItems.hpp>
#include <sys/time.h>

namespace OpenCLIPER {

SimpleMRIReconSOS::SimpleMRIReconSOS(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP):Process(pCLapp, pPP) {
    // Create subprocess objects
    pProcInvFFT = Process::create<FFT>(pCLapp);
    pRSoS = Process::create<RSoS>(pCLapp);
}

SimpleMRIReconSOS::SimpleMRIReconSOS(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP): SimpleMRIReconSOS(pCLapp, pPP) {
    // Set input/output as given
    setInput(pIn);
    setOutput(pOut);
}

void SimpleMRIReconSOS::init() {
    // Initialize subprocesses
    pProcInvFFT->setInput(getInput());		// Note: in-place transformation
    pProcInvFFT->setOutput(getInput());		//
    pProcInvFFT->init();

    pRSoS->setInput(getInput());
    pRSoS->setOutput(getOutput());
    pRSoS->init();
}

void SimpleMRIReconSOS::launch() {
    checkCommonLaunchParameters();
    try {
	// Step 0: Inverse FFT of initial KData in place
	auto launchParmsInvFFT = std::make_shared<FFT::LaunchParameters>(FFT::BACKWARD);
	pProcInvFFT->setLaunchParameters(launchParmsInvFFT);

	struct timeval t0, t1;

	gettimeofday(&t0, 0);
	pProcInvFFT->launch();

	gettimeofday(&t1, 0);

	long elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;

	FILE* fp1 = fopen("TiemposEjecucionFFT.txt", "a+");
	fseek(fp1, 0, SEEK_END);
	fprintf(fp1, "%ld\n", elapsed);
	fclose(fp1);

	// FFT in-place, save results of first kernel
	std::dynamic_pointer_cast<KData>(getInput())->matlabSave("Temporal_IFFT_MRIReconSOS.mat");

	// Step 1: Root of Sum of Squares
	pRSoS->setInput(getInput());
	pRSoS->setOutput(getOutput());

	gettimeofday(&t0, 0);

	pRSoS->launch();

	getApp()->getCommandQueue().finish();

	gettimeofday(&t1, 0);//Elisa

	elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec; //Elisa

	FILE* fp2 = fopen("TiemposEjecucionRSS.txt", "a+");
	fseek(fp2, 0, SEEK_END);
	fprintf(fp2, "%ld\n", elapsed);
	fclose(fp2);
    } catch(cl::Error& err) {
	BTTHROW(CLError(err), "SimpleMRIReconSOS::launch");
    }
}
}
