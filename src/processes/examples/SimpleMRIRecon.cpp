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
#include <OpenCLIPER/processes/examples/SimpleMRIRecon.hpp>
#include <OpenCLIPER/processes/FFT.hpp>

namespace OpenCLIPER {

SimpleMRIRecon::SimpleMRIRecon(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP):Process(pCLapp, pPP) {
    // Create subprocess objects
    pProcInvFFT = Process::create<FFT>(pCLapp);
    pProcSensMapProd = Process::create<ComplexElementProd>(pCLapp);
    pProcAddXImages = Process::create<XImageSum>(pCLapp);
}

SimpleMRIRecon::SimpleMRIRecon(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP): SimpleMRIRecon(pCLapp, pPP) {
    // Set input/output as given
    setInput(pIn);
    setOutput(pOut);
}

/**
 * @brief Method for process initialization.
 *
 * Initializes subkernels (FFT, complex element product of x-images by conjugated sensitivity maps and
 * sum of x-images captured by all the coils at the same time frame)
 *
 */
void SimpleMRIRecon::init() {
    pProcInvFFT->setInput(getInput());		// Note: in-place transformation
    pProcInvFFT->setOutput(getInput());		//
    pProcInvFFT->init();

    pProcSensMapProd->setInput(getInput());	// Note: in-place transformation
    pProcSensMapProd->setOutput(getInput());	//
    pProcSensMapProd->init();

    pProcAddXImages->setInput(getInput());
    pProcAddXImages->setOutput(getOutput());
    pProcAddXImages->init();
}

/**
 * @brief Launches the simple MRI reconstruction process.
 *
 * It sets kernel execution parameters and requests kernel execution 1 or serveral times (according to field of
 * profileParamters struct). If profiling is enabled (according to field of profileParameters struct),
 * kernel execution times are stored.
 * @param[in] profileParameters profiling configuration
 */
void SimpleMRIRecon::launch() {
    checkCommonLaunchParameters();
    try {
	// Step 0: Inverse FFT of K-space data
	auto launchParmsInvFFT = std::make_shared<FFT::LaunchParameters>(FFT::BACKWARD);
	pProcInvFFT->setLaunchParameters(launchParmsInvFFT);
	pProcInvFFT->launch();

	// Step 1: Multiply X-space data by their sensitivity maps
        auto pInKData = std::dynamic_pointer_cast<KData>(getInput());
	auto sensMapsProdLP = std::make_shared<ComplexElementProd::LaunchParameters>(ComplexElementProd::conjugate, pInKData->getSensitivityMapsData());
	pProcSensMapProd->setLaunchParameters(sensMapsProdLP);
	pProcSensMapProd->launch();

	// Step 2: add all x-images in each frame together
	pProcAddXImages->launch();
    }
    catch(cl::Error& err) {
	BTTHROW(CLError(err), "SimpleMRIRecon::launch");
    }
}

} // namespace OpenCLIPER
