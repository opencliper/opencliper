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

#include <OpenCLIPER/processes/MotionCompensation.hpp>
#include <libgen.h>

namespace OpenCLIPER {
    
MotionCompensation::MotionCompensation(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP): Process(pCLapp, pPP) {
    // Create subprocess objects
    pCopy = Process::create<CopyDataGPU>(getApp());
    pProcess = Process::create<Interpolator>(getApp());
}

MotionCompensation::MotionCompensation(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP): MotionCompensation(pCLapp, pPP) {
    // Set input/output as given
    setInput(pIn);
    setOutput(pOut);
}

void MotionCompensation::init(){
    pCopy->init();
    pProcess->init();

}

void MotionCompensation::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    infoItems.addInfoItem("Title", "MotionCompensation info");
    startProfiling();
    try {

	pCopy->setInput(getInput());
	pCopy->setOutput(getOutput());
	pCopy->launch();

	// Now, we compute the interpolation
	pProcess->setInput(getInput());
	pProcess->setOutput(getOutput());
	auto params = std::make_shared<OpenCLIPER::Interpolator::LaunchParameters>(pLP->argsMC->xnData, pLP->argsMC->xData, pLP->argsMC->bound_box);
	pProcess->setLaunchParameters(params);
	pProcess->launch();

	stopProfiling();
    }
    catch(cl::Error& err) {
	BTTHROW(CLError(err), "MotionCompensation::launch");
    }
}
}
