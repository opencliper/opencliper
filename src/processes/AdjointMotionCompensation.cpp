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

#include <OpenCLIPER/processes/AdjointMotionCompensation.hpp>
#include <libgen.h>

namespace OpenCLIPER {

AdjointMotionCompensation::AdjointMotionCompensation(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP): Process(pCLapp, pPP) {
    // Create subprocess objects
    pCopy = Process::create<CopyDataGPU>(getApp());
    pInitZeroRect = Process::create<InitZeroRect>(getApp());
    pAdjointProcess = Process::create<AdjointInterpolator>(getApp());
}


AdjointMotionCompensation::AdjointMotionCompensation(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP): AdjointMotionCompensation(pCLapp, pPP) {
    // Set input/output as given
    setInput(pIn);
    setOutput(pOut);
}

void AdjointMotionCompensation::init(){
    pCopy->init();
    pInitZeroRect->init();
    pAdjointProcess->init();
}

void AdjointMotionCompensation::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    infoItems.addInfoItem("Title", "AdjointMotionCompensation info");
    startProfiling();
    try {

	pCopy->setInput(getInput());
	pCopy->setOutput(getOutput());
	pCopy->launch();

	pInitZeroRect->setInput(getInput());
	pInitZeroRect->setOutput(getOutput());
	auto parinitZeroRect = std::make_shared<OpenCLIPER::InitZeroRect::LaunchParameters>(pLP->argsMC->bound_box);
	pInitZeroRect->setLaunchParameters(parinitZeroRect);
	pInitZeroRect->launch();

	pAdjointProcess->setInput(getInput());
	pAdjointProcess->setOutput(getOutput());
	auto parAdjoint = std::make_shared<OpenCLIPER::AdjointInterpolator::LaunchParameters>(pLP->argsMC->xnData, pLP->argsMC->xData,
			  pLP->argsMC->bound_box);
	pAdjointProcess->setLaunchParameters(parAdjoint);
	pAdjointProcess->launch();

	stopProfiling();
    }
    catch(cl::Error& err) {
	BTTHROW(CLError(err), "AdjointMotionCompensation::launch");
    }
}
}
