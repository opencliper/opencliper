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

#ifndef MOTIONCOMPENSATION_HPP
#define MOTIONCOMPENSATION_HPP

#include <OpenCLIPER/Process.hpp>
#include <OpenCLIPER/processes/GroupwiseRegistration.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/initialize/CopyDataGPU.hpp>
#include <OpenCLIPER/processes/groupwiseRegistration/optimizer/Interpolator.hpp>

namespace OpenCLIPER {

/**
 * @brief Process class to apply the motion compensation obtained by the groupwise registration.
 *        From the groupwise registration we obtain the original mesh, the new mesh with the pixelwise
 *        displacements and the bounding box where it has to be applied.
 *        It can be used in NESTA reconstruction to refine a solution.
 *
 */
class MotionCompensation : public Process {
    public:
        struct LaunchParameters: Process::LaunchParameters {
	    ArgumentsMotionCompensation* argsMC;

	    explicit LaunchParameters(ArgumentsMotionCompensation* args): argsMC(args) {}
	};

	void init();
	void launch();

    private:
	using Process::Process;
        
        // We need to create subprocesses, so can't just inherit out parent class' constructors
	MotionCompensation(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP = nullptr);
	MotionCompensation(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP = nullptr);

	// We must allow our constructors to be called from Process::create()
	friend std::shared_ptr<MotionCompensation> Process::create<MotionCompensation>(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP);
	friend std::shared_ptr<MotionCompensation> Process::create<MotionCompensation>(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP);
        
        std::shared_ptr<Process> pCopy;
        std::shared_ptr<Process> pProcess;
};

} // namespace OpenCLIPER
#endif // MOTIONCOMPENSATION_HPP
