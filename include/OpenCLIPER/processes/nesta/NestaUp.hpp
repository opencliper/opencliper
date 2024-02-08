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
/*
 * NestaUp.hpp
 *
 *  Modified on: 29 de oct. de 2021
 *      Author: Emilio López-Ales
 */

#ifndef NESTAUP_HPP
#define NESTAUP_HPP

#include <OpenCLIPER/Process.hpp>
#include <OpenCLIPER/processes/FFT.hpp>
#include <OpenCLIPER/processes/ComplexElementProd.hpp>
#include <OpenCLIPER/processes/XImageSum.hpp>
#include <OpenCLIPER/processes/ApplyMask.hpp>
#include <OpenCLIPER/processes/nesta/TemporalTV.hpp>
#include <OpenCLIPER/processes/nesta/VectorNormalization.hpp>
#include <OpenCLIPER/processes/GroupwiseRegistration.hpp>
#include <OpenCLIPER/processes/MotionCompensation.hpp>
#include <OpenCLIPER/processes/AdjointMotionCompensation.hpp>
#include <OpenCLIPER/processes/ComplexAbsPow2.hpp>
#include <OpenCLIPER/processes/ComplexAbs.hpp>
#include <clblast_c.h>
#include <algorithm>

namespace OpenCLIPER {

/**
 * @brief Process class to apply Nesta algorithm
 *
 */

class NestaUp : public Process {
    public:
	struct LaunchParameters: Process::LaunchParameters {
	    float lambda_i;
	    float mu_f;
	    float La;
	    uint maxIntIter;
	    float tolVar; //Definido segun DemoUP
	    int verbose;
	    unsigned int maxIter;
	    uint stoptest;
	    uint miniter;
	    ArgumentsMotionCompensation* argsMC;
	    bool showProgress;

	    LaunchParameters(float lambda_i, float mu_f, float La, uint maxIntIter, float tolVar, int verbose, int maxIter, uint stoptest, uint miniter,
			     ArgumentsMotionCompensation* args, bool sp=false): lambda_i(lambda_i), mu_f(mu_f), La(La), maxIntIter(maxIntIter), tolVar(tolVar), verbose(verbose),
			     maxIter(maxIter), stoptest(stoptest), miniter(miniter), argsMC(args), showProgress(sp) {}
	};

	void init();
	void launch();

    private:
	// We need to create subprocesses, so can't just inherit out parent class' constructors
	NestaUp(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP = nullptr);
	NestaUp(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP = nullptr);

	// We must allow our constructors to be called from Process::create()
	friend std::shared_ptr<NestaUp> Process::create<NestaUp>(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP);
	friend std::shared_ptr<NestaUp> Process::create<NestaUp>(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP);

	// Internal methods
	void operatorA(std::shared_ptr<Data> inputData, std::shared_ptr<SensitivityMapsData> sensitivityMapsData, std::shared_ptr<SamplingMasksData> samplingMasksData,
		       std::shared_ptr<Data> outputData, std::shared_ptr<Data> auxDataFFT);
	void operatorAt(std::shared_ptr<Data> inputData, std::shared_ptr<SensitivityMapsData> sensitivityMapsData, std::shared_ptr<Data> outputData, std::shared_ptr<Data> auxDataFFT);
	void operatorU(std::shared_ptr<Data> inputData, std::shared_ptr<Data> outputData, std::shared_ptr<Data> auxiliarDataMC, ArgumentsMotionCompensation* argsMC);
	void operatorUt(std::shared_ptr<Data> inputData, std::shared_ptr<Data> outputData, std::shared_ptr<Data> auxiliarDataMC, ArgumentsMotionCompensation* argsMC);
	float myNormest(uint rows, uint cols, uint slices, uint NumFrames,  ArgumentsMotionCompensation* argsMC);

	// Attributes
	std::shared_ptr<Process> pFFTOutOfPlace;
	std::shared_ptr<Process> pDataAndSensitivityMapsProduct;
	std::shared_ptr<Process> pDataAndSamplingMasksProduct;
	std::shared_ptr<Process> pXImagesAllCoilSameFrameAddition;
	std::shared_ptr<Process> pTemporalTV;
	std::shared_ptr<Process> pTemporalTVt;
	std::shared_ptr<Process> pVectorNormalization;
	std::shared_ptr<Process> pMotionCompensation;
	std::shared_ptr<Process> pAdjointMotionCompensation;
	std::shared_ptr<Process> pCopy;
	std::shared_ptr<Process> pComplexAbsPow2;
	std::shared_ptr<Process> pComplexAbs;

	std::shared_ptr<Data> pAuxFFT;
};

} // namespace OpenCLIPER

#endif // NESTAUP_HPP
