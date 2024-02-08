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
#ifndef SIMPLEMRIRECONSOS_HPP
#define SIMPLEMRIRECONSOS_HPP

#include <OpenCLIPER/Process.hpp>
#include <OpenCLIPER/processes/FFT.hpp>
#include <OpenCLIPER/processes/RSoS.hpp>

namespace OpenCLIPER {

class SimpleMRIReconSOS: public Process {
    public:
	void init();
	void launch();

    private:
	// We need to create subprocesses, so can't just inherit out parent class' constructors
	SimpleMRIReconSOS(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP = nullptr);
	SimpleMRIReconSOS(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP = nullptr);

	// We must allow our constructors to be called from Process::create()
	friend std::shared_ptr<SimpleMRIReconSOS> Process::create<SimpleMRIReconSOS>(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP);
	friend std::shared_ptr<SimpleMRIReconSOS> Process::create<SimpleMRIReconSOS>(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP);

	// Subprocesses
	std::shared_ptr<Process> pProcInvFFT;
	std::shared_ptr<Process> pRSoS;
};

} //namespace OpenCLIPER

#endif // SIMPLEMRIRECONSOS_HPP
