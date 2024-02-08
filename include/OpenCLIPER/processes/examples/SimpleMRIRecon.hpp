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
#ifndef SIMPLEMRIRECON_HPP
#define SIMPLEMRIRECON_HPP

#include <OpenCLIPER/Process.hpp>

namespace OpenCLIPER {

/// @brief Class that makes a simple MRI reconstruction of a group of k-images captured by several coils.
class SimpleMRIRecon: public Process {
    public:
	void init();
	void launch();

    private:
	// We need to create subprocesses, so can't just inherit out parent class' constructors
	SimpleMRIRecon(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP = nullptr);
	SimpleMRIRecon(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP = nullptr);

	// We must allow our constructors to be called from Process::create()
	friend std::shared_ptr<SimpleMRIRecon> Process::create<SimpleMRIRecon>(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP);
	friend std::shared_ptr<SimpleMRIRecon> Process::create<SimpleMRIRecon>(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP);

	/// Pointer to Process subclass in charge of obtaining the inverse FFT of a group of k-images
	std::shared_ptr<Process> pProcInvFFT;

	/// Pointer to Process subclass in charge of multiplying every x-image by the sensitivity map of the coil used to capture it
	std::shared_ptr<Process> pProcSensMapProd;

        /// Pointer to Process subclass in charge of adding images captured from all the coils at the same time frame
	std::shared_ptr<Process> pProcAddXImages;
};

} //namespace OpenCLIPER

#endif // SIMPLEMRIRECON_HPP
