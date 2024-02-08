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

#ifndef GRADIENTWITHSMOOTHTERMS_HPP
#define GRADIENTWITHSMOOTHTERMS_HPP

#include <OpenCLIPER/Process.hpp>

namespace OpenCLIPER {

/**
 * @brief Process class to join every gradient.
 *
 */
class GradientWithSmoothTerms: public Process {
    public:
	struct LaunchParameters: Process::LaunchParameters {
	    std::shared_ptr<Data> dyData;
	    std::shared_ptr<Data> dxData;
	    std::shared_ptr<Data> BBgallData;
	    std::shared_ptr<Data> coefg;

	    LaunchParameters(const std::shared_ptr<Data>& dy, const std::shared_ptr<Data>& dx, const std::shared_ptr<Data>& BBgall, const std::shared_ptr<Data>& coefg):
		dyData(dy), dxData(dx), BBgallData(BBgall), coefg(coefg)  {}
	};

	void init();
	void launch();

        const std::string getKernelFile() const { return "optimizer.cl"; }

    private:
	using Process::Process;

	enum BBgx { BBg = 0, BB1g = 1, BB2g = 2, BB11g = 3};
	enum dV { original = 0, permuted = 1};
};

} // namespace OpenCLIPER

#endif // GRADIENTWITHSMOOTHTERMS_HPP
