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
#ifndef REDUCTION7DTO6D_HPP
#define REDUCTION7DTO6D_HPP
#include <OpenCLIPER/Process.hpp>
#include <OpenCLIPER/CLapp.hpp>

namespace OpenCLIPER {

/**
 * @brief Process class to perform a reduction, from 7D to 6D (in a specific way)
 *
 */
class Reduction7Dto6D: public Process {
    public:
	struct LaunchParameters: Process::LaunchParameters {
	    float lambda0;
	    float lambda1;
	    float lambda2;
	    float lambda3;

	    LaunchParameters(float term_0, float term_1, float term_2, float term_3): lambda0(term_0), lambda1(term_1), lambda2(term_2), lambda3(term_3) {}
	};

	void init();
	void launch();

        const std::string getKernelFile() const { return "reductions.cl"; }

    private:
	enum auxiliar { aux = 0, aux1 = 1, aux2 = 2};
	enum dtheta { dthetax = 0, dthetax2 = 1, dthetaxy = 2, dthetat = 3, dthetat2 = 4};

    private:
	using Process::Process;
};

} // namespace OpenCLIPER

#endif // REDUCTION7DTO6D_HPP
