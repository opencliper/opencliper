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
#ifndef PERMUTEBB_HPP
#define PERMUTEBB_HPP

#include <OpenCLIPER/Process.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <iostream>

namespace OpenCLIPER {

/**
 * @brief Process class to permute the matrices containing the BSpline products, to operate whith them.
 *
 *
 */
class PermuteBB: public Process {
    public:
	struct LaunchParameters: Process::LaunchParameters {
	    cl::Buffer Dpobj;

	    LaunchParameters(cl::Buffer Dp): Dpobj(Dp) {}
	};

	void init();
	void launch();

        const std::string getKernelFile() const { return "creabspline.cl"; }

    private:
	using Process::Process;

	enum BBx { BB = 0, BB1 = 1, BB2 = 2, BB11 = 3};
	enum auxBBx { auxBB = 0, auxBB1 = 1, auxBB2 = 2, auxBB11 = 3};
};

} // namespace OpenCLIPER

#endif // PERMUTEBB_HPP
