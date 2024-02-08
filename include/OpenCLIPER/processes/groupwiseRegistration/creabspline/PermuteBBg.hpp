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

#ifndef PERMUTEBBG_HPP
#define PERMUTEBBG_HPP

#include <OpenCLIPER/Process.hpp>

namespace OpenCLIPER {

/**
 * @brief Process class to permute the matrices containing the BSpline products used
 *        with the gradient, to operate whith them.
 *
 *
 */
class PermuteBBg: public Process {
    public:
	void init();
	void launch();

        const std::string getKernelFile() const { return "creabspline.cl"; }

    private:
	using Process::Process;

	enum BBxg { BBg = 0, BB1g = 1, BB2g = 2, BB11g = 3};
	enum aux1BBxg { aux1BB1g = 0, aux1BB2g = 1, aux1BB11g = 2};
};

} // namespace OpenCLIPER

#endif // PERMUTEBBG_HPP
