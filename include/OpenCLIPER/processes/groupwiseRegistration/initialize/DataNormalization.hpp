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
#ifndef DATANORMALIZATION_HPP
#define DATANORMALIZATION_HPP

#include <OpenCLIPER/Process.hpp>

namespace OpenCLIPER {

/**
 * @brief Process class to calculate the modulus of a complex number, and fill the real part of
 *        destination with the value, normalized to a new maximum.
 *
 */
class DataNormalization: public Process {
    public:
        struct LaunchParameters: Process::LaunchParameters {
	    float oldmax;
	    float newmax;

	    LaunchParameters(float oldmax, float newmax):
		oldmax(oldmax), newmax(newmax) {}
	};

	void init();
	void launch();

        const std::string getKernelFile() const { return "initialize.cl"; }

    private:
	using Process::Process;
};

} // namespace OpenCLIPER

#endif // DATANORMALIZATION_HPP
