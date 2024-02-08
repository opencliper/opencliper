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

#ifndef ARRAYADDPROCESS_HPP
#define ARRAYADDPROCESS_HPP

#include <OpenCLIPER/Process.hpp>

namespace OpenCLIPER {
/**
 * @brief Process class for the addition operation between 2 arrays
 *
 */
class ArrayAddProcess : public OpenCLIPER::Process {
    public:
        using Process::Process;

        struct LaunchParameters : Process::LaunchParameters {
	    /// Data handle for second operand of the multiplication operation
	    DataHandle inHandleB;
	    /// number of rows of arrrays
	    unsigned int rows;
	    /// number of columns of arrays
	    unsigned int cols;
	    /// block size
	    unsigned int blockSize;
	    /**
	     * @brief Default constructor for struct fields initialization.
	     *
	     */
	    LaunchParameters(DataHandle h, unsigned int r, unsigned int c, unsigned int bs) :
		inHandleB(h), rows(r), cols(c), blockSize(bs) {}
	};

	void init();
	void launch();

        const std::string getKernelFile() const { return "performanceTests/arrayAdd.cl"; }
};
} // namespace OpenCLIPER
#endif // ARRAYADDPROCESS_HPP

