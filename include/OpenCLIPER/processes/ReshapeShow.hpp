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
 * ReshapeShow.hpp
 *
 *  Created on: 18 de mar. de 2019
 *      Author: fedsim
 */

#ifndef RESHAPESHOW_HPP
#define RESHAPESHOW_HPP

#include <OpenCLIPER/Process.hpp>

namespace OpenCLIPER {
/**
 * @brief Process class to "reshape" a Data object in a canvas window for visualization.
 *
 */

class CLapp;
class XData;

class ReshapeShow: public Process {
    public:
	struct InitParameters: ProcessCore::InitParameters {
	    /// No. of temporal (dynamic) dimension to use as time when showing videos
	    unsigned timeDimension;

	    /// constructor
	    explicit InitParameters(unsigned td = 0): timeDimension(td) {}
	};

	void init();
	void launch();

        const std::string getKernelFile() const { return "internalKernels.cl"; }

	std::shared_ptr<XData>			getCanvas() { return canvas; }
	const std::vector<std::array<uint,2>>&	getTileCoords() { return tileCoords; }

    private:
	using Process::Process;

	/// No. of temporal (dynamic) dimension to use as time when showing videos
        unsigned timeDimension;

        std::shared_ptr<XData> canvas;
	std::vector<std::array<uint, 2>> tileCoords;

	unsigned winWidth = 0;
	unsigned winHeight = 0;
};

} // namespace OpenCLIPER

#endif // RESHAPESHOW_HPP
