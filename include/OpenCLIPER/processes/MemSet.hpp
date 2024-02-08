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
 * MemSet.hpp
 *
 *  Created on: 7 de mar. de 2019
 *      Author: fedsim
 */

#ifndef MEMSET_HPP
#define MEMSET_HPP

#include <OpenCLIPER/Process.hpp>

namespace OpenCLIPER {
/**
 * @brief Process class to multiply Data objects by a scalar
 *
 */

class CLapp;

class MemSet: public Process {
    public:
	union ElementSupportedTypes {
	    complexType complex = {0.0, 0.0};
	    realType real;
	    dimIndexType dimIndex;
	};
	struct LaunchParameters: ProcessCore::LaunchParameters {
	    ElementSupportedTypes value;   //

	    /// constructor
	    explicit LaunchParameters() {}
	    explicit LaunchParameters(ElementSupportedTypes f): value(f) {}
	};

	void init();
	void launch();

    private:
	using Process::Process;

	uint globalSize;
};

} // namespace OpenCLIPER

#endif // MEMSET_HPP
