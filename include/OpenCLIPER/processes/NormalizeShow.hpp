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
 * NormalizeShow.hpp
 *
 *  Created on: 18 de mar. de 2019
 *      Author: fedsim
 */

#ifndef NORMALIZESHOW_HPP
#define NORMALIZESHOW_HPP

#include <OpenCLIPER/Process.hpp>
#include <OpenCLIPER/Data.hpp>

namespace OpenCLIPER {
/**
 * @brief Process class to "normalize" a Data object for visualization. Note: this is not a normalization in a mathematical sense
 *
 */

class CLapp;

class NormalizeShow: public Process {
    public:
	struct LaunchParameters: ProcessCore::LaunchParameters {
	    const std::shared_ptr<Data> sum;

	    /// constructor
	    explicit LaunchParameters(const std::shared_ptr<Data>& s): sum(s) {}
	};

	void init();
	void launch();

        const std::string getKernelFile() const { return "internalKernels.cl"; }

    private:
	using Process::Process;

	// non-spatial size and stride (including coils, if any)
	cl_uint batchSize;
	cl_uint batchDistance;

	cl::NDRange globalSize;
	cl::NDRange localSize;

	cl_uint nWorkgroups;
};

} // namespace OpenCLIPER

#endif // NORMALIZESHOW_HPP
