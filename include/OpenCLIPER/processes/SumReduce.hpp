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
 * SumReduce.hpp
 *
 *  Created on: 7 de mar. de 2019
 *      Author: fedsim
 */

#ifndef SUMREDUCE_HPP
#define SUMREDUCE_HPP

#include <OpenCLIPER/Process.hpp>

namespace OpenCLIPER {
/**
 * @brief Process class to reduce a Data object by summing up all its pixels
 *
 */

class CLapp;

class SumReduce: public Process {
    public:
	void init();
	void launch();

        const std::string getKernelFile() const { return "internalKernels.cl"; }

    private:
	// non-spatial size and stride (including coils, if any)
	cl_uint batchSize;
	cl_uint batchDistance;

	cl_uint realGlobalSize;
	cl::NDRange globalSize;
	cl::NDRange localSize;

	cl_uint nWorkgroups;

	std::shared_ptr<Data> partialOutputs;
	std::shared_ptr<Data> partialOutputs2;

    private:
	using Process::Process;
};

} // namespace OpenCLIPER

#endif // SUMREDUCE_HPP
