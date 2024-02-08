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

#ifndef TRANSFORMATIONAUX_HPP
#define TRANSFORMATIONAUX_HPP

#include <OpenCLIPER/Process.hpp>
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/ConcreteNDArray.hpp>

namespace OpenCLIPER {

/**
 * @brief Process class to extract an auxiliar from the transformation
 *
 */
class TransformationAux: public Process {
    public:
        struct InitParameters: ProcessCore::InitParameters {
	    uint longr1;
            uint longr2;
            std::vector<dimIndexType>* bound_box;
	                
	    /// constructor
	    explicit InitParameters(uint longr1, uint longr2, std::vector<dimIndexType>* bound_box): longr1(longr1), longr2(longr2), bound_box(bound_box) {}
	};
        
        struct LaunchParameters: Process::LaunchParameters {
	    std::shared_ptr<Data> coef;

	    LaunchParameters(std::shared_ptr<Data>& coef):
		coef(coef) {}
	};

	void launch();
	void init();

        const std::string getKernelFile() const { return "optimizer.cl"; }

    private:
	using Process::Process;

	//cl::Buffer r1obj;
	//cl::Buffer r2obj;
        std::shared_ptr<Data> r1Data;
        std::shared_ptr<Data> r2Data;

};

} // namespace OpenCLIPER

#endif // TRANSFORMATIONAUX_HPP
