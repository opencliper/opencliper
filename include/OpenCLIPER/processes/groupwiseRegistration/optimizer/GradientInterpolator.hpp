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

#ifndef GRADIENTINTERPOLATOR_HPP
#define GRADIENTINTERPOLATOR_HPP

#include <OpenCLIPER/Process.hpp>
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/ConcreteNDArray.hpp>

namespace OpenCLIPER {

/**
 * @brief Process class to apply the interpolator to the image gradient, to deform it
 *        in the way given by the transformation.
 *
 */
class GradientInterpolator: public Process {
    public:
        struct InitParameters: ProcessCore::InitParameters {
	    uint longr1margin;
            uint longr2margin;
            std::vector<dimIndexType>* margin;
            std::vector<dimIndexType>* bound_box;
	                
	    /// constructor
	    explicit InitParameters(uint longr1margin, uint longr2margin, std::vector<dimIndexType>* margin, std::vector<dimIndexType>* bound_box): longr1margin(longr1margin), longr2margin(longr2margin), margin(margin), bound_box(bound_box) {}
	};
        
        
	struct LaunchParameters: Process::LaunchParameters {
	    std::shared_ptr<Data> dallData;
	    std::shared_ptr<Data> xData;

	    LaunchParameters(const std::shared_ptr<Data>& dallData, const std::shared_ptr<Data>& x): dallData(dallData), xData(x) {}
	};

	void init();
	void launch();

        const std::string getKernelFile() const { return "optimizer.cl"; }

    private:
	using Process::Process;

	std::shared_ptr<Data> r1marginData;
        std::shared_ptr<Data> r2marginData;

	enum d { d1 = 0, d2 = 1};
};

} // namespace OpenCLIPER

#endif // GRADIENTINTERPOLATOR_HPP
