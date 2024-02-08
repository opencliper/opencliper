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
#ifndef COMPLEXELEMENTPROD_HPP
#define COMPLEXELEMENTPROD_HPP

#include <OpenCLIPER/Process.hpp>
#include <OpenCLIPER/SensitivityMapsData.hpp>

namespace OpenCLIPER {

/**
 * @brief Process class for the element-wise complex multplication of two array of data.
 *
 */
class ComplexElementProd: public Process {
    public:
	/**
	 * Enumerated type with options for conjugate or not a sensitivity map before multiplying it
	 * by an x-space image
	 */
	enum ConjugateSensMap_t {
	    /// map must not be conjugated
	    notConjugate = 0,
	    /// maps must be conjugated
	    conjugate = 1
	};
	/**
	 * Enumerated type with options for class of input and output data parameters (used as source and
	 * destination for the product operation)
	 */
	enum DataParametersTypes_t {
	    /// input and output data are of class XData
	    BOTHXData = 0,
	    /// input data is of class KData (output is a XData object)
	    ONLYINPUTKDATA = 1,
	    /// output data is of class KData (input is a XData object)
	    ONLYOUTPUTKDATA = 2,
	    /// input and output data are of class KData
	    BOTHKDATA = 3
	};

	/**
	 * @brief Parameters used during kernel launching
	 *
	 */
	struct LaunchParameters: Process::LaunchParameters {
	    /// option to select conjugation or not of the sensitivity maps before multiplying them by images data
	    ConjugateSensMap_t conjugateSensMap;
	    // / data handle of the sensitivity maps to be multiplied
	    //DataHandle sensitivityMapsDataHandle=INVALIDDATAHANDLE;
	    /// Pointer to the sensitivity maps to be multiplied with
	    std::shared_ptr<SensitivityMapsData> sensitivityMapsData = nullptr;
	    //DataParametersTypes_t dataParametersTypes;
	    //Parameters(ConjugateSensMap_t c, DataParametersTypes_t t):conjugateSensMap(c), dataParametersTypes(t) {}

	    LaunchParameters(ConjugateSensMap_t c, const std::shared_ptr<SensitivityMapsData>& m): conjugateSensMap(c), sensitivityMapsData(m) {}
	};

	void init();
	void launch();

        const std::string getKernelFile() const { return "complexElementProd.cl"; }

    private:
        using Process::Process;
};

} // namespace OpenCLIPER

#endif // COMPLEXELEMENTPROD_HPP
