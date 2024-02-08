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
#ifndef INCLUDE_OPENCLIPER_PERFTESTCONFRESULT_HPP
#define INCLUDE_OPENCLIPER_PERFTESTCONFRESULT_HPP

#include <LPISupport/Utils.hpp>
#include <memory>
#include <LPISupport/SampleCollection.hpp>
#include <LPISupport/InfoItems.hpp>
#include <LPISupport/PerfTestConfResult.hpp>
#include <iostream>
#include <OpenCLIPER/CLapp.hpp>
#include <OpenCLIPER/ProgramConfig.hpp>

namespace OpenCLIPER {
/**
 * @brief Abstract class with common variables and methods for test program configuration (base for derived classes specific to testing programs)
 *
 */
class PerfTestConfResult : public LPISupport::PerfTestConfResult, public OpenCLIPER::ProgramConfig {
    public:
	/**
	 * @brief Struct for object configuration
	 *
	 */
	struct ConfigTraits: LPISupport::PerfTestConfResult::ConfigTraits, OpenCLIPER::ProgramConfig::ConfigTraits {
	    unsigned int numOfComputeUnits = 0;
	    unsigned int warpOrWavefrontSize = 0;
	    unsigned int clockFreq = 0;
	    unsigned long globalMemSizeBytes = 0;

	    //DeviceTraits(DeviceType t=DEVICE_TYPE_ANY,cl::QueueProperties p=cl::QueueProperties::None): type(t),queueProperties(p) {}
	    ConfigTraits() {

	    }
	    /// Destructor for the class
	    virtual ~ConfigTraits() {}
	    virtual void configure();
	};

	PerfTestConfResult(int argc, char* argv[], std::string extraSummary = "");
	virtual ~PerfTestConfResult();

    protected:
	PerfTestConfResult();
	virtual void buildSpecificInfo(void* extraInfo = nullptr);
    private:
};
} /* namespace OpenCLIPER */
#endif // INCLUDE_OPENCLIPER_PERFTESTCONFRESULT_HPP
