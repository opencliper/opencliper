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
#ifndef INCLUDE_OPENCLIPERPROGRAMCONFIG_HPP
#define INCLUDE_OPENCLIPERPROGRAMCONFIG_HPP

#include <LPISupport/Utils.hpp>
#include <memory>
#include <LPISupport/SampleCollection.hpp>
#include <LPISupport/InfoItems.hpp>
#include <LPISupport/ProgramConfig.hpp>
#include <iostream>
#include <OpenCLIPER/CLapp.hpp>

namespace OpenCLIPER {
/**
 * @brief Abstract class with common variables and methods for test program configuration (base for derived classes specific to testing programs)
 *
 */
class ProgramConfig : public virtual LPISupport::ProgramConfig {
    public:
	/**
	 * @brief Struct for object configuration
	 *
	 */
	struct ConfigTraits: public virtual LPISupport::ProgramConfig::ConfigTraits {
	    CLapp::PlatformTraits platformTraits;
	    CLapp::DeviceTraits deviceTraits;
	    bool showImagesOrVideos = false;
	    unsigned int numOfMotionCompensIters = 2;
	    float tolVar = 1.0;
	    bool showTimes = false;
	    //DeviceTraits(DeviceType t=DEVICE_TYPE_ANY,cl::QueueProperties p=cl::QueueProperties::None): type(t),queueProperties(p) {}
	    /// Destuctor for the class
	    virtual ~ConfigTraits() {}
	    ConfigTraits() {
		addSupportedShortOption('t', "OpenCLDeviceType", "set OpenCL device type (CPU|GPU)", false);
		addSupportedShortOption('d', "OpenCLDeviceName", "set OpenCL device name", false);
		addSupportedShortOption('p', "OpenCLPlatformName", "set OpenCL platform Name", false);
		addSupportedShortOption('i', "", "Use HIP if available", false);
		addSupportedShortOption('s', "", "show result images/videos", false);
		addSupportedShortOption('m', "iterations", "set number of motion compensation iterations", false);
		addSupportedShortOption('l', "tolVar", "set value for tolVar reconstruction paremeter", false);
		addSupportedShortOption('c', "", "show elapsed times", false);
	    }
	    virtual void configure();
	};

	ProgramConfig(int argc, char* argv[], std::string extraSummary="");
	virtual ~ProgramConfig();

    protected:
	ProgramConfig();

    private:
};
} /* namespace OpenCLIPER */
#endif // INCLUDE_OPENCLIPERPROGRAMCONFIG_HPP
