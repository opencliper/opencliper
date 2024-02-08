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

#include <OpenCLIPER/ProgramConfig.hpp>

// Uncomment to show class-specific debug messages
//#define PROGRAMCONFIG_DEBUG

#if !defined NDEBUG && defined PROGRAMCONFIG_DEBUG
    #define PROGRAMCONFIG_CERR(x) CERR(x)
#else
    #define PROGRAMCONFIG_CERR(x)
    #undef PROGRAMCONFIG_DEBUG
#endif

namespace OpenCLIPER {

/**
 * @brief Class constructor (empty)
 *
 */

ProgramConfig::ProgramConfig() {

}

ProgramConfig::ProgramConfig(int argc, char* argv[], std::string extraSummary) {
    pConfigTraits = std::make_shared<ConfigTraits>();
    init(argc, argv, extraSummary);
}

/**
 * @brief Class destructor (empty)
 *
 */
ProgramConfig::~ProgramConfig() {

}

/**
 * @brief Sets configuration fields of pConfigTraits configuration object) from map of read program arguments field.
 *
 * It also calls setSpecificConfig config method defined by subclasses (includes configuration tasks specific of subclasses).
 */
void ProgramConfig::ConfigTraits::configure() {
    LPISupport::ProgramConfig::ConfigTraits::configure();
    //for (auto& mapElement: execArgsMap) { // segmentation fault in iteration after erase in debian 10 gcc 8
    for(ExecArgsMap::const_iterator pMapElement = execArgsMap.cbegin() ; pMapElement != execArgsMap.cend() ;) {
	char option = pMapElement->first.at(0);
	switch(option) {
	    case 't':
		if(pMapElement->second.compare("GPU") == 0) {
		    deviceTraits.type = CLapp::DEVICE_TYPE_GPU;
		}
		else {
		    deviceTraits.type = CLapp::DEVICE_TYPE_CPU;
		}
		pMapElement = execArgsMap.erase(pMapElement);
		break;
	    case 'd':
		deviceTraits.name = pMapElement->second;
		pMapElement = execArgsMap.erase(pMapElement);
		break;
	    case 'p':
		if(!pMapElement->second.empty()) {
		    platformTraits.name = pMapElement->second;
		}
		pMapElement = execArgsMap.erase(pMapElement);
		break;
	    case 'i':
		deviceTraits.useHIP = true;
		pMapElement = execArgsMap.erase(pMapElement);
		break;
	    case 's':
		showImagesOrVideos = true;
		pMapElement = execArgsMap.erase(pMapElement);
		break;
	    case 'm':
	    	numOfMotionCompensIters = stoul(pMapElement->second);
		pMapElement = execArgsMap.erase(pMapElement);
		break;
	    case 'l':
	    	tolVar = stof(pMapElement->second);
		pMapElement = execArgsMap.erase(pMapElement);
		break;
	    case 'c':
		showTimes = true;
		pMapElement = execArgsMap.erase(pMapElement);
		break;
	    default:
		pMapElement = std::next(pMapElement);
		break;
	}
    }
}

} /* namespace OpenCLIPER */
#undef PROGRAMCONFIG_DEBUG

