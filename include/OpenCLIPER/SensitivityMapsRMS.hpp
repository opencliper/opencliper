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

#ifndef SENSITIVITYMAPSRMS_HPP
#define SENSITIVITYMAPSRMS_HPP

#include <OpenCLIPER/defs.hpp>
#include <OpenCLIPER/Data.hpp>

namespace OpenCLIPER {

/// class SensitivityMapsRMS - Group of RMS values for sensitivity maps
class SensitivityMapsRMS: public Data {
    public:
	SensitivityMapsRMS();
	virtual ~SensitivityMapsRMS();

	// Inherit shared_ptr counter from base class
	std::shared_ptr<SensitivityMapsRMS> shared_from_this() {
	    return std::dynamic_pointer_cast<SensitivityMapsRMS>(Data::shared_from_this());
	}
    private:
	static constexpr const char* errorPrefix = "OpenCLIPER::SensitivityMapsRMS::";
};

} // namespace OpenCLIPER
#endif // SENSITIVITYMAPSRMS_HPP
