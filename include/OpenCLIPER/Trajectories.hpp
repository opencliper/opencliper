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

#ifndef INCLUDE_OPENCLIPER_TRAJECTORIES_HPP_
#define INCLUDE_OPENCLIPER_TRAJECTORIES_HPP_

#include <OpenCLIPER/Data.hpp>

namespace OpenCLIPER {
class Trajectories : public Data {
    public:
	/// @brief Name for variable inside matlab files (specific of Trajectories class)
	static constexpr const char* matVarNameTrajectories = "Trajectories";


	/**
	* @brief Deault constructor without parameters
	*/
	Trajectories() {};
	Trajectories(const std::shared_ptr<CLapp>& pCLapp, matvar_t* pMatlabVar, dimIndexType numOfSpatialDimensions, SyncSource host2DeviceSync = SYNCSOURCEDEFAULT);
	Trajectories(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Trajectories>& sourceData, bool copyData);
	Trajectories(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Trajectories>& sourceData, ElementDataType newElementDataType);

	virtual ~Trajectories() {};

	virtual std::shared_ptr<Data> clone(bool deepCopy) const override;
	virtual std::shared_ptr<Data> clone(ElementDataType newElementDataType) const override;


	void calcDataDims() {
	    Data::calcDataDims();
	}

    protected:
	// Inherit shared_ptr counter from base class
	std::shared_ptr<Trajectories> shared_from_this() const {
	    return std::dynamic_pointer_cast<Trajectories> (std::const_pointer_cast<Data>(Data::shared_from_this()));
	}

    private:
	static constexpr const char* errorPrefix = "OpenCLIPER::Trajectories::";
};
} /* namespace OpenCLIPER */

#endif // INCLUDE_OPENCLIPER_TRAJECTORIES_HPP_



