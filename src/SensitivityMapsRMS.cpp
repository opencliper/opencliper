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
 * SensitivityMapsRMS.cpp
 */

#include <OpenCLIPER/SensitivityMapsRMS.hpp>
// Uncomment to show class-specific debug messages
#define SENSITIVITYMAPSRMS_DEBUG

#if !defined NDEBUG && defined SENSITIVITYMAPSRMS_DEBUG
    #define SENSITIVITYMAPSRMS_CERR(x) CERR(x)
#else
    #define SENSITIVITYMAPSRMS_CERR(x)
    #undef SENSITIVITYMAPSRMS_DEBUG
#endif

namespace OpenCLIPER {

/**
 * @brief Default constructor.
 *
 * It calls superclass constructor.
 */
SensitivityMapsRMS::SensitivityMapsRMS(): Data() {

}

/**
 * Default constructor.
 */
SensitivityMapsRMS::~SensitivityMapsRMS() {

}

}
#undef SENSITIVITYMAPSRMS_DEBUG
