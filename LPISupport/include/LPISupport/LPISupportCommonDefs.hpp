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

#ifndef INCLUDE_LPISUPPORT_LPISUPPORTCOMMONDEFS_HPP
#define INCLUDE_LPISUPPORT_LPISUPPORTCOMMONDEFS_HPP

// macro for general debug logging
#ifndef NDEBUG
    #define CERR(x) do { std::cerr << x << std::flush; } while (0)
#else
    #define CERR(x)
#endif

// macros for element data types
#define TYPEID_COMPLEX std::type_index(typeid(complexType))
#define TYPEID_REAL std::type_index(typeid(realType))
#define TYPEID_DIMINDEX std::type_index(typeid(dimIndexType))
#define TYPEID_CL_UCHAR std::type_index(typeid(cl_uchar))

#endif //INCLUDE_LPISUPPORT_LPISUPPORTCOMMONDEFS_HPP_
