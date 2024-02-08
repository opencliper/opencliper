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
 * Utils.hpp
 *
 *  Created on: 12 mar. 2021
 *      Author: Federico Simmross
 */

#ifndef INCLUDE_LPISUPPORT_TIMER_HPP
#define INCLUDE_LPISUPPORT_TIMER_HPP

#include <chrono>

namespace LPISupport {
/**
 * @brief Class to count time elapsed between two events
 *
 */
class Timer {
    public:
	Timer() { start(); }
	virtual ~Timer() {}

	void start() { timer = std::chrono::steady_clock::now(); }
	double get() {  std::chrono::duration<double> k = std::chrono::steady_clock::now() - timer; return k.count(); }
	double getms() {  std::chrono::duration<double,std::milli> k = std::chrono::steady_clock::now() - timer; return k.count(); }
	double getus() {  std::chrono::duration<double,std::micro> k = std::chrono::steady_clock::now() - timer; return k.count(); }
	double getns() {  std::chrono::duration<double,std::nano> k = std::chrono::steady_clock::now() - timer; return k.count(); }

    private:
	/// Timer value at contruction or start() call
	std::chrono::time_point<std::chrono::steady_clock> timer;
};

} // namespace OpenCLIPER

#endif // INCLUDE_LPISUPPORT_TIMER_HPP */


