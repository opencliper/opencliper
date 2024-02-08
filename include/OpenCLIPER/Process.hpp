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

#ifndef PROCESS_HPP
#define PROCESS_HPP

#include <OpenCLIPER/ProcessCore.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <type_traits>

namespace OpenCLIPER {

class CLapp;

/**
 * @brief Class representing common data and behaviour for all the processes in charge of operations on data.
 *
 */
class Process: public ProcessCore {
    public:
	/// Destructor
	virtual ~Process() {}

	const std::shared_ptr<CLapp>   getApp() const;
	void                           setApp(const std::shared_ptr<CLapp>& pCLapp);

        virtual const std::string getKernelFile() const;

	/**
	 * @brief Creates a process object
	 * @param[in] pCLapp the CLapp in which this process will operate
	 * @param[in] pPP profile parameters to this process
	 * @return shared pointer to the newly created process
	 */
        template<typename T>
        static std::shared_ptr<T> create(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP = nullptr) {
	    static_assert(std::is_base_of<Process, T>::value, "Process::create: specified class does not derive from Process");

            // std::make_shared<T> requires a public constructor. Use std:shared_ptr<T>(new T()) instead
            auto pProcess = std::shared_ptr<T>(new T(pCLapp, pPP));

            pCLapp->addKernelFile(pProcess->getKernelFile());

            return pProcess;
        }

	/**
	 * @brief Creates a process object
	 * @param[in] pCLapp the CLapp in which this process will operate
	 * @param[in] pIn input Data object for this process
	 * @param[in] pOut output Data object for this process
	 * @param[in] pPP profile parameters to this process
	 * @return shared pointer to the newly created process
	 */
        template<typename T>
        static std::shared_ptr<T> create(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut,
            const std::shared_ptr<ProfileParameters>& pPP = nullptr) {
	    static_assert(std::is_base_of<Process, T>::value, "Process::create: specified class does not derive from Process");

            // std::make_shared<T> requires a public constructor. Use std:shared_ptr<T>(new T()) instead
            auto pProcess = std::shared_ptr<T>(new T(pCLapp, pIn, pOut, pPP));

            pCLapp->addKernelFile(pProcess->getKernelFile());

            return pProcess;
        }

    protected:
	/// Constructors. These are protected; use create() methods to create processes
	Process(const std::shared_ptr<ProfileParameters>& pPP = nullptr): ProcessCore(pPP) {}
	Process(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP = nullptr);
	Process(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data>& pInputData, const std::shared_ptr<Data>& pOutputData,
                const std::shared_ptr<ProfileParameters>& pPP = nullptr);

    private:
        /// The CLapp in which this process lives
	std::shared_ptr<CLapp> pCLapp = nullptr;
};
} // namespace OpenCLIPER
#endif // PROCESS_HPP
