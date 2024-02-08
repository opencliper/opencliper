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
 *  Created on: 15 de nov. de 2016
 *      Author: manrod
 */

#ifndef INCLUDE_OPENCLIPER_UTILS_HPP
#define INCLUDE_OPENCLIPER_UTILS_HPP
#include <LPISupport/Utils.hpp>
#include <string>
#include <vector>
#include <sstream>
#include <chrono> // measurement of execution times
#include <iomanip> // for std::setprecision

// macro for general debug logging
#ifndef NDEBUG
    #define CERR(x) do { std::cerr << x << std::flush; } while (0)
#else
    #define CERR(x)
#endif


//------------------------
// Exception handling
//------------------------
// For java-like stack trace information
#ifdef USE_BACKWARD_STACKTRACE
    #include <backward.hpp>
#endif // USE_BACKWARD_STACKTRACE

#if !defined NDEBUG && defined USE_BACKWARD_STACKTRACE
// Macro to show a stack trace of method calls after an exception has been thrown
#define BTTHROW(exception,where) {\
        std::ostringstream btStream;\
        backward::StackTrace st; \
        st.load_here(32); \
        backward::Printer p; \
        p.color_mode = backward::ColorMode::always; \
        p.print(st, btStream); \
        int     statusBT; \
        char   *realname; \
        realname = abi::__cxa_demangle(typeid((exception)).name(), 0, 0, &statusBT);\
        std::cerr << "Exception: " << realname<< "\nCause: " << (exception).what() << "\nAt " << where << "\n" << btStream.str(); \
        free(realname);\
        throw(exception); \
        }
#else
//#define BTTHROW2(exception,where) {std::cerr << "at " << (where) << std::endl; throw(exception);}
#define BTTHROW(exception, where) {throw(exception);}
#endif //!defined NDEBUG && defined USE_BACKWARD_STACKTRACE


/// Namespace for the LPISupport library (support classes for the OpenCLIPER framework)
namespace LPISupport {
/**
  * @brief Class containing general purpose methods.
  *
  */
class Utils {
    public:

	Utils();
	virtual ~Utils();

	static void checkAndSetValue(unsigned long& value, unsigned long min, unsigned long max);
	static void showExceptionInfo(std::exception& e, const std::string& msg);
	static std::streampos fileLength(std::fstream &f);
	static void openFile(const std::string &fileName, std::fstream &f, std::ios_base::openmode mode, const std::string &debugInfo);
	static void readBytesFromFile(std::fstream &f, char* s, std::streamsize sizeInBytes);
	static std::string basename(const std::string &name);
	static std::string extensionname(const std::string &name);
    private:
#if !defined NDEBUG && defined USE_BACKWARD_STACKTRACE
	// A variable of a type defined in backward.hpp must be defined in a class of the LPISupport library
	// to force inclusion of backward required libraries (libbfd and libdl, for example) into libLPISupport.so
	// (this avoid errors of undefined symbols in libOpenCLIPER.so as BTTHROW macro, that calls
	// backtrace library methods, is called from OpenCLIPER classes but not directly from LPISupport classes).
	backward::Printer p;
#endif // USE_BACKWARD_STACKTRACE
};

} /* namespace OpenCLIPER */

#endif /* INCLUDE_OPENCLIPER_UTILS_HPP */
