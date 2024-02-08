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
 * Utils.cpp
 *
 *  Created on: 15 de nov. de 2016
 *      Author: manrod
 */

#include <exception>
#include <stdexcept>    //g++ 4.8 doesn't seem to include <stdexcept> from <exception>
#include <LPISupport/Utils.hpp>
//#include <IL/devil_cpp_wrapper.hpp>
#include <iostream>

// Uncomment to show class-specific debug messages
//#define UTILS_DEBUG

#if !defined NDEBUG && defined UTILS_DEBUG
    #define UTILS_CERR(x) CERR(x)
#else
    #define UTILS_CERR(x)
    #undef UTILS_DEBUG
#endif
namespace LPISupport {

Utils::Utils() {
    // TODO Auto-generated constructor stub

}

Utils::~Utils() {
    // TODO Auto-generated destructor stub
}

/**
 * @brief Check if value is between min and max value (if value is less than min, it is set to min;
 * if value is greater than max, it is set to max)
 * @param[in,out] value value to be checked (and modified if out of range)
 * @param[in] min minimum value
 * @param[in] max maximum value
 */
void Utils::checkAndSetValue(unsigned long& value, unsigned long min, unsigned long max) {
    if(value < min) {
	UTILS_CERR("Valor " << value << " incorrecto, usando el valor " << min << std::endl);
	value = min;
    }
    else if(value > max) {
	UTILS_CERR("Valor " << std::to_string(value) << " incorrecto, usando el valor " << max << std::endl);
	value = max;
    }
}

void Utils::showExceptionInfo(std::exception& exception, const std::string& msg) {
    std::cerr << "Error: " << (exception).what() << "\nat " << (msg) << std::endl;
}

std::streampos Utils::fileLength(std::fstream &f) {
    // get length of file:
	std::streampos pos = f.tellg();
	//UTILS_CERR("File offset before move to end to read length: " << pos << std::endl);
    f.seekg(0, f.end);
    std::streampos fileLength = f.tellg();
    f.seekg(pos, f.beg);
	//UTILS_CERR("File offset before method return: " << f.tellg() << std::endl);
    return fileLength;
}

void Utils::openFile(const std::string &fileName, std::fstream &f, std::ios_base::openmode mode, const std::string &debugInfo) {
    UTILS_CERR("Open file " << fileName << "... ");
    f.open(fileName, mode);
    if(!f.good()) {
        BTTHROW(std::invalid_argument(fileName + " cannot be read\n"), debugInfo);
    }
    UTILS_CERR(" file length: " << Utils::fileLength(f) << " bytes\n");
    UTILS_CERR("Done." << std::endl);

}

void Utils::readBytesFromFile(std::fstream &f, char* s, std::streamsize sizeInBytes) {
	UTILS_CERR("Reading " << sizeInBytes << " characters... ");
    f.read(s, sizeInBytes);
    if(f)
    	UTILS_CERR("all characters read successfully." << std::endl);
    else {
    	UTILS_CERR("error: only " << f.gcount() << " could be read" << std::endl);
    	std::stringstream strstr;
    	strstr << "only " << f.gcount() << " bytes could be read (" << sizeInBytes << " requested)";
    	BTTHROW(std::invalid_argument(strstr.str()), "Utils::readBytesFromFile");
    }
}

std::string Utils::basename(const std::string &fileName) {
	std::size_t found = fileName.find_last_of(".");
	std::string baseFileName;
	if(found == std::string::npos) {
		baseFileName = fileName;	// If KData CFL file name lacks extension, this name is the base name for header file
	} else {
		baseFileName = fileName.substr(0, found);
	}
	return baseFileName;
}

std::string Utils::extensionname(const std::string &fileName) {
	std::size_t found = fileName.find_last_of(".");
	std::string extensionName;
	if(found == std::string::npos) {
		extensionName = "";	// If file name lacks extension, the extension name is empty
	} else {
		extensionName = fileName.substr(found+1, fileName.size()-1);
	}
	return extensionName;
}

}/* namespace OpenCLIPER */
#undef UTILS_DEBUG
