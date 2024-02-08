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
#include <LPISupport/InfoItems.hpp>
//#include <IL/devil_cpp_wrapper.hpp>
#include <iostream>

namespace LPISupport {
InfoItems::InfoItems() {}
InfoItems::~InfoItems() {}

/**
 * @brief Adds an InfoItem (struct with a name and value) to the vector of InfoItem elements (value is a float number)
 *
 * @param[in] name string value for InfoItem name field
 * @param[in] value double value for InfoItem value field (stored as a string with a fixed precision)
 * @param[in] numDigitsPrec number of precision digits for double value (double is always stored as string)
 */
void InfoItems::addInfoItem(std::string name, double value, unsigned int numDigitsPrec) {
    std::stringstream strstr;
    strstr << std::setprecision(numDigitsPrec);
    strstr.str(std::string()); // Emtpy string associated to string stream
    strstr << value;
    infoItemsVector.push_back({name, strstr.str()});
}

/**
 * @brief Adds an InfoItem (struct with a name and value) to the vector of InfoItem elements (value is an unsigned integer number)
 *
 * @param[in] name string value for InfoItem name field
 * @param[in] value unsigned integer value for InfoItem value field (stored as a string)
 */
void InfoItems::addInfoItem(std::string name, unsigned int value) {
    infoItemsVector.push_back({name, std::to_string(value)});
}

/**
 * @brief Adds an InfoItem (struct with a name and value) to the vector of InfoItem elements (value is an unsigned long integer number)
 *
 * @param[in] name string value for InfoItem name field
 * @param[in] value unsigned long integer value for InfoItems value field (stored as a string)
 */
void InfoItems::addInfoItem(std::string name, unsigned long value) {
    infoItemsVector.push_back({name, std::to_string(value)});
}

/**
 * @brief Appends a vector of InfoItem elements to the vector of InfoItem elements of this object (elements are copied from source to destination vector)
 *
 * @param[in] pNewInfoItems vector of InfoItem elements
 */
void InfoItems::append(const std::unique_ptr<InfoItems> pNewInfoItems) {
    std::copy(pNewInfoItems->infoItemsVector.begin(), pNewInfoItems->infoItemsVector.end(), std::back_inserter(infoItemsVector));
}

/**
  * @brief Returns a text string representation of the InfoItem vector of this object.
  * @param[in] outputFormat format of output
  * @return text string with InfoItem vector contents
  */
std::string InfoItems::to_string(OutputFormat outputFormat) {
    // stream for storing text strings
    std::stringstream ss;
    // if outputFormat is CSV (format for spreadsheet)
    if(outputFormat != HUMAN) {
	// if csv mode is CSVWITHEADERS (store row with header values)
	if(outputFormat == CSVWITHHEADERS) {
	    for(unsigned int i = 0; i < infoItemsVector.size(); i++) {
		ss << infoItemsVector.at(i).name;
		if(i < infoItemsVector.size() - 1)  // do not and ";" after last element
		    ss << ";";
	    }
	    ss << std::endl;
	}
    }
    for(unsigned int i = 0; i < infoItemsVector.size(); i++) {
	// if csv mode is disabled
	if(outputFormat == HUMAN) {
	    ss << infoItemsVector.at(i).name << ": " << infoItemsVector.at(i).value << std::endl;
	}
	else {
	    ss << infoItemsVector.at(i).value;
	    if(i < infoItemsVector.size() - 1)  // do not and ";" after last element
		ss << ";";
	}
    }
    ss << std::endl << std::flush;
    return ss.str();

}

/**
 * @brief save InfoItem vector contents to a file or prints to standard output (if outputFileName is empty)
 *
 * @param[in] outputFormat format of output
 * @param[in] outputFileName name of output file (if empty, InfoItem vector contents are printed to standard output)
 */
void InfoItems::saveOrPrint(OutputFormat outputFormat, std::string outputFileName) {
    if(outputFileName.compare("") != 0) {
	std::ofstream outputFile;
	outputFile.open(outputFileName);
	outputFile << to_string(outputFormat);
	outputFile.close();
    }
    else {
	std::cout << to_string(outputFormat);
    }
}

} /* namespace OpenCLIPER */
