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
 * InfoItems.hpp
 *
 *  Created on: 15 de nov. de 2016
 *      Author: manrod
 */
#ifndef INCLUDE_OPENCLIPER_INFOITEMS_HPP_
#define INCLUDE_OPENCLIPER_INFOITEMS_HPP_
#include <LPISupport/Utils.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <getopt.h>
#include <sstream> // string stream
#include <fstream>
#include <chrono> // measurement of execution times
#include <iomanip> // for std::setprecision

namespace LPISupport {
/**
 * @brief Class that stores a vector of pairs (name, value) used for grouping output related data and for storing
 * them in human-friendly or CSV spreadsheet format (CSV: Comma Separated Values)
 *
 */
class InfoItems {
    public:
	/// @brief Pair of name and value fields (InfoItem element)
	struct InfoItem {
	    /// title for the info element
	    std::string name;
	    /// value for the info element
	    std::string value;
	};

	/// @brief Type of availlable output formats
	enum OutputFormat {
	    /// Human friendly
	    HUMAN = 0,
	    /// CSV format without a header row
	    CSVWITHOUTHEADERS = 1,
	    /// CSV format with a header row
	    CSVWITHHEADERS = 2
	};

	InfoItems();
	virtual ~InfoItems();

	/**
	 * @brief Adds an InfoItem (struct with a name and value) to the vector of InfoItem elements (value is a string)
	 *
	 * @param[in] name string value for InfoItem name field
	 * @param[in] value string value for InfoItem value field
	 */
	void addInfoItem(const std::string &name, const std::string &value) {
	    infoItemsVector.push_back({name, value});
	}
	void addInfoItem(std::string name, unsigned int value);
	void addInfoItem(std::string name, unsigned long value);
	void addInfoItem(std::string name, double value, unsigned int numDigitsPrec);
	/// @brief Erases InfoItem vector contents
	void clear() {
	    infoItemsVector.resize(0);
	}
	void append(const std::unique_ptr<InfoItems> pNewInfoItems);
	std::string to_string(OutputFormat outputFormat);
	void saveOrPrint(OutputFormat outputFormat, std::string outputFileName = "");
    private:
	/// Vector of InfoItem elements
	std::vector<InfoItem> infoItemsVector;
};

} /* namespace OpenCLIPER */

#endif /* INCLUDE_OPENCLIPER_INFOITEMS_HPP_ */
