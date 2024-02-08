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

#ifndef LPISUPPORT_PROGRAMCONFIG_H
#define LPISUPPORT_PROGRAMCONFIG_H
#include <LPISupport/Utils.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <map>
#include <getopt.h>
#include <sstream> // string stream


namespace LPISupport {

/**
 * @brief Class with common variables and methods for program configuration (base for derived classes specific to testing programs)
 *
 */
class ProgramConfig {
    public:
	/// Data type for the map with names of program arguments and their parameters
	typedef std::map<std::string, std::string> ExecArgsMap;

	/**
	 * @brief Struct for basic program configuration
	 */
	struct ConfigTraits {
	    /// Name of the program
	    std::string     programName = "";

	    /// First line of program usage (summary)
	    std::string     usageSummary = "[options]";

	    /// Rest of line of program usage (with valid options and their parameters)
	    std::string     usageOptions = "-h\tshow this help\n";

	    /// Supported short arguments string (one letter for every supported option, in standard getopt function format)
	    std::string     shortArgs = "h";

	    /// Short required supported arguments string (not all the short arguments of previous field have to be mandatory)
	    std::string     shortRequiredArgs = "";

	    /// Map for storing short arguments read from program command line
	    ProgramConfig::ExecArgsMap execArgsMap;

	    /// Vector with non-option run arguments
	    std::vector<std::string> nonOptionArgs;

	    ConfigTraits() {}

	    /// Destructor for object
	    virtual ~ConfigTraits() {}

	    void addSupportedShortOption(char optionName, std::string optionParameterName, std::string explanation, bool mandatory);

	    virtual void configure();

	    std::string getUsage();
	};

	ProgramConfig(int argc, char* argv[], std::string extraSummary = "");

	virtual ~ProgramConfig();

	/**
	 * @brief Returns the configuration object for this object
	 *
	 * @return smart pointer to the configuration object
	 */
	virtual std::shared_ptr<ConfigTraits> getConfigTraits() {
	    return pConfigTraits;
	}

	static ExecArgsMap readProgramArguments(int argc, char* argv[], struct option longOptions[], std::string shortArgs,
						std::string shortRequiredArgs, std::string usage);

	static ExecArgsMap readProgramShortArguments(int argc, char* argv[], std::string shortArgs, std::string shortRequiredArgs,
		std::string usage, std::vector<std::string>* pNonOptionArgs);

	static std::string fileName(std::string path);

	static std::string exeFile();
	static std::string exeDir();

    protected:
	ProgramConfig();
	void init(int argc, char* argv[], std::string extraSummary);
	void readExecArgs(int argc, char* argv[]);

	void checkRequiredArgsPresent();

	/// Smart pointer to configuration object
	std::shared_ptr<ConfigTraits> pConfigTraits = nullptr;

    private:

};
} /* namespace LPISupport */

#endif // LPISUPPORT_PROGRAMCONFIG_H
