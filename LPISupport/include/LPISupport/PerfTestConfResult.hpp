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
#ifndef INCLUDE_OPENCLIPERPERFORMANCETEST_HPP
#define INCLUDE_OPENCLIPERPERFORMANCETEST_HPP

#include <unistd.h>
#include <LPISupport/Utils.hpp>
#include <memory>
#include <LPISupport/SampleCollection.hpp>
#include <LPISupport/InfoItems.hpp>
#include <LPISupport/ProgramConfig.hpp>
#include <iostream>

namespace LPISupport {
/**
 * @brief Abstract class with common variables and methods for test program configuration (base for derived classes specific to testing programs)
 *
 */
class PerfTestConfResult : public virtual ProgramConfig {
    public:
	/**
	 * @brief Struct for object configuration
	 *
	 */
	struct ConfigTraits: public virtual ProgramConfig::ConfigTraits {
	    /// Number of precision digits for float numbers
	    unsigned int       numDigitsPrec = 12;
	    /// Device type information
	    std::string        deviceType = "";
	    /// Device name
	    std::string        deviceName = "";
	    /// Number of repetitions of the test
	    unsigned int       repetitions = 1;
	    /// Number of opereations per 1 repetition of the test
	    unsigned long       numOpsPerCalc = 0.0;
	    /// Output representation format
	    InfoItems::OutputFormat outputFormat = InfoItems::OutputFormat::HUMAN;
	    /// Name of the output file
	    std::string        outputFileName = "";
	    std::vector<std::string> fileNameSuffixList;

	    //DeviceTraits(DeviceType t=DEVICE_TYPE_ANY,cl::QueueProperties p=cl::QueueProperties::None): type(t),queueProperties(p) {}
	    /// Destuctor for the class
	    ConfigTraits() {
		addSupportedShortOption('n', "deviceName", "set device name for output summary", false);
		addSupportedShortOption('r', "repetitions", "set number of repetitions", false);
		addSupportedShortOption('o', "outputFileName", "set output file name", false);
		addSupportedShortOption('f', "outputFormat",
					"set output format: 0 -> human-readable format, 1 -> csv format without headers, 2 -> csv format with headers",
					false);
	    }
	    virtual ~ConfigTraits() {}
	    virtual void configure();
	};

	PerfTestConfResult(int argc, char* argv[], std::string extraSummary = "");
	virtual ~PerfTestConfResult();
	virtual void buildTestInfo(std::shared_ptr<SampleCollection> pSamples, void* extraInfo = nullptr);
	/**
	 * @brief Returns a smart pointer to an InfoItems object containing test program summary output
	 *
	 * @return Smart pointer to InfoItems object
	 */
	std::shared_ptr<std::vector<InfoItems>> to_infoItems() {
	    return pInfoItems;
	}
	std::string to_string(unsigned int index = 0);
	void saveOrPrint();

    protected:
	PerfTestConfResult();

	/// Pointer to InfoItems object storing information of the test result
	std::shared_ptr<std::vector<InfoItems>> pInfoItems  = std::make_shared<std::vector<InfoItems>>();

    private:
	void buildInitialCommonInfo();
	/// @brief Pure virtual method for specific test information (must be implemented in subclasses)
	virtual void buildSpecificInfo(void* extraInfo = nullptr);
	void buildFinalCommonInfo(std::shared_ptr<SampleCollection> pSamples);

};
} /* namespace LPISUPPORT */
#endif // INCLUDE_LPISUPPORT_PERFORMANCETEST_HPP
