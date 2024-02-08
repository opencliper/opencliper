/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2018  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <LPISupport/PerfTestConfResult.hpp>

// Uncomment to show class-specific debug messages
//#define PERFTESTCONFRESULT_DEBUG

#if !defined NDEBUG && defined PERFTESTCONFRESULT_DEBUG
    #define PERFTESTCONFRESULT_CERR(x) CERR(x)
#else
    #define PERFTESTCONFRESULT_CERR(x)
    #undef PERFTESTCONFRESULT_DEBUG
#endif

namespace LPISupport {

PerfTestConfResult::PerfTestConfResult() {

}

PerfTestConfResult::PerfTestConfResult(int argc, char* argv[], std::string extraSummary) {
    pConfigTraits = std::make_shared<ConfigTraits>();
    init(argc, argv, extraSummary);
}

/**
 * @brief Class destructor (empty)
 *
 */
PerfTestConfResult::~PerfTestConfResult() {

}

/**
 * @brief Sets configuration fields of pConfigTraits configuration object) from map of read program arguments field.
 *
 * It also calls setSpecificConfig config method defined by subclasses (includes configuration tasks specific of subclasses).
 */
void PerfTestConfResult::ConfigTraits::configure() {
    ProgramConfig::ConfigTraits::configure();
    unsigned long outputFormatInteger;
    // Visible ConfigTraits struct is this class ConfigTraits
    //for (auto& mapElement: execArgsMap) {// segmentation fault in iteration after erase in debian 10 gcc 8
    for(ExecArgsMap::const_iterator pMapElement = execArgsMap.cbegin() ; pMapElement != execArgsMap.cend() ;) {
	char option = pMapElement->first.at(0);
	switch(option) {
	    case 'n':
		deviceName = pMapElement->second;
		pMapElement = execArgsMap.erase(pMapElement);
		break;
	    case 'r':
		repetitions = stoul(pMapElement->second);
		pMapElement = execArgsMap.erase(pMapElement);
		break;
	    case 'f':
		outputFormatInteger = stoul(pMapElement->second);
		switch(outputFormatInteger) {
		    case(InfoItems::OutputFormat::CSVWITHOUTHEADERS) :
			outputFormat = InfoItems::OutputFormat::CSVWITHOUTHEADERS;
			break;
		    case(InfoItems::OutputFormat::CSVWITHHEADERS) :
			outputFormat = InfoItems::OutputFormat::CSVWITHHEADERS;
			break;
		    default:
			outputFormat = InfoItems::OutputFormat::HUMAN;
			break;
		}
		pMapElement = execArgsMap.erase(pMapElement);
		break;
	    case 'o':
		outputFileName = pMapElement->second;
		pMapElement = execArgsMap.erase(pMapElement);
		break;
	    default:
		pMapElement = std::next(pMapElement);
		break;
	}
    }
}

/**
 * @brief Converts test program results to a string representation (output format depends on configuration object)
 *
 */
std::string PerfTestConfResult::to_string(unsigned int index) {
    auto pSelfConfigTraits = std::dynamic_pointer_cast<ConfigTraits>(pConfigTraits);
    return pInfoItems->at(index).to_string(pSelfConfigTraits->outputFormat);
}

/**
 * @brief Saves or prints output information (if file name stored in configuration object is empty)
 */
void PerfTestConfResult::saveOrPrint() {
    auto pSelfConfigTraits = std::dynamic_pointer_cast<ConfigTraits>(pConfigTraits);
    std::string completeFileName = "";
    if (pSelfConfigTraits->outputFileName.empty()) {
    	for (unsigned int i=0; i< pInfoItems->size();i++) {
    		pInfoItems->at(i).saveOrPrint(pSelfConfigTraits->outputFormat);
    	}
    } else {
    	for (unsigned int i=0; i< pInfoItems->size();i++) {
    		completeFileName = LPISupport::Utils::basename(pSelfConfigTraits->outputFileName) + "_" +
    				pSelfConfigTraits->fileNameSuffixList.at(i) + "." +
					LPISupport::Utils::extensionname(pSelfConfigTraits->outputFileName);
        	pInfoItems->at(i).saveOrPrint(pSelfConfigTraits->outputFormat, completeFileName);
    	}
    }
}


/**
 * @brief Builds summary info of the test program
 *
 * @param[in] pSamples pointer to sample collection of measurements of the test
 */
void PerfTestConfResult::buildTestInfo(std::shared_ptr<SampleCollection> pSamples, void* extraInfo) {
    this->buildInitialCommonInfo();
    this->buildSpecificInfo(extraInfo);
    this->buildFinalCommonInfo(pSamples);
}

/**
 * @brief Builds first part of summary info (information about device used for the test)
 */
void PerfTestConfResult::buildInitialCommonInfo() {
    auto pSelfConfigTraits = std::dynamic_pointer_cast<ConfigTraits>(pConfigTraits);
    InfoItems localInfoItems;
    localInfoItems.addInfoItem("Test name", pConfigTraits->programName);
    const unsigned int HOST_NAME_MAX = 50;
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    localInfoItems.addInfoItem("Host", hostname);
    pInfoItems->push_back(localInfoItems);
}

void PerfTestConfResult::buildSpecificInfo(void* extraInfo) {
    auto pSelfConfigTraits = std::dynamic_pointer_cast<ConfigTraits>(pConfigTraits);
    pInfoItems->back().addInfoItem("Device name", pSelfConfigTraits->deviceName);
};

/**
 * @brief Builds last part of summary info (information about number of iterations, number of operations and MFLOPS for the test)
 * @param pSamples smart shared pointer to a collection of samples of the performance parameter measured
 */
void PerfTestConfResult::buildFinalCommonInfo(std::shared_ptr<SampleCollection> pSamples) {
    auto pSelfConfigTraits = std::dynamic_pointer_cast<ConfigTraits>(pConfigTraits);
    std::string suffixWithoutSpaces = pSamples->getSampleName();
    std::replace(suffixWithoutSpaces.begin(), suffixWithoutSpaces.end(), ' ', '_');
    pSelfConfigTraits->fileNameSuffixList.push_back(suffixWithoutSpaces);
    pInfoItems->back().addInfoItem("Number of iterations", std::to_string(pSamples->getNumOfSamples()));
    pInfoItems->back().append(pSamples->to_infoItems(pSelfConfigTraits->numDigitsPrec));
    if (pSelfConfigTraits->numOpsPerCalc != 0) {
    	pInfoItems->back().addInfoItem("Number of operations (per iteration)", pSelfConfigTraits->numOpsPerCalc);
    	double MFLOPS = pSelfConfigTraits->numOpsPerCalc / 1e6 / pSamples->getMean();
    	pInfoItems->back().addInfoItem("Throughput (MFLOPS)", MFLOPS, pSelfConfigTraits->numDigitsPrec);
    }
}
} /* namespace LPISupport */
#undef PERFTESTCONFRESULT_DEBUG

