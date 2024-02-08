#include "PerformanceTestArrayOpParallel.hpp"

PerformanceTestArrayOpParallel::PerformanceTestArrayOpParallel() {

}

PerformanceTestArrayOpParallel::PerformanceTestArrayOpParallel(int argc, char* argv[]) {
    pConfigTraits = std::make_shared<ConfigTraits>();
    readExecArgs(argc, argv);
    checkRequiredArgsPresent();
    pConfigTraits->configure();
}

PerformanceTestArrayOpParallel::~PerformanceTestArrayOpParallel() {
}

void PerformanceTestArrayOpParallel::ConfigTraits::configure() {
    LPISupport::PerfTestConfResult::ConfigTraits::configure();
    // for (auto& mapElement: execArgsMap) {// segmentation fault in iteration after erase in debian 10 gcc 8
    for(ExecArgsMap::const_iterator pMapElement = execArgsMap.cbegin() ; pMapElement != execArgsMap.cend() ;) {
	char option = pMapElement->first.at(0);
	std::string reqarg = pMapElement->second;
	switch(option) {
	    case 'a':
		size = stoul(reqarg);
		pMapElement = execArgsMap.erase(pMapElement);
		break;
	    case 'b':
		dimBlockOrLocalSize = stoul(reqarg);
		pMapElement = execArgsMap.erase(pMapElement);
		break;
	    default:
		pMapElement = std::next(pMapElement);
	}
    }
}

void PerformanceTestArrayOpParallel::buildSpecificInfo(void* extraInfo) {
    auto pSelfConfigTraits = std::dynamic_pointer_cast<ConfigTraits>(pConfigTraits);
    pInfoItems->back().addInfoItem("Dim grid / global size", pSelfConfigTraits->dimGridOrGlobalSize);
    pInfoItems->back().addInfoItem("Dim block / local size", pSelfConfigTraits->dimBlockOrLocalSize);
    pInfoItems->back().addInfoItem("Data size", pSelfConfigTraits->size);
}
