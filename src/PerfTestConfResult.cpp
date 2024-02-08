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

#include <OpenCLIPER/PerfTestConfResult.hpp>

// Uncomment to show class-specific debug messages
//#define PERFTESTCONFRESULT_DEBUG

#if !defined NDEBUG && defined PERFTESTCONFRESULT_DEBUG
    #define PERFTESTCONFRESULT_CERR(x) CERR(x)
#else
    #define PERFTESTCONFRESULT_CERR(x)
    #undef PERFTESTCONFRESULT_DEBUG
#endif

namespace OpenCLIPER {

/**
 * @brief Class constructor (empty)
 *
 */

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
    LPISupport::PerfTestConfResult::ConfigTraits::configure();
    OpenCLIPER::ProgramConfig::ConfigTraits::configure();
}

void PerfTestConfResult::buildSpecificInfo(void* extraInfo) {
    auto pSelfConfigTraits = std::dynamic_pointer_cast<ConfigTraits>(pConfigTraits);
    std::shared_ptr<CLapp> pCLapp;
    std::string deviceVendor = "";
    std::string deviceName;
    std::string deviceTypeName;
    if(extraInfo != nullptr) {
	pCLapp = * (static_cast<std::shared_ptr<CLapp>*>(extraInfo));
	cl::Device device = pCLapp->getDevice();
	deviceName = device.getInfo<CL_DEVICE_NAME>();
	deviceVendor = device.getInfo<CL_DEVICE_VENDOR>();
	cl_device_type devType;
	device.getInfo(CL_DEVICE_TYPE, &devType);
	if(devType & CL_DEVICE_TYPE_DEFAULT)
	    deviceTypeName = "DEFAULT";
	if(devType & CL_DEVICE_TYPE_CPU)
	    deviceTypeName = "CPU";
	if(devType & CL_DEVICE_TYPE_GPU)
	    deviceTypeName = "GPU";
	if(devType & CL_DEVICE_TYPE_ACCELERATOR)
	    deviceTypeName = "ACCELERATOR";
#ifdef CL_VERSION_1_2
	if(devType & CL_DEVICE_TYPE_CUSTOM)
	    deviceTypeName = "CUSTOM";
#endif //CL_VERSION_1_2

	device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &pSelfConfigTraits->numOfComputeUnits);
	device.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &pSelfConfigTraits->clockFreq);
	device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &pSelfConfigTraits->globalMemSizeBytes);

	if(devType & CL_DEVICE_TYPE_CPU) {  // Device is CPU
	    pSelfConfigTraits->warpOrWavefrontSize = 1;
	}
	else {
	    cl::Kernel& kernel = pCLapp->getKernel();
	    pSelfConfigTraits->warpOrWavefrontSize =
		kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
	}
	pInfoItems->back().addInfoItem("Device type", deviceTypeName);
	pInfoItems->back().addInfoItem("Device vendor", deviceVendor);
	pInfoItems->back().addInfoItem("Device name", deviceName);
	pInfoItems->back().addInfoItem("Number of compute units", pSelfConfigTraits->numOfComputeUnits);
	pInfoItems->back().addInfoItem("Warp/wavefront size", pSelfConfigTraits->warpOrWavefrontSize);
	pInfoItems->back().addInfoItem("Compute units x Warp/wavefront size", pSelfConfigTraits->numOfComputeUnits * pSelfConfigTraits->warpOrWavefrontSize);
	pInfoItems->back().addInfoItem("Clock frequency (MHz)", pSelfConfigTraits->clockFreq);
	pInfoItems->back().addInfoItem("Global memory size (Megabytes)", pSelfConfigTraits->globalMemSizeBytes / 1024 / 1024);
	pInfoItems->back().addInfoItem("Device score", (unsigned long) CLapp::score(pCLapp->getDevice()));

    }
    else {   // pCLapp not passed as extra info parameter
	deviceName = pSelfConfigTraits->deviceName;
	deviceTypeName = pSelfConfigTraits->deviceType;
	pInfoItems->back().addInfoItem("Device type", deviceTypeName);
	pInfoItems->back().addInfoItem("Device name", deviceName);
    }
}

} /* namespace OpenCLIPER */
#undef PERFTESTCONFRESULT_DEBUG

