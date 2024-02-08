#include "OpenCLIPER/processes/ApplyMask.hpp"

#include <OpenCLIPER/hostKernelFunctions.hpp>
#include <OpenCLIPER/CLapp.hpp>

// Uncomment to show class-specific debug messages
//#define APPLYMASK_DEBUG

#if !defined NDEBUG && defined APPLYMASK_DEBUG
    #define APPLYMASK_CERR(x) CERR(x)
#else
    #define APPLYMASK_CERR(x)
    #undef APPLYMASK_DEBUG
#endif

namespace OpenCLIPER {

void ApplyMask::init() {
}

void ApplyMask::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

    checkCommonLaunchParameters();
    try {
		if(pLP->samplingMasksData == nullptr)
			BTTHROW(std::invalid_argument("Empty sampling masks data"), "ApplyMask::launch");

		cl::Buffer* deviceBuffer = getInput()->getDeviceBuffer();
		cl::Buffer* pSamplingMasks = pLP->samplingMasksData->getDeviceBuffer();

		uint nDArrayTotalSize = getInput()->getNDArrayTotalSize(0);
		cl::NDRange localSizes = cl::NDRange();

		cl::NDRange globalSizes = cl::NDRange(nDArrayTotalSize);
                
		if (this->getInput()->getElementDataType() == TYPEID_COMPLEX) {
		    kernel = getApp()->getKernel("applyMask_complex");
		} else if (this->getInput()->getElementDataType() == TYPEID_REAL) {
			kernel = getApp()->getKernel("applyMask_real");
		} else {
			BTTHROW(std::invalid_argument("Element data type not supported"), "ApplyMask::launch()")
		}
		kernel.setArg(0, *deviceBuffer);
		kernel.setArg(1, *pSamplingMasks);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSizes, localSizes, NULL, NULL);
    }
    catch(cl::Error& err) {
    	BTTHROW(CLError(err), "ApplyMask::launch()");
    }
}

} // namespace OpenCLIPER
