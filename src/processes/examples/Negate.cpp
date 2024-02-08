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
#include <OpenCLIPER/processes/examples/Negate.hpp>

// Uncomment to show class-specific debug messages
//#define NEGATE_DEBUG

#if !defined NDEBUG && defined NEGATE_DEBUG
    #define NEGATE_CERR(x) CERR(x)
#else
    #define NEGATE_CERR(x)
    #undef NEGATE_DEBUG
#endif

namespace OpenCLIPER {
void Negate::init() {
    kernel = getApp()->getKernel("negate");
}

void Negate::launch() {
    checkCommonLaunchParameters();
    try {
	// Set input and output OpenCL buffers on device memory
	cl::Buffer* pInBuf = getInput()->getDeviceBuffer();
	cl::Buffer* pOutBuf = getOutput()->getDeviceBuffer();

	// Set kernel parameters
	kernel.setArg(0, *pInBuf);
	kernel.setArg(1, *pOutBuf);

	// Set kernel work items size: number of pixels to process is image width x height
	cl::NDRange globalSizes = {NDARRAYWIDTH(getInput()->getNDArray(0))* NDARRAYHEIGHT(getInput()->getNDArray(0))};
#if !defined NDEBUG && defined NEGATE_DEBUG
	dimIndexType deviceMemBaseAddrAlignInBytes = getApp()->getDevice().getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() / 8;
#endif
	NEGATE_CERR("Negate process, deviceMemBaseAddrAlignInBytes: " << deviceMemBaseAddrAlignInBytes << std::endl);
	NEGATE_CERR("Negate process, sizeof(dimIndexType): " << sizeof(dimIndexType) << std::endl);
	NEGATE_CERR("Negate process, sizeof(realType): " << sizeof(realType) << std::endl);
#if !defined NDEBUG && defined NEGATE_DEBUG
	cl::NDRange offset = {CLapp::roundUp(sizeof(dimIndexType), deviceMemBaseAddrAlignInBytes) / sizeof(realType)};
#endif
	NEGATE_CERR("Negate process, kernel offset in bytes / size of realType: " << offset[0] << std::endl);
	NEGATE_CERR("Negate process, getting dimsAndStridesArray from host buffer (not vector)... ");

#if !defined NDEBUG && defined NEGATE_DEBUG
	const cl_uint* dimsAndStridesArray = (const cl_uint*)(getInput()->getDataDimsAndStridesHostBuffer());
#endif
	NEGATE_CERR("Done.\n");
	NEGATE_CERR("======== From Negate process ========\nnsd: " << dimsAndStridesArray[0] << "\nallSizesEqual: " << dimsAndStridesArray[1] << "\nnumCoils: " << dimsAndStridesArray[2] <<
		    "\nntd: " << dimsAndStridesArray[3] << "\nTD0: " << dimsAndStridesArray[4] << "\nSD0: " << dimsAndStridesArray[5] << "\nSD1: " << dimsAndStridesArray[6] <<
		    "\n=====================================\n");
	// Execute kernel
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSizes, cl::NDRange(), NULL, NULL);
	//queue.enqueueNDRangeKernel(kernel, offset, globalSizes, cl::NDRange(), NULL, NULL);
    } catch(cl::Error& err) {
	BTTHROW(CLError(err), "Negate::launch");
    }
}
}
#undef NEGATE_DEBUG
