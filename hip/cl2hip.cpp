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

#ifdef HAVE_HIP

#include <OpenCLIPER/cl2hip.hpp>
#include <hcc/hc_am.hpp>

// Uncomment to show class-specific debug messages
//#define CL2HIP_DEBUG

#if !defined NDEBUG && defined CL2HIP_DEBUG
    #define CL2HIP_CERR(x) CERR(x)
#else
    #define CL2HIP_CERR(x)
    #undef CL2HIP_DEBUG
#endif

/// @brief Constructs a HIP object that points to a given CL object which already resides in a CL device
void* cl2hip(cl::Memory clPointer, cl::Context context, cl::CommandQueue queue, cl::Kernel kernel, hipDevice_t hipDevice) {

    // We assume that device pointers on HIP compatible devices are always 64bit
    cl::Buffer pointerBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_ulong), NULL, NULL);

    // Get raw device pointer via kernel
    cl::NDRange globalSize = {1};
    kernel.setArg(0, clPointer);
    kernel.setArg(1, pointerBuffer);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize);
    queue.finish();

    // Read back raw pointer and release pointer buffer
    void* hipPointer;
    queue.enqueueReadBuffer(pointerBuffer, CL_TRUE, 0, sizeof(void*), &hipPointer);

    // Register raw pointer in HIP memtracker
    hc::accelerator acc;
    hc::AmPointerInfo ampi(NULL, hipPointer, hipPointer, sizeof(hipPointer), acc, true, false);
    am_status_t am_status = hc::am_memtracker_add(hipPointer, ampi);
    hc::am_memtracker_update(hipPointer, hipDevice, 0);


    return hipPointer;
}

#undef CL2HIP_DEBUG

#endif // HAVE_HIP
