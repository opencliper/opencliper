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
/**
 * @file
 * @brief File with macros and type definitions used by several classes and CL kernels
 */
#ifndef INCLUDE_OPENCLIPER_DEFS_HPP
#define INCLUDE_OPENCLIPER_DEFS_HPP

// These must be defined before including CL/cl.h
#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120


/****************************************************************************************
 * Definitions for C++ source code
 ****************************************************************************************/
#ifdef __cplusplus // OpenCL kernels may include this header and they only support C language syntax

#include <OpenCLIPER/buildconfig.hpp>
#include <CL/cl.h>
#include <complex>
#include <vector>
#include <set>
#include <exception>
#include <stdexcept>	//g++ 4.8 doesn't seem to include <stdexcept> from <exception>
#include <memory> // for smart pointers
#include <chrono> // Execution times measurement
#include <atomic>
#include <stdint.h>
// Needed by libmatio, library for reading matlab data files (.mat)
#include <matio.h>
#include <typeindex>
#include <assert.h>

// Calculus using floating point double precision if label defined
//#define DOUBLE_PREC

#ifdef NDEBUG
    /// Options for OpenCL command queues in release mode
    #define COMMANDQUEUEOPTIONS cl::QueueProperties::None
#else
    /// Options for OpenCL command queues in debug mode
    #define COMMANDQUEUEOPTIONS cl::QueueProperties::Profiling
#endif

/// Number of precision digits for time values using in profiling code
#define PROFILINGTIMESPRECISION 12

/// Type used for storing complex values
#define complexType std::complex<realType>

/// String with prefix for error messages (the name of the namespace)
#define ERRORNAMESPACEPREFIX "OpenCLIPER::"

/// Type for specifying fragments of images
#define FragmentSpecif uint8_t

/// Handle for a process bound to an OpenCL context
typedef unsigned int ProcessHandle;

/// Value for an invalid process handle
#define INVALIDPROCESSHANDLE 0

/// Value for first valid process handle
#define FIRSTVALIDPROCESSHANDLE (INVALIDPROCESSHANDLE+1)

/// Handle for a Data object bound to an OpenCL context
typedef uint64_t DataHandle;

/// Value for an invalid data handle
#define INVALIDDATAHANDLE 0

/// Value for first valid data handle
#define FIRSTVALIDDATAHANDLE (INVALIDDATAHANDLE+1)

/// data type for variables storing number of coils or coils index
typedef uint32_t numCoilsType;

/// data type to tell which coils are used in an acquisition
typedef std::set<bool> usedCoilsType;

/// data type for variables storing a 1-dimensional index calculated from indexes of several dimensions
typedef uint32_t index1DType;

/// data type for variables storing an index from a several dimensions image
typedef uint32_t dimIndexType;

/// data type for variables storing the number of dimensions of an image
typedef uint32_t numberOfDimensionsType;

// macros for element data types
#define TYPEID_COMPLEX std::type_index(typeid(complexType))
#define TYPEID_REAL std::type_index(typeid(realType))
#define TYPEID_INDEX std::type_index(typeid(dimIndexType))
#define TYPEID_CL_UCHAR std::type_index(typeid(cl_uchar))

// common names for CFL format files
#define CFLSensMapsSuffix "_SensitivityMaps.cfl"
#define CFLSampMasksSuffixRowMask "_SamplingMasks.u32"
#define CFLSampMasksSuffixPixelMask "_SamplingMasks.u8"
#define CFLHeaderExtension ".hdr"

/// Main namespace of the OpenCLIPER framework
namespace OpenCLIPER {
/**
* @brief Enumerated type for device memory formats
*
* Enumerated type for describing device memory formats for storing data. Used by
* host2device method to get format of data to store in device memory and by
* device2host method to get format of data to be read from device memory.
*/
enum class SyncSource {
    /// Both buffer and image formats used (not valid for device2host method)
    ALL,
    /// Buffer format used
    BUFFER_ONLY,
    /// Image format used
    IMAGE_ONLY
};

/// Default source of data (buffers, images, etc.)
#define SYNCSOURCEDEFAULT SyncSource::BUFFER_ONLY
}

// Complex to real conversion types
enum ComplexPart {
    REAL,	// Get real part, discard imaginary one
    IMAG,	// Discard real part, get imaginary one
    ABS,	// Calculate modulus
    ARG	// Calculate argument
};

/// Macro for printing a vector <i>vector</i> of type <i>baseType</i> prefixed by title <i>title</i>
#define PRINTVECTOR(labelText, vector, baseType) CERR((labelText) << ": ");\
	    std::copy((vector).begin(), (vector).end(), std::ostream_iterator<baseType>(std::cerr, " ")); CERR(std::endl);
/// Macro for printing an array <i>array</i> of size <i>size</i> prefixed by title <i>title</i>
#define PRINTARRAY(labelText, array, size) CERR(labelText << ": "); for (uint i = 0; i < (size); i++)  CERR((array)[i] << " ");

/// @brief Data type of NDArray elements
typedef std::type_index ElementDataType;

#ifndef NDEBUG
// In debug mode, insert a clFinish() after each call to clEnqueueNDRangeKernel so that exceptions are caught at the right kernel
// (this has to be combined with -Wl,wrap compiler option)
extern "C" {
    cl_int __real_clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset,
					 const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list,
					 const cl_event* event_wait_list, cl_event* event);

    cl_int __wrap_clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset,
					 const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list,
					 const cl_event* event_wait_list, cl_event* event);
}
#endif

#else // __cplusplus


/****************************************************************************************
 * Definitions for CL source code
 ****************************************************************************************/
#ifdef DOUBLE_PREC
    typedef double2 complexType;
#else
    typedef float2 complexType;
#endif

#endif // __cplusplus


/****************************************************************************************
 * Unconditional definitions (must be valid in C++ and in CL sources)
 ****************************************************************************************/
#ifdef DOUBLE_PREC
    /// Type for storing float point values
    typedef double realType;
#else
    /// Type for storing float point values
    typedef float realType;
#endif

#ifdef DOUBLE_PREC
    #define IMAGEELELEMENTTYPE double4
#else
    /// Data type for an OpenCL image element
    #define IMAGEELEMENTTYPE float4
#endif

/// Size of data type used for OpenCL vector operations (in number of elements)
#define VECTORDATATYPESIZE 16

enum DimsPos {
    /// Position of width value in dimensions vector
    WIDTHPOS = 0,
    /// Position of height value in dimensions vector
    HEIGHTPOS = 1,
    /// Position of depth value in dimensions vector
    DEPTHPOS = 2
};

/// Macro for getting width of an ndarray object (returns 1 if that dimension is not defined)
#define NDARRAYWIDTH(ndarray) (((ndarray)->getNDims() > WIDTHPOS) ? ((ndarray)->getDims()->at(WIDTHPOS)) : (1))
/// Macro for getting height of an ndarray object (returns 1 if that dimension is not defined)
#define NDARRAYHEIGHT(ndarray) (((ndarray)->getNDims() > HEIGHTPOS) ? ((ndarray)->getDims()->at(HEIGHTPOS)) : (1))
/// Macro for getting depth of an ndarray object (returns 1 if that dimension is not defined)
#define NDARRAYDEPTH(ndarray) (((ndarray)->getNDims() > DEPTHPOS) ? ((ndarray)->getDims()->at(DEPTHPOS)) : (1))
/// Macro for getting first NDArray from a Data object
#define PNDARRAY0 getData()->at(0)

/// Type of array with Data dimensions information
typedef uint* dimsInfo_t;

/// Type of array with Data stride dimensions information
typedef uint* stridesInfo_t;

/// Minimum number of positions of dimensions info array
#define NUMINITIALPOSITIONSDIMSINFO 4
/// Known positions for array storing data dimensions (spatial, temporal and number of coils)
enum knownDimPos {NumSpatialDimsPos = 0, AllSizesEqualPos = 1, NumCoilsPos = 2, NumTemporalDimsPos = 3, FirstTemporalDimPos = 4};
/// Known positions for array storing data strides (spatial, temporal and number of coils)
// Note: stride position for element and column are the same thing (sure!)
enum knownStridePos {ElementStridePos = 0, FirstSpatialStridePos = 0};

/// Known positions for spatial dimensions in array storing data dimensions
enum knownSpatialDimPos {COLUMNS = WIDTHPOS, ROWS = HEIGHTPOS, SLICES = DEPTHPOS};

//typedef cl_uint indexType;

/// Macro for calculation of modulus of a complex number
#define MOD(complex) sqrt(pow((complex).real(), 2) + pow((complex).imag(), 2))
/// Macro for calculation of modulus of a real number
#define MODREALS(real1, real2) sqrt(pow((real1), 2) + pow((real2), 2))
/// Format used for storing complex numbers as image data with two color channels (one channel for the real part and the other for the imaginary part)
#define OPENCLCHANNELFORMAT CL_RG

/// Macro for declaring bT (begin time) variable and storing current time in it
#define BEGIN_TIME(bT) std::chrono::high_resolution_clock::time_point bT = \
                                      std::chrono::high_resolution_clock::now();
/// Macro for declaring eT (end time) variable and storing current time in it
#define END_TIME(eT) std::chrono::high_resolution_clock::time_point eT = \
                                      std::chrono::high_resolution_clock::now();
/// Data type for a number storing the difference between two time values
#define TIME_DIFF_TYPE double
/// Macro for calculation of time difference betweent eT and bT and storing result in diffT
#define TIME_DIFF(diffT, bT,eT) diffT = \
(std::chrono::duration_cast<std::chrono::nanoseconds>(eT - bT).count()) / 1e9

/// @brief File with common (kernel/host) utility functions
#define HOST_KERNEL_FUNCTIONS_FILE "hostKernelFunctions.cl"

/// @brief File with internal kernels
#define INTERNAL_KERNELS_FILE "internalKernels.cl"

#endif // INCLUDE_OPENCLIPER_DEFS_HPP
