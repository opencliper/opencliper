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
 *
 * FFT.cpp
 *
 *  Created on: 23 de nov. de 2016
 *      Author: fedsim
 */

#include <OpenCLIPER/processes/FFT.hpp>
#include <OpenCLIPER/CLapp.hpp>
#include <OpenCLIPER/hostKernelFunctions.hpp>
#include <LPISupport/InfoItems.hpp>
#include <OpenCLIPER/KData.hpp>
#include <OpenCLIPER/XData.hpp>
#include <OpenCLIPER/processes/ScalarMultiply.hpp>

// Uncomment to show class-specific debug messages
//#define FFT_DEBUG

#if !defined NDEBUG && defined FFT_DEBUG
    #define FFT_CERR(x) CERR(x)
#else
    #define FFT_CERR(x)
    #undef FFT_DEBUG
#endif

namespace OpenCLIPER {

void FFT::init() {
    auto pIP=std::dynamic_pointer_cast<InitParameters>(pInitParameters);
    if(!pIP) pIP=std::unique_ptr<InitParameters>(new InitParameters());

    FFT_CERR("FFT::init()\n");

    // -----------------------------------------------------------------------------------------------------------------------------------------
    // Initial error handling and sanity checks
    // -----------------------------------------------------------------------------------------------------------------------------------------

    // Add clFFT error codes to OpenCLIPER error handling. Ideally this should be done
    // only once per CLapp object, not once per FFT object but, anyway...
    getApp()->setOpenCLErrorCodeStr(CLFFT_BUGCHECK, "CLFFT_BUGCHECK");
    getApp()->setOpenCLErrorCodeStr(CLFFT_NOTIMPLEMENTED, "CLFFT_NOTIMPLEMENTED");
    getApp()->setOpenCLErrorCodeStr(CLFFT_TRANSPOSED_NOTIMPLEMENTED, "CLFFT_TRANSPOSED_NOTIMPLEMENTED");
    getApp()->setOpenCLErrorCodeStr(CLFFT_FILE_NOT_FOUND, "CLFFT_FILE_NOT_FOUND");
    getApp()->setOpenCLErrorCodeStr(CLFFT_FILE_CREATE_FAILURE, "CLFFT_FILE_CREATE_FAILURE");
    getApp()->setOpenCLErrorCodeStr(CLFFT_VERSION_MISMATCH, "CLFFT_VERSION_MISMATCH");
    getApp()->setOpenCLErrorCodeStr(CLFFT_INVALID_PLAN, "CLFFT_INVALID_PLAN");
    getApp()->setOpenCLErrorCodeStr(CLFFT_DEVICE_NO_DOUBLE, "CLFFT_DEVICE_NO_DOUBLE");
    getApp()->setOpenCLErrorCodeStr(CLFFT_DEVICE_MISMATCH, "CLFFT_DEVICE_MISMATCH");

#ifdef HAVE_ROCFFT
    // rocfft status codes are positive whereas CL codes are negative. Hopefully they won't clash!
    getApp()->setOpenCLErrorCodeStr(rocfft_status_success, "rocfft_status_success");
    getApp()->setOpenCLErrorCodeStr(rocfft_status_failure, "rocfft_status_failure");
    getApp()->setOpenCLErrorCodeStr(rocfft_status_invalid_arg_value, "rocfft_status_invalid_arg_value");
    getApp()->setOpenCLErrorCodeStr(rocfft_status_invalid_dimensions, "rocfft_status_invalid_dimensions");
    getApp()->setOpenCLErrorCodeStr(rocfft_status_invalid_array_type, "rocfft_status_invalid_array_type");
    getApp()->setOpenCLErrorCodeStr(rocfft_status_invalid_strides, "rocfft_status_invalid_strides");
    getApp()->setOpenCLErrorCodeStr(rocfft_status_invalid_distance, "rocfft_status_invalid_distance");
    getApp()->setOpenCLErrorCodeStr(rocfft_status_invalid_offset, "rocfft_status_invalid_offset");
#endif

    dimIndexType nSpatialDims = getInput()->getNumSpatialDims();
    dimIndexType nTotalDims = nSpatialDims + (getInput()->getNumCoils() != 0? 1:0) + getInput()->getNumTemporalDims();

    if(!getInput())
	BTTHROW(CLError(CL_INVALID_MEM_OBJECT, "init() called before setInputData()"), "FFT::init");

    if(!getOutput())
	BTTHROW(CLError(CL_INVALID_MEM_OBJECT, "init() called before setOutputData()"), "FFT::init");

    if(getInput()->getData()->size() != getOutput()->getData()->size())
	BTTHROW(CLError(CL_INVALID_MEM_OBJECT, "OpenCLIPER::FFT::launch(): inputData and outputData must have the same number of images"), "FFT::init");

    if(!getInput()->getAllSizesEqual())
	BTTHROW(CLError(CLFFT_NOTIMPLEMENTED, "FFT for variable-size data objects is not implemented at this time"), "FFT::init");

    // Hopefully nSpatialDims will fit in a signed int
    if((pIP->dim >= static_cast<int>(nSpatialDims)) || (pIP->dim < -1))
	BTTHROW(CLError(CLFFT_INVALID_ARG_VALUE, "Requested FFT along a nonexistent dimension"), "FFT::init");


    // -----------------------------------------------------------------------------------------------------------------------------------------
    // Set data FFT dimensions, sizes, strides, batch sizes and batch distances according to specified sampling mask (if set) and dimension to traverse (if set).
    // -----------------------------------------------------------------------------------------------------------------------------------------
    size_t fftDataSize[3];
    size_t strides[3];
    size_t batchSize = 1;
    size_t batchDistance;
    size_t FFTnDims = nSpatialDims;


    // If sampling mask is unset, we do the FFT at every row (if a transform across the first dimension is requested)
    if(!pIP->samplingMask) {

	// dim=-1: nothing special. One full transform per NDArray; one batch covering all NDArrays
	if(pIP->dim == -1) {

	    // Set strides and fftDataSize
	    for(unsigned i = 0; i < nSpatialDims; i++) {
		strides[i] = getInput()->getSpatialDimStride(i, 0);
		fftDataSize[i] = getInput()->getSpatialDimSize(i, 0);
	    }

	    // Set batchSize
	    for(unsigned i = nSpatialDims; i < nTotalDims; i++)
		batchSize *= getInput()->getDimSize(i, 0);

	    // Set batchDistance (getDimStride will return batchDistance=0 if there are spatial dimensions only)
	    batchDistance = getInput()->getDimStride(nSpatialDims,0);

	    // Set number of batches and their offsets
	    nBatches = 1;
	    batchOffsets.clear();
	    batchOffsets.push_back(0);
	}

	// dim=0: one 1D transform per row; one batch covering all dimensions but the first one
	else if(pIP->dim == 0) {
	    // We want 1D transforms, independently of how many dimensions the volume has
	    FFTnDims = 1;

	    // Set strides and fftDataSize
	    strides[0] = getInput()->getSpatialDimStride(0, 0);
	    fftDataSize[0] = getInput()->getSpatialDimSize(0, 0);

	    // Set batchSize (one 1D transform per row in the whole volume)
	    for(unsigned i = 1; i < nTotalDims; i++)
		batchSize *= getInput()->getDimSize(i, 0);

	    // Set batchDistance
	    batchDistance = getInput()->getDimStride(1, 0);

	    // Set number of batches and their offsets
	    nBatches = 1;
	    batchOffsets.clear();
	    batchOffsets.push_back(0);
	}

	// dim>=1: one 1D transform per line across the 'dim' dimension; numNDArrays batches covering one NDArray each
	else if(pIP->dim >= 1) {
	    // We want 1D transforms, independently of how many dimensions the volume has
	    FFTnDims = 1;

	    // Set strides and fftDataSize
	    strides[0] = getInput()->getSpatialDimStride(pIP->dim, 0);
	    fftDataSize[0] = getInput()->getSpatialDimSize(pIP->dim, 0);

	    // Set batchSize (one 1D transform per line along 'dim' in the whole volume)
	    batchSize = getInput()->getNumNDArrays();

	    // Set batchDistance
	    batchDistance = getInput()->getSpatialDimStride(nSpatialDims, 0);

	    // Set number of batches and their offsets (one batch per NDArray)
	    nBatches = getInput()->getSpatialDimSize(0, 0);

	    batchOffsets.clear();
	    size_t batch2batchDistance = 1;
	    size_t curOffset = 0;
	    for(unsigned i = 0; i < nBatches; i++) {
		batchOffsets.push_back(curOffset);
		curOffset += batch2batchDistance;
	    }
	}
    }

    // If samplingMask is set and a transform across the first dimension is requested, do batches of 1D transforms over the first non-spatial dimension for each row included in samplingMask
    else {
	BTTHROW(CLError(CLFFT_INVALID_ARG_VALUE, "AARGH! No sampling masks yet!"), "FFT::init");
    }

    // -----------------------------------------------------------------------------------------------------------------------------------------
    // Compose an FFT plan using rocFFT (if HIP is available)
    // -----------------------------------------------------------------------------------------------------------------------------------------

#ifdef HAVE_ROCFFT
    if(getApp()->getHIPDevice() != -1) {
	rocfft_result_placement rocfftPlace;
	if(getInput() == getOutput())
	    rocfftPlace = rocfft_placement_inplace;
	else
	    rocfftPlace = rocfft_placement_notinplace;

	size_t rocFFTnDims = FFTnDims;
	if((rocFFTnDims < 1) || (rocFFTnDims > 3))
	    BTTHROW(CLError(CL_INVALID_WORK_DIMENSION, "Only 1, 2 and 3-dimensional FFTs are supported"), "FFT::init");

	rocfft_status err;

	if((err = rocfft_setup()) != rocfft_status_success) {
	    errStr = "rocfft_setup: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}

	if((err = rocfft_plan_description_create(&rocfftPlanDescription)) != rocfft_status_success) {
	    errStr = "rocfft_plan_description_create: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}

	if((err = rocfft_plan_description_set_data_layout(rocfftPlanDescription, rocfft_array_type_complex_interleaved, rocfft_array_type_complex_interleaved, nullptr, nullptr,
		  nº strides in, strides in, batchDistance,
		  nº strides out, strides out, batchDistance)
	   ) != rocfft_status_success) {
	    errStr = "rocfft_plan_description_set_data_layout: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}

	// In rocFFT, the transform direction is embedded in the plan, so we create two plans instead of one
	if(((err = rocfft_plan_create(&rocPlanHandleFW, rocfftPlace, rocfft_transform_type_complex_forward, OPENCLIPER_ROCFFT_PRECISION, rocFFTnDims, fftDataSize, batchSize,
				      rocfftPlanDescription)) != rocfft_status_success) ||
		((err = rocfft_plan_create(&rocPlanHandleBW, rocfftPlace, rocfft_transform_type_complex_inverse, OPENCLIPER_ROCFFT_PRECISION, rocFFTnDims, fftDataSize, batchSize,
					   rocfftPlanDescription)) != rocfft_status_success)) {
	    errStr = "rocfft_plan_create: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}

	if(((err = rocfft_plan_get_work_buffer_size(rocPlanHandleFW, &rocWorkBufferBytesFW)) != rocfft_status_success) ||
		((err = rocfft_plan_get_work_buffer_size(rocPlanHandleBW, &rocWorkBufferBytesBW)) != rocfft_status_success)) {
	    errStr = "rocfft_plan_get_work_buffer_size: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}

	// See if a work buffer is required and allocate it. NOTE: if one is required but not allocated and passed to rocfft, a segfault occurs!
	if(rocWorkBufferBytesFW != 0) {
	    if((err = rocfft_execution_info_create(&rocExecInfoFW)) != rocfft_status_success) {
		errStr = "rocfft_execution_info_create: ";
		errStr += getApp()->getOpenCLErrorCodeStr(err);
		BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	    }

	    hipError_t hipErr;
	    if((hipErr = hipMalloc(&rocWorkBufferFW, rocWorkBufferBytesFW)) != hipSuccess) {
		errStr = "hipMalloc: ";
		errStr += hipGetErrorString(hipErr);
		BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	    }

	    if((err = rocfft_execution_info_set_work_buffer(rocExecInfoFW, rocWorkBufferFW, rocWorkBufferBytesFW)) != rocfft_status_success) {
		errStr = "rocfft_execution_info_set_work_buffer: ";
		errStr += getApp()->getOpenCLErrorCodeStr(err);
		BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	    }
	}
	else
	    rocExecInfoFW = nullptr;

	if(rocWorkBufferBytesBW != 0) {
	    if((err = rocfft_execution_info_create(&rocExecInfoBW)) != rocfft_status_success) {
		errStr = "rocfft_execution_info_create: ";
		errStr += getApp()->getOpenCLErrorCodeStr(err);
		BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	    }

	    hipError_t hipErr;
	    if((hipErr = hipMalloc(&rocWorkBufferBW, rocWorkBufferBytesBW)) != hipSuccess) {
		errStr = "hipMalloc: ";
		errStr += hipGetErrorString(hipErr);
		BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	    }

	    if((err = rocfft_execution_info_set_work_buffer(rocExecInfoBW, rocWorkBufferBW, rocWorkBufferBytesBW)) != rocfft_status_success) {
		errStr = "rocfft_execution_info_set_work_buffer: ";
		errStr += getApp()->getOpenCLErrorCodeStr(err);
		BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	    }
	}
	else
	    rocExecInfoBW = nullptr;

	// Apparently, rocfft generates the plan in the first invocation of rocfft_execute. Force plan creation here so that the first
	// call to launch() does not lag horribly
	void* tmp;
	void* tmp2;
	hipMalloc(&tmp, batchDistance * batchSize * sizeof(cl_float2));
	if(rocfftPlace == rocfft_placement_notinplace)
	    hipMalloc(&tmp2, batchDistance * batchSize * sizeof(cl_float2));
	else
	    tmp2 = nullptr;

	rocfft_execute(rocPlanHandleFW, &tmp, &tmp2, rocExecInfoFW);
	rocfft_execute(rocPlanHandleBW, &tmp, &tmp2, rocExecInfoBW);

	hipFree(tmp);
	if(tmp2) hipFree(tmp2);

	// rocFFT returns values unnormalized (only in the inverse transform). We need to divide them by the number of elements in each batch
	scalarMultiply = std::make_shared<ScalarMultiply>(getApp());
	scalarMultiply->setInput(getOutput());
	scalarMultiply->setOutput(getOutput());
	scalarMultiply->init();
	scalarMultiplyLP = std::make_shared<ScalarMultiply::LaunchParameters>(1.0 / batchDistance);
	scalarMultiply->setLaunchParameters(scalarMultiplyLP);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------------
    // If HIP is unavailable, compose an FFT plan using plain clFFT
    // -----------------------------------------------------------------------------------------------------------------------------------------
    else {
#endif // HAVE_ROCFFT
	clfftResultLocation clfftPlace;
	if(getInput() == getOutput())
	    clfftPlace = CLFFT_INPLACE;
	else
	    clfftPlace = CLFFT_OUTOFPLACE;

	clfftDim clFFTnDims;
	switch(FFTnDims) {
	    case 1:
		clFFTnDims = CLFFT_1D;
		break;
	    case 2:
		clFFTnDims = CLFFT_2D;
		break;
	    case 3:
		clFFTnDims = CLFFT_3D;
		break;
	    default:
		BTTHROW(CLError(CL_INVALID_WORK_DIMENSION, "Only 1, 2 and 3-dimensional FFTs are supported"), "FFT::init");
	}

	/* Setup clFFT. */
	clfftSetupData fftSetup;
	cl_int err;

	if((err = clfftInitSetupData(&fftSetup)) != CL_SUCCESS) {
	    errStr = "clfftInitSetupData: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}

	if((err = clfftSetup(&fftSetup)) != CL_SUCCESS) {
	    errStr = "clfftSetup: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}

	//Create a default plan
	if((err = clfftCreateDefaultPlan(&clPlanHandle, (getApp()->getContext())(), clFFTnDims, fftDataSize)) != CL_SUCCESS) {
	    errStr = "clfftCreateDefaultPlan: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    if(err == CLFFT_NOTIMPLEMENTED)
		errStr += ". Hint: data dimensions must be combinations of powers of 2, 3, 5, and 7";
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}

	//Set plan parameters: precision
	if((err = clfftSetPlanPrecision(clPlanHandle, OPENCLIPER_CLFFT_PRECISION)) != CL_SUCCESS) {
	    errStr = "clfftSetPlanPrecision: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}

	//Set plan parameters: data layout
	if((err = clfftSetLayout(clPlanHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED)) != CL_SUCCESS) {
	    errStr = "clfftSetLayout: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}

	//Set plan parameters: in place/out of place
	if((err = clfftSetResultLocation(clPlanHandle, clfftPlace)) != CL_SUCCESS) {
	    errStr = "clfftSetResultLocation: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}

	// Set plan parameters: in/out strides
	if((err = clfftSetPlanInStride(clPlanHandle, clFFTnDims, strides)) != CL_SUCCESS) {
	    errStr = "clfftSetPlanInStride: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}
	if((err = clfftSetPlanOutStride(clPlanHandle, clFFTnDims, strides)) != CL_SUCCESS) {
	    errStr = "clfftSetPlanOutStride: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}

	//Set plan parameters: batch size
	if((err = clfftSetPlanBatchSize(clPlanHandle, batchSize)) != CL_SUCCESS) {
	    errStr = "clfftSetPlanBatchSize: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}

	//Set plan parameters: batch distance
	if((err = clfftSetPlanDistance(clPlanHandle, batchDistance, batchDistance)) != CL_SUCCESS) {
	    errStr = "clfftSetPlanDistance: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}

	// Bake the plan
	if((err = clfftBakePlan(clPlanHandle, 1, &(getApp()->getCommandQueue(0))(), NULL, NULL)) != CL_SUCCESS) {
	    errStr = "clfftBakePlan: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}

	// Get needed work buffer size
	size_t bufferSize;
	if((err = clfftGetTmpBufSize(clPlanHandle,&bufferSize)) != CL_SUCCESS) {
	    errStr = "clfftGetTmpBufSize: ";
	    errStr += getApp()->getOpenCLErrorCodeStr(err);
	    BTTHROW(CLError(err, errStr.c_str()), "FFT::init");
	}
	FFT_CERR("FFT work buffer size="<<bufferSize<<"\n");

        // Allocate work buffer if needed size is not zero
        if(bufferSize > 0)
            clWorkBuffer = std::make_shared<XData> (getApp(), bufferSize / NDArray::getElementSize(getInput()->getElementDataType()), TYPEID_COMPLEX);
        else
            clWorkBuffer = nullptr;

#ifdef HAVE_ROCFFT
    }
#endif
}

void FFT::launch() {
    auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);
    checkCommonLaunchParameters();
    if(!pLP)
	pLP = std::unique_ptr<LaunchParameters>(new LaunchParameters());

    FFT_CERR("FFT::launch\n");

    /* Execute the plan. */

    infoItems.addInfoItem("Title", "FFT info");
    BEGIN_TIME(beginTime);
    std::vector<cl::Event> kernelsExecEventList;
    startKernelProfiling();

    // If HIP is available, use it
#ifdef HAVE_ROCFFT
    if(getApp()->getHIPDevice() != -1) {
	rocfft_status err;
	void* inputData = getInput()->getHIPDeviceBuffer();
	void* outputData = getOutput()->getHIPDeviceBuffer();

	// Wait for CL queue before using HIP
	getApp()->getCommandQueue().finish();
	if(pLP->dir == FORWARD) {
	    if((err = rocfft_execute(rocPlanHandleFW, &inputData, &outputData, rocExecInfoFW)) != rocfft_status_success) {
		errStr = "rocfft_execute (forward plan): ";
		errStr += getApp()->getOpenCLErrorCodeStr(err);
		BTTHROW(CLError(err, errStr.c_str()), "FFT::launch");
	    }
	    // Wait for HIP queue before continuing with CL
	    hipDeviceSynchronize();
	}
	else {
	    if((err = rocfft_execute(rocPlanHandleBW, &inputData, &outputData, rocExecInfoBW)) != rocfft_status_success) {
		errStr = "rocfft_execute (backwards plan): ";
		errStr += getApp()->getOpenCLErrorCodeStr(err);
		BTTHROW(CLError(err, errStr.c_str()), "FFT::launch");
	    }

	    // Wait for HIP queue before continuing with CL
	    hipDeviceSynchronize();
	    scalarMultiply->setInput(getOutput());
	    scalarMultiply->setOutput(getOutput());
	    scalarMultiply->launch();
	}
    }
    // If no HIP, use clFFT
    else {
#endif // HAVE_ROCFFT
	cl_int err;
	cl_mem inputData = (*(getInput()->getDeviceBuffer()))();
	cl_mem outputData = (*(getOutput()->getDeviceBuffer()))();

	for(unsigned batch = 0; batch < nBatches; batch++) {

	    // Set offsets for this batch
	    clfftSetPlanOffsetIn(clPlanHandle, batchOffsets[batch]);
	    clfftSetPlanOffsetOut(clPlanHandle, batchOffsets[batch]);

	    // Launch transform
            // Note: if tmpBuffer is set to nullptr, each new call to clfftEnqueueTransform allocates a new temporary buffer, which is not freed until clfftTearDown is called!
            //       Always use a preallocated tmpBuffer if clfftGetTmpBufSize returns non-zero!
	    if((err = clfftEnqueueTransform(clPlanHandle, static_cast<clfftDirection>(pLP->dir), 1, &(getApp()->getCommandQueue(0))(), 0, nullptr, nullptr, &inputData, &outputData,
                                            clWorkBuffer? (*(clWorkBuffer->getDeviceBuffer()))() : nullptr )) != CL_SUCCESS) {
		errStr = "clfftEnqueueTransform: ";
		errStr += getApp()->getOpenCLErrorCodeStr(err);
		BTTHROW(CLError(err, errStr.c_str()), "FFT::launch");
	    }
	}
#ifdef HAVE_ROCFFT
    }
#endif
    stopKernelProfiling();
    if(pProfileParameters->enable) {
	END_TIME(endTime);
	TIME_DIFF_TYPE elapsedTime;
	TIME_DIFF(elapsedTime, beginTime, endTime);
	std::ostringstream ostream;
	ostream << std::fixed << std::setprecision(PROFILINGTIMESPRECISION) << elapsedTime;
	infoItems.addInfoItem("Total (host+device) FFT time (s)", ostream.str());
	if(profilingSupported) {
	    getKernelGroupExecutionTimes(kernelsExecEventList, "OpenCLIPER::FFT::launch kernel", "OpenCLIPER::FFT::launch group of kernels");
	}
	//this->getInfoItems().saveOrPrint(LPISupport::InfoItems::HUMAN);
    }
}

FFT::~FFT() {
#ifdef HAVE_ROCFFT
    if(getApp()->getHIPDevice() != -1) {
	//Release plans
	rocfft_plan_destroy(rocPlanHandleFW);
	rocfft_plan_destroy(rocPlanHandleBW);

	if(rocExecInfoFW != nullptr) {
	    hipFree(rocWorkBufferFW);
	    rocfft_execution_info_destroy(rocExecInfoFW);
	}

	if(rocExecInfoBW != nullptr) {
	    hipFree(rocWorkBufferBW);
	    rocfft_execution_info_destroy(rocExecInfoBW);
	}

	//Release rocFFT library
	rocfft_cleanup();
    }
    else {
#endif
	//Release the plan
	clfftDestroyPlan(&clPlanHandle);

	//Release clFFT library
	clfftTeardown();
#ifdef HAVE_ROCFFT
    }
#endif
}

} /* namespace OpenCLIPER */

#undef FFT_DEBUG
