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
/*
 * NestaUp.cpp
 *
 *  Created on: 20 de nov. de 2017
 *      Author: Elisa Moya-Saez
 * 
 *  Modified on: 29 de oct. de 2021
 *      Author: Emilio López-Ales
 */

#include <OpenCLIPER/processes/nesta/NestaUp.hpp>
#include <complex.h>

// Uncomment to show class-specific debug messages
#define NESTAUP_DEBUG

#if !defined NDEBUG && defined NESTAUP_DEBUG
    #define NESTAUP_CERR(x) CERR(x)
#else
    #define NESTAUP_CERR(x)
    #undef NESTAUP_DEBUG
#endif

namespace OpenCLIPER {

NestaUp::NestaUp(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<ProfileParameters>& pPP): Process(pCLapp, pPP) {
	// Create subprocess objects
	pFFTOutOfPlace = Process::create<FFT>(pCLapp);
	pDataAndSensitivityMapsProduct = Process::create<ComplexElementProd>(pCLapp);
	pXImagesAllCoilSameFrameAddition = Process::create<XImageSum>(pCLapp);
	pDataAndSamplingMasksProduct = Process::create<ApplyMask>(pCLapp); 
	pTemporalTV = Process::create<TemporalTV>(pCLapp, pProfileParameters);
	pTemporalTVt = Process::create<TemporalTV>(pCLapp, pProfileParameters);
	pVectorNormalization = Process::create<VectorNormalization>(pCLapp, pProfileParameters);
	pMotionCompensation = Process::create<MotionCompensation>(pCLapp, pProfileParameters);
	pAdjointMotionCompensation = Process::create<AdjointMotionCompensation>(pCLapp, pProfileParameters);
	pCopy = Process::create<CopyDataGPU>(pCLapp, pProfileParameters);
	pComplexAbsPow2 = Process::create<ComplexAbsPow2>(pCLapp);
	pComplexAbs = Process::create<ComplexAbs>(pCLapp);

}

NestaUp::NestaUp(const std::shared_ptr<CLapp>& pCLapp, const std::shared_ptr<Data> pIn, const std::shared_ptr<Data> pOut, const std::shared_ptr<ProfileParameters>& pPP): NestaUp(pCLapp, pPP) {
	// Set input/output as given
	setInput(pIn);
	setOutput(pOut);
}

void NestaUp::init() {
	// Create auxiliary data objects (depends on getInput(), so can't do this in the constructor)
	pAuxFFT = std::make_shared<KData>(getApp(), std::dynamic_pointer_cast<KData>(getInput()), false, false);

	// Initialize subprocesses
	pFFTOutOfPlace->setInput(getInput());
	pFFTOutOfPlace->setOutput(pAuxFFT);
	pFFTOutOfPlace->init();

	pDataAndSensitivityMapsProduct->init();
	pXImagesAllCoilSameFrameAddition->init();
	pDataAndSamplingMasksProduct->init();
	pVectorNormalization->init();
	pCopy->init();
	pMotionCompensation->init();
	pAdjointMotionCompensation->init();

	pTemporalTV->setInitParameters(std::make_shared<TemporalTV::InitParameters>(TemporalTV::FORWARD));
	pTemporalTV->init();

	pTemporalTVt->setInitParameters(std::make_shared<TemporalTV::InitParameters>(TemporalTV::ADJOINT));
	pTemporalTVt->init();
    
	pComplexAbsPow2->init();
	pComplexAbs->init();
}


/**
 * Adjoint Encoding operator.
 * @param[in] inputDataHandle KData [Nx*Ny*numFrames*numCoils]
 * @param[in] sensitivityMapsDataHandle Sensitivity maps
 * @param[out] outputDataHandle XData [Nx*Ny*numFrames]
 * @param[in] auxDataHandleFFT Axiliar DataHandle to the output of inverse FFT
 * @param[in] pProfileParameters->enable
 */
void NestaUp::operatorAt(std::shared_ptr<Data> inputData, std::shared_ptr<SensitivityMapsData> sensitivityMapsData,
			 std::shared_ptr<Data> outputData, std::shared_ptr<Data> auxDataFFT) {

	try {
		// IFFT out of place (backward FFT)
		pFFTOutOfPlace->setInput(inputData);
		pFFTOutOfPlace->setOutput(auxDataFFT);
		pFFTOutOfPlace->setLaunchParameters(std::make_shared<FFT::LaunchParameters>(FFT::BACKWARD));
		pFFTOutOfPlace->launch();

		// XImage*Conj(SensMap) in place
		pDataAndSensitivityMapsProduct->setInput(auxDataFFT);
		pDataAndSensitivityMapsProduct->setOutput(auxDataFFT);
		pDataAndSensitivityMapsProduct->setLaunchParameters(std::make_shared<ComplexElementProd::LaunchParameters>(ComplexElementProd::conjugate,
		sensitivityMapsData));
		pDataAndSensitivityMapsProduct->launch();

		if((std::dynamic_pointer_cast<KData>(inputData))->getNCoils() != 1) {
			pXImagesAllCoilSameFrameAddition->setInput(auxDataFFT);
			pXImagesAllCoilSameFrameAddition->setOutput(outputData);
			pXImagesAllCoilSameFrameAddition->launch();
		}
		else {
			pCopy->setInput(auxDataFFT);
			pCopy->setOutput(outputData);
			pCopy->launch();
		}
	}
	catch(cl::Error& err) {
		BTTHROW(CLError(err), "NestaUp::operatorAt()");
	}
	catch(std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
}

/**
 * Encoding operator.
 * @param[in] inputDataHandle XData [Nx*Ny*numFrames]
 * @param[in] sensitivityMapsDataHandle Sensitivity maps
 * @param[in] samplingMasksDataHandle Sampling mask
 * @param[out] outputDataHandle KData [Nx*Ny*numFrames*numCoils]
 *
 * @param[in] pProfileParameters->enable
 */
void NestaUp::operatorA(std::shared_ptr<Data> inputData, std::shared_ptr<SensitivityMapsData> sensitivityMapsData,
			std::shared_ptr<SamplingMasksData> samplingMasksData, std::shared_ptr<Data> outputData,
			std::shared_ptr<Data> auxDataFFT) {

	try {
		// XImage*SensMap Out of place
		pDataAndSensitivityMapsProduct->setInput(inputData);
		pDataAndSensitivityMapsProduct->setOutput(auxDataFFT);
		pDataAndSensitivityMapsProduct->setLaunchParameters(std::make_shared<ComplexElementProd::LaunchParameters>(ComplexElementProd::notConjugate, sensitivityMapsData));
		pDataAndSensitivityMapsProduct->launch();

		// FFT out of place (forward FFT)
		pFFTOutOfPlace->setInput(auxDataFFT);
		pFFTOutOfPlace->setOutput(outputData);
		pFFTOutOfPlace->setLaunchParameters(std::make_shared<FFT::LaunchParameters>(FFT::FORWARD));
		pFFTOutOfPlace->launch();

		pDataAndSamplingMasksProduct->setInput(outputData);
		pDataAndSamplingMasksProduct->setOutput(outputData);
		pDataAndSamplingMasksProduct->setLaunchParameters(std::make_shared<ApplyMask::LaunchParameters>(samplingMasksData));
		pDataAndSamplingMasksProduct->launch();
	}
	catch(cl::Error& err) {
		BTTHROW(CLError(err), "NestaUp::operatorA()");
	}
	catch(std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
}


/**
 * Sparsity operator. Motion Conpensation operator (if the recontruction is with motion compensation) and Temporal Total Variation (tTV) operator.
 * @param[in] inputDataHandle XData [Nx*Ny*numFrames]
 * @param[out] outputDataHandle XData [Nx*Ny*numFrames]
 * @param[in] auxiliarDataHandleMC Auxiliar XData [Nx*Ny*numFrames] for MotionCompensation
 * @param[in] argsMC  Structure with arguments obtained in Motion Estimation, needed in Motion Compensation
 * @param[in] pProfileParameters->enable
 */
void NestaUp::operatorU(std::shared_ptr<Data> inputData, std::shared_ptr<Data> outputData,
			std::shared_ptr<Data> auxiliarDataMC, ArgumentsMotionCompensation* argsMC) {

	// In the case we perform Nesta reconstruction w/ motion compensation
	if(argsMC != nullptr) {
		try {
			pMotionCompensation->setInput(inputData);
			pMotionCompensation->setOutput(auxiliarDataMC);
			pMotionCompensation->setLaunchParameters(std::make_shared<MotionCompensation::LaunchParameters>(argsMC));
			pMotionCompensation->launch();
		}
		catch(cl::Error& err) {
			BTTHROW(CLError(err), "NestaUp::operatorU()");
		}
		catch(std::exception& e) {
			std::cout << "Error: " << e.what() << std::endl;
		}
	}
	else {
		auxiliarDataMC = inputData;
	}
	try {
		pTemporalTV->setInput(auxiliarDataMC);
		pTemporalTV->setOutput(outputData);
		pTemporalTV->launch();
	}
	catch(cl::Error& err) {
		BTTHROW(CLError(err), "NestaUp::operatorU()");
	}
	catch(std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
}


/**
 * Adjoint Sparsity operator. Adjoint Motion Conpensation operator (if the recontruction is with motion compensation) and adjoint Temporal Total Variation operator.
 * @param[in] inputDataHandle XData [Nx*Ny*numFrames]
 * @param[out] outputDataHandle XData [Nx*Ny*numFrames]
 * @param[in] auxiliarDataHandleMC Auxiliar XData [Nx*Ny*numFrames] for MotionCompensation
 * @param[in] argsMC  Structure with arguments obtained in Motion Estimation, needed in Motion Compensation
 * @param[in] pProfileParameters->enable
 */
void NestaUp::operatorUt(std::shared_ptr<Data> inputData, std::shared_ptr<Data> outputData,
			 std::shared_ptr<Data> auxiliarDataMC, ArgumentsMotionCompensation* argsMC) {

	try {
		pTemporalTVt->setInput(inputData);
		pTemporalTVt->setOutput(auxiliarDataMC);
		pTemporalTVt->launch();
	}
	catch(cl::Error& err) {
		BTTHROW(CLError(err), "NestaUp::operatorUt()");
	}
	catch(std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
	// In the case we perform Nesta reconstruction w/ motion compensation
	if(argsMC != nullptr) {
		try {
			pAdjointMotionCompensation->setInput(auxiliarDataMC);
			pAdjointMotionCompensation->setOutput(outputData);
			pAdjointMotionCompensation->setLaunchParameters(std::make_shared<AdjointMotionCompensation::LaunchParameters>(argsMC));
			pAdjointMotionCompensation->launch();
		}
		catch(cl::Error& err) {
			BTTHROW(CLError(err), "NestaUp::operatorUt()");
		}
		catch(std::exception& e) {
			std::cout << "Error: " << e.what() << std::endl;
		}
	}
	else {
		pCopy->setInput(auxiliarDataMC);
		pCopy->setOutput(outputData);
		pCopy->launch();
	}
}


/**
 * Estimates the spectral norm of a operator using the power method. Used for estimate the norm of Sparsity operator.
 * @param[in] rows Rows of the data
 * @param[in] cols Cols of the data
 * @param[in] slices Slices of the data
 * @param[in] NumFrames Number of frames of the data
 * @param[in] argsMC  Structure with arguments obtained in Motion Estimation, needed in Motion Compensation
 * @param[in] pProfileParameters->enable
 * @return the norm of the operator
 */
float NestaUp::myNormest(uint rows, uint cols, uint slices, uint NumFrames,  ArgumentsMotionCompensation* argsMC) {

	float n = rows * cols * slices * NumFrames;
	float tol = 1E-3;
	unsigned int maxiter = 100;
	unsigned int cnt = 0;
	float e;
	float e0;
	cl_float2 aux;

	cl_int status;
	cl_command_queue queue;
	queue = (getApp()->getCommandQueue(0))();
	cl_event event;

	float* norm = new float[1]();
	float* nnzSx = new float[1];

	cl_mem normObj = clCreateBuffer((getApp()->getContext())(), CL_MEM_READ_WRITE, sizeof(cl_float2), NULL, NULL);
	cl_mem nnzSxObj = clCreateBuffer((getApp()->getContext())(), CL_MEM_READ_WRITE, sizeof(cl_float2), NULL, NULL);

	std::vector<NDArray*>* pObjNDArrays = new std::vector<NDArray*>();
	for(uint i = 0; i < NumFrames; i++){
		complexType* pElement = new complexType(1, 0);
		std::vector <complexType>* pData = new std::vector<complexType>(rows * cols * slices, *pElement);
		std::vector <dimIndexType>* pDims = new std::vector<dimIndexType>({ rows, cols, slices });
		NDArray* pObjNDArray = new ConcreteNDArray<complexType>(pDims, pData);
		pObjNDArrays->push_back(pObjNDArray);
	}
	std::vector<dimIndexType>* pDynDims = new std::vector<dimIndexType>({NumFrames});

	std::shared_ptr<Data> pXData = std::make_shared<XData>(getApp(), pObjNDArrays, pDynDims);
	std::shared_ptr<Data> pSxData = std::make_shared<XData>(getApp(), std::dynamic_pointer_cast<XData>(pXData), false);
	std::shared_ptr<Data> pMCAux = std::make_shared<XData>(getApp(), std::dynamic_pointer_cast<XData>(pXData), false);

	std::vector<complexType>* Sx = nullptr;
	cl_mem pXBuffer = (*(pXData->getDeviceBuffer()))();
	cl_mem pSxBuffer = (*(pSxData->getDeviceBuffer()))();

	status = CLBlastScnrm2(n, normObj, 0, pXBuffer, 0, 1, &queue, &event);
	if(status != CL_SUCCESS) {
		printf("CLBlastScnrm2() failed with %d\n", status);
	}
	else {
		// Wait for calculations to be finished.
		status = clWaitForEvents(1, &event);
		// Fetch results of calculations from GPU memory.
		status = clEnqueueReadBuffer(queue, normObj, CL_TRUE, 0, sizeof(cl_float), norm, 0, NULL, NULL);
	}

	e = norm[0];
	aux.x = 1 / e;
	aux.y = 0;

	status = CLBlastCscal(n, aux, pXBuffer, 0, 1, &queue, &event);
	if(status != CL_SUCCESS) {
		printf("CLBlastCscal() failed with %d\n", status);
	}

	e0 = 0;

	while((abs(e - e0) > (tol * e)) && (cnt < maxiter)) {
		e0 = e;

		operatorU(pXData, pSxData, pMCAux, argsMC);

		status = CLBlastScasum(n, nnzSxObj, 0, pSxBuffer, 0, 1, &queue, &event);
		if(status != CL_SUCCESS) {
			printf("CLBlastSasum() failed with %d\n", status);
			
		}
		else {
			// Wait for calculations to be finished.
			status = clWaitForEvents(1, &event);
			// Fetch results of calculations from GPU memory.
			status = clEnqueueReadBuffer(queue, nnzSxObj, CL_TRUE, 0, sizeof(cl_float), nnzSx, 0, NULL, NULL);
		}

		if(nnzSx[0] == 0) {
			pObjNDArrays = new std::vector<NDArray*>();

			for(uint i = 0; i < NumFrames; i++) {
				Sx = new std::vector<complexType>(rows * cols * slices);
				for(uint j = 0; j < rows * cols * slices; j++) {
					Sx->at(j) = complexType(drand48(), 0);
				}
				std::vector <dimIndexType>* pDims = new std::vector<dimIndexType>({ rows, cols, slices });
				NDArray* pObjNDArray = new ConcreteNDArray<complexType>(pDims, Sx);
				pObjNDArrays->push_back(pObjNDArray);
				NESTAUP_CERR("Sx inside for i loop after sentence new ConcreteNDArray<complexType>(pDims, Sx) :" << Sx << std::endl);
			}
			pDynDims = new std::vector<dimIndexType>({NumFrames});
			pSxData = std::make_shared<XData>(getApp(), pObjNDArrays, pDynDims);
			pSxBuffer = (*(pSxData->getDeviceBuffer()))();
		}

		status = CLBlastScnrm2(n, normObj, 0, pSxBuffer, 0, 1, &queue, &event);
		if(status != CL_SUCCESS) {
			printf("CLBlastScnrm2() failed with %d\n", status);
		}
		else {
			// Wait for calculations to be finished.
			status = clWaitForEvents(1, &event);
			// Fetch results of calculations from GPU memory.
			status = clEnqueueReadBuffer(queue, normObj, CL_TRUE, 0, sizeof(cl_float), norm, 0, NULL, NULL);
		}

		e = norm[0];

		operatorUt(pSxData, pXData, pMCAux, argsMC);

		pXBuffer = (*(pXData->getDeviceBuffer()))();
		status = CLBlastScnrm2(n, normObj, 0, pXBuffer, 0, 1, &queue, &event);
		if(status != CL_SUCCESS) {
			printf("CLBlastScnrm2() failed with %d\n", status);
		}
		else {
			// Wait for calculations to be finished.
			status = clWaitForEvents(1, &event);
			// Fetch results of calculations from GPU memory.
			status = clEnqueueReadBuffer(queue, normObj, CL_TRUE, 0, sizeof(cl_float), norm, 0, NULL, NULL);
		}

		aux.x = 1 / norm[0];
		aux.y = 0;

		status = CLBlastCscal(n, aux, pXBuffer, 0, 1, &queue, &event);
		if(status != CL_SUCCESS) {
			printf("CLBlastCscal() failed with %d\n", status);
		}
		cnt = cnt + 1;
	}

	pXData = nullptr;
	pSxData = nullptr;

	status = clReleaseMemObject(normObj);
	status = clReleaseMemObject(nnzSxObj);

	delete[](norm);
	delete[](nnzSx);

	delete(pObjNDArrays);
	delete(pDynDims);
	NESTAUP_CERR("Sx value: " << Sx << std::endl);
	delete(Sx);

	return e;
}


void NestaUp::launch() {
	auto pLP = std::dynamic_pointer_cast<LaunchParameters>(pLaunchParameters);

	infoItems.addInfoItem("Title", "NestaUp info");
	startProfiling();

	std::shared_ptr<SensitivityMapsData> sensitivityMapsData = std::dynamic_pointer_cast<KData>(getInput())->getSensitivityMapsData();
	std::shared_ptr<SamplingMasksData> samplingMasksData = std::dynamic_pointer_cast<KData>(getInput())->getSamplingMasksData();
	std::shared_ptr<Data> pInputKData = std::make_shared<KData>(getApp(), std::dynamic_pointer_cast<KData>(getInput()), false, true);

	uint cols = NDARRAYWIDTH(getInput()->getNDArray(0));
	uint rows = NDARRAYHEIGHT(getInput()->getNDArray(0));
	uint slices = NDARRAYDEPTH(getInput()->getNDArray(0));
	if(slices==0)
		slices = 1;
	uint numFrames = getInput()->getDynDimsTotalSize();
	uint numCoils = (std::dynamic_pointer_cast<KData>(getInput()))->getNCoils();
    
	cl_command_queue queue = (getApp()->getCommandQueue(0))();
	cl_event calcEvents[3];
	cl_event readEvents[3];
	cl_int status;
    
	//Copies complex-float elements from inputBuffer to pOriginalInputKDataBuffer.
	cl_mem inputBuffer;
	cl_mem pOriginalInputKDataBuffer;
	inputBuffer = (*(getInput()->getDeviceBuffer()))();
	pOriginalInputKDataBuffer = (*(pInputKData->getDeviceBuffer()))();
	status = CLBlastCcopy(rows * cols * slices * numFrames * numCoils, inputBuffer, 0, 1, pOriginalInputKDataBuffer, 0, 1, &queue, NULL);
	if(status != CL_SUCCESS)
	    BTTHROW(CLError(status),"NestaUp: CLBlastCcopy() failed");

	
	cl_float2 f;
	f.x = sqrt(float(cols*rows*slices));
	f.y = 0.0f;
    
	status = CLBlastCscal(rows * cols * slices * numFrames * numCoils, f, pOriginalInputKDataBuffer, 0, 1, &queue, NULL);
	if(status != CL_SUCCESS)
	    BTTHROW(CLError(status),"NestaUp: CLBlastCsscal() failed");

	// Soluci�n del adjunto para inicializar Nesta en la primera reconstrucci�n (sin MC)
	if(pLP->argsMC == nullptr) {
		operatorAt(pInputKData, sensitivityMapsData, getOutput(), pAuxFFT);
	}

	std::shared_ptr<Data> pAuxMC = std::make_shared<XData>(getApp(), std::dynamic_pointer_cast<XData>(getOutput()), false);
	std::shared_ptr<Data> pUx_RefImage = std::make_shared<XData>(getApp(), std::dynamic_pointer_cast<XData>(getOutput()), false);

	operatorU(getOutput(), pUx_RefImage, pAuxMC, pLP->argsMC);

	std::shared_ptr<Data> pWkXData = std::make_shared<XData>(getApp(), std::dynamic_pointer_cast<XData>(getOutput()), true); // Cambiado a true, mejora funcionamiento, pero revisar si es necesario
	std::shared_ptr<Data> pXkXData = std::make_shared<XData>(getApp(), std::dynamic_pointer_cast<XData>(getOutput()), false);
	std::shared_ptr<Data> pYkXData = std::make_shared<XData>(getApp(), std::dynamic_pointer_cast<XData>(getOutput()), false);
	std::shared_ptr<Data> pZkXData = std::make_shared<XData>(getApp(), std::dynamic_pointer_cast<XData>(getOutput()), false);
	std::shared_ptr<Data> pUkXData = std::make_shared<XData>(getApp(), std::dynamic_pointer_cast<XData>(getOutput()), false);
	std::shared_ptr<Data> pAuxFxXData = std::make_shared<XData>(getApp(), std::dynamic_pointer_cast<XData>(getOutput()), false);
	std::shared_ptr<Data> pDfXData = std::make_shared<XData>(getApp(), std::dynamic_pointer_cast<XData>(getOutput()), false);
	std::shared_ptr<Data> pResKData = std::make_shared<KData>(getApp(), std::dynamic_pointer_cast<KData>(getInput()), false, true);
	std::shared_ptr<Data> pAuxResKData = std::make_shared<KData>(getApp(), std::dynamic_pointer_cast<KData>(getInput()), false, true);
	std::shared_ptr<Data> pAResXData = std::make_shared<XData>(getApp(), std::dynamic_pointer_cast<XData>(getOutput()), false);
	std::shared_ptr<Data> pOutputAbsXData = std::make_shared<XData>(getApp(), std::dynamic_pointer_cast<XData>(getOutput()), false);

	cl_mem fxObj = clCreateBuffer((getApp()->getContext())(), CL_MEM_READ_WRITE, sizeof(cl_float2), NULL, NULL);
	cl_mem normResObj = clCreateBuffer((getApp()->getContext())(), CL_MEM_READ_WRITE, sizeof(cl_float2), NULL, NULL);
	cl_mem normUkObj = clCreateBuffer((getApp()->getContext())(), CL_MEM_READ_WRITE, sizeof(cl_float2), NULL, NULL);
	cl_mem l2normObj = clCreateBuffer((getApp()->getContext())(), CL_MEM_READ_WRITE, sizeof(cl_float2), NULL, NULL);

	cl_mem pXrefBuffer;
	cl_mem pXkBuffer;
	cl_mem pYkBuffer;
	cl_mem pZkBuffer;
	cl_mem pAuxFxBuffer;
	cl_mem pUkBuffer;
	cl_mem pResBuffer;
	cl_mem pDfBuffer;
	cl_mem pAResBuffer;
	cl_mem pWkBuffer;

	// Image maximum calculation (host operation)

	getOutput()->device2Host();
	pComplexAbs->setInput(getOutput());
	pComplexAbs->setOutput(pOutputAbsXData);
	pComplexAbs->launch();

	pOutputAbsXData->device2Host();
	float maxXref = LONG_MIN;
	dimIndexType width, height, depth;
	for(uint i = 0; i < pOutputAbsXData->getNumNDArrays(); i++) {
		width = NDARRAYWIDTH(pOutputAbsXData->getData()->at(i));
		height = NDARRAYHEIGHT(pOutputAbsXData->getData()->at(i));
		depth = NDARRAYDEPTH(pOutputAbsXData->getData()->at(i));
		if(depth == 0)
			depth = 1;
		complexType* pComplexArray = (complexType*) pOutputAbsXData->getHostBuffer(i);
		for(dimIndexType index1D = 0;  index1D < height * width * depth; index1D++) {
			float pixelSearch = pComplexArray[index1D].real();
			if(pixelSearch > maxXref)
				maxXref = pixelSearch;
		}
	}

	pOutputAbsXData = NULL;

	// Image maximum calculation (host operation)
	pUx_RefImage->device2Host();
	pComplexAbs->setInput(pUx_RefImage);
	pComplexAbs->setOutput(pUx_RefImage);
	pComplexAbs->launch();

	pUx_RefImage->device2Host();
	float maxUXref = LONG_MIN;
	for(uint i = 0; i < pUx_RefImage->getNumNDArrays(); i++) {
		width = NDARRAYWIDTH(pUx_RefImage->getData()->at(i));
		height = NDARRAYHEIGHT(pUx_RefImage->getData()->at(i));
		depth = NDARRAYDEPTH(pUx_RefImage->getData()->at(i));
		if(depth == 0)
			depth = 1;
		complexType* pComplexArray = (complexType*) pUx_RefImage->getHostBuffer(i);
		for(dimIndexType index1D = 0;  index1D < height * width * depth; index1D++) {
			float pixelSearch = pComplexArray[index1D].real();
			if(pixelSearch > maxUXref)
				maxUXref = pixelSearch;
		}
	}
	pUx_RefImage = NULL;

	//Initizalize variables
	uint miniter = (pLP->miniter);
	float lambda = (pLP->lambda_i) * maxXref;
	float muf = (pLP->mu_f) * maxXref;
	float mu0 = 0.9 * maxUXref;
	float normU = myNormest(rows, cols, slices, numFrames,  pLP->argsMC);
	float muL = lambda / (pLP->La);
	if(muL > mu0) {
		mu0 = muL;
	}
	float gamma = pow((muf / mu0), (1.0 / (pLP->maxIntIter)));
	float mu = mu0;
	float gammat = pow(((pLP->tolVar) / 0.1), (1.0 / (pLP->maxIntIter)));
	pLP->tolVar = 0.1;

	float* fx = new float[1];
	float* l1term = new float[1];
	float* l2term = new float[1];
	float* normRes = new float[1];
	float* normUk = new float[1];

	float apk;
	float tauk;

	float* fmean = new float[miniter];

	float qp; //Stopping criterion

	// These are used only if pLP->showProgress == true
	auto showDims = pXkXData->getDynDims();
	unsigned nShowTotalFrames = showDims->at(0);
	unsigned nShowFrames = nShowTotalFrames;
	unsigned curFrame = 0;

	for(uint nl = 1; nl <= (pLP->maxIntIter); nl++) {
		float Ak;
		float Lmu;
		float Lmu1;
		mu = mu * gamma;
		pLP->tolVar = (pLP->tolVar) * gammat;
		NESTAUP_CERR("\n Beginning L1 Minimization; mu = " << mu << std::endl);
		/////////////////////////////
		////   CORE NESTEROV UP  ////
		/////////////////////////////

		f.x = 0.0f;
		f.y = 0.0f;

		//Scales a complex-float pWkBuffer by a complex-float constant (initialize to zero).
		pWkBuffer = (*(pWkXData->getDeviceBuffer()))();
		status = CLBlastCscal(rows * cols * slices * numFrames, f, pWkBuffer, 0, 1, &queue, NULL);
		if(status != CL_SUCCESS)
		    BTTHROW(CLError(status),"NestaUp: CLBlastCsscal() failed");

		//Copies complex-float elements from pXrefBuffer to pXkBuffer.
		pXrefBuffer = (*(getOutput()->getDeviceBuffer()))();
		pXkBuffer = (*(pXkXData->getDeviceBuffer()))();
		status = CLBlastCcopy(rows * cols * slices * numFrames, pXrefBuffer, 0, 1, pXkBuffer, 0, 1, &queue, NULL);
		if(status != CL_SUCCESS)
		    BTTHROW(CLError(status),"NestaUp: CLBlastCopy() failed");

		//Scales complex-float pYkBuffer by a complex-float constant (initialize to zero).
		pYkBuffer = (*(pYkXData->getDeviceBuffer()))();
		status = CLBlastCscal(rows * cols * slices * numFrames, f, pYkBuffer, 0, 1, &queue, NULL);
		if(status != CL_SUCCESS)
		    BTTHROW(CLError(status),"NestaUp: CLBlastCsscal() failed");

		//Scales a complex-float pZkBuffer by a complex-fliat constant (initialize to zero).
		pZkBuffer = (*(pZkXData->getDeviceBuffer()))();
		status = CLBlastCscal(rows * cols * slices * numFrames, f, pZkBuffer, 0, 1, &queue, NULL);
		if(status != CL_SUCCESS)
		    BTTHROW(CLError(status),"NestaUp: CLBlastCsscal() failed");

		Ak = 0.0f;
		Lmu = normU / mu;
		fmean[miniter-1] = LONG_MAX;

		Lmu = lambda * Lmu + (pLP->La);
		Lmu1 = 1 / Lmu;
		bool qpDone = false;

		for(int k = 0; k < (pLP->maxIter); k++) {
			////----START PERFORM L1 CONSTRAINT----////

			// show a progress window if requested
			if(pLP->showProgress) {
				Data::ShowParms sp;
				sp.loops=1;
				sp.title="Progress View";
				sp.firstFrame=curFrame;
				if((curFrame+=nShowFrames) >= nShowTotalFrames)
					curFrame -= nShowTotalFrames;
				sp.nFrames=nShowFrames--;
				if(nShowFrames == 0)
					nShowFrames = 1;
				pXkXData->show(&sp);
			}

			//Apply sparse operator
			operatorU(pXkXData, pUkXData, pAuxMC, pLP->argsMC);

			pAuxFxBuffer = (*(pAuxFxXData->getDeviceBuffer()))();
			pUkBuffer = (*(pUkXData->getDeviceBuffer()))();

			//Copies complex-float elements from pUkBuffer to pAuxFxBuffer.
			status = CLBlastCcopy(rows * cols * slices * numFrames, pUkBuffer, 0, 1, pAuxFxBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
			    BTTHROW(CLError(status),"NestaUp: CLBlastCcopy() failed");
			
			//process vectorNormalization to normalize pUkXData
			pVectorNormalization->setInput(pUkXData);
			pVectorNormalization->setOutput(pUkXData);
			pVectorNormalization->setLaunchParameters(std::make_shared<VectorNormalization::LaunchParameters>(mu));
			pVectorNormalization->launch();

			//computes the euclidean norm of pUkBuffer containing float-complex elements
			status = CLBlastScnrm2(rows * cols * slices * numFrames, normUkObj, 0, pUkBuffer, 0, 1, &queue, &(calcEvents[0]));
			if(status != CL_SUCCESS)
			    BTTHROW(CLError(status),"NestaUp: CLBlastScnrm2() failed");
		    
			//dot product of two vectors containing float-complex elements (pUkBuffer and pAuxFxBuffer) conjugating pUkBuffer
			status = CLBlastCdotc(rows * cols * slices * numFrames, fxObj, 0, pUkBuffer, 0, 1, pAuxFxBuffer, 0, 1, &queue, &(calcEvents[1]));
			if(status != CL_SUCCESS)
			    BTTHROW(CLError(status),"NestaUp: CLBlastCdotc() failed");

			//Apply sparse operator (adjoint)
			operatorUt(pUkXData, pDfXData, pAuxMC, pLP->argsMC);

			f.x = -1.0f;
			f.y = 0.0f;

			pDfBuffer = (*(pDfXData->getDeviceBuffer()))();
			status = CLBlastCscal(rows * cols * slices * numFrames, f, pDfBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCsscal() failed");


			////----END PERFORM L1 CONSTRAINT----////

			//Apply encoding operator
			operatorA(pXkXData, sensitivityMapsData, samplingMasksData, pResKData, pAuxFFT);

			f.x = 1/(sqrt(cols*rows*slices));
			f.y = 0.0f;

			pResBuffer = (*(pResKData->getDeviceBuffer()))();
			status = CLBlastCscal(rows * cols * slices * numFrames * numCoils, f, pResBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCsscal() failed");

			f.x = -1.0f;
			f.y = 0.0f;

			//Scale vector pInitialKImageBuffer of complex-float elements and add to pResBuffer.
			cl_mem pInputBuffer = (*(getInput()->getDeviceBuffer()))();
			status = CLBlastCaxpy(rows * cols * slices * numFrames * numCoils,  f, pInputBuffer, 0, 1, pResBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCaxpy() failed");

			//Copies complex-float elements from pResBuffer to pAuxResBuffer.
			pResBuffer = (*(pResKData->getDeviceBuffer()))();
			cl_mem pAuxResBuffer = (*(pAuxResKData->getDeviceBuffer()))();
			status = CLBlastCcopy(rows * cols * slices * numFrames * numCoils, pResBuffer, 0, 1, pAuxResBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCcopy() failed");


			//computes the euclidean norm of pResBuffer containing float-complex elements
			pComplexAbsPow2->setInput(pResKData);
			pComplexAbsPow2->setOutput(pResKData);
			pComplexAbsPow2->launch();

			//computes the absolute value of all elements in the pResBuffer
			pResBuffer = (*(pResKData->getDeviceBuffer()))();
			status = CLBlastScsum(rows * cols * slices * numFrames * numCoils, l2normObj, 0, pResBuffer, 0, 1, &queue, &(calcEvents[2]));
			if(status!=CL_SUCCESS)
		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCcsum() failed");


			f.x = sqrt(float(cols*rows*slices));
			f.y = 0.0f;
			status = CLBlastCscal(rows * cols * slices * numFrames * numCoils, f, pAuxResBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
    		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCsscal() failed");

			//Apply encoding operator (adjoint)
			operatorAt(pAuxResKData, sensitivityMapsData, pAResXData, pAuxFFT);

			//Scales a complex-float pDfBuffer by a complex-float constant.
			f.x = lambda;
			f.y = 0;
			pDfBuffer = (*(pDfXData->getDeviceBuffer()))();
			status = CLBlastCscal(rows * cols * slices * numFrames, f, pDfBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCsscal() failed");

			//Scale vector pAResBuffer of complex-float elements and add to pDfBuffer.
			f.x = 1.0f;
			f.y = 0;
			pAResBuffer = (*(pAResXData->getDeviceBuffer()))();
			status = CLBlastCaxpy(rows * cols * slices * numFrames, f, pAResBuffer, 0, 1, pDfBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCaxpy() failed");
		    
			//--- Updating yk ---//

			//Copies complex-float elements from pXkBuffer to pYkBuffer.
			status = CLBlastCcopy(rows * cols * slices * numFrames, pXkBuffer, 0, 1, pYkBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCcopy() failed");

			//Scale vector pDfBuffer of complex-float elements and add to pYkBuffer.
			f.x = -Lmu1;
			f.y = 0;
			status = CLBlastCaxpy(rows * cols * slices * numFrames, f, pDfBuffer, 0, 1, pYkBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCaxpy() failed");

			//-------------------------//
			//---Stopping criterion ---//

			clWaitForEvents(3, calcEvents);
			status = clEnqueueReadBuffer(queue, normUkObj, CL_FALSE, 0, sizeof(cl_float), normUk, 0, NULL, &(readEvents[0]));
			status = clEnqueueReadBuffer(queue, fxObj, CL_FALSE, 0, sizeof(cl_float), fx, 0, NULL, &(readEvents[1]));
			status = clEnqueueReadBuffer(queue, l2normObj, CL_FALSE, 0, sizeof(cl_float), l2term, 0, NULL, &(readEvents[2]));
			clWaitForEvents(3, readEvents);

			l1term[0] = fx[0] - (mu / 2.0f) * pow(normUk[0], 2);
			fx[0] = (0.5f) * l2term[0] + lambda * l1term[0];
			switch(pLP->stoptest) {
				case 1:
					float fmeanMean = 0.0f;
					if(k>=miniter){
						for(int nelem = 0; nelem<miniter; nelem++){
							fmeanMean+=fmean[miniter-nelem-1];
						}
						fmeanMean = (float)fmeanMean/miniter;
						qp = (fmeanMean - fx[0])/fmeanMean;
						if(qp<=pLP->tolVar){
							if(qpDone)
								k = pLP->maxIter;
							NESTAUP_CERR("Done. qp = "<< qp << '\n');
							qpDone = true;
							break;
						}
					}
					else{
						qp = fmean[miniter-1];
					}
					if(fx[0]>fmean[(miniter-1)-((k)%miniter)]) {
						NESTAUP_CERR("Warning: Function is increasing");
					}
					fmean[(miniter-1)-((k+1)%miniter)] = *fx;
					break;
			}

			//-------------------//
			//--- Updating zk ---//

			apk = 0.5f * (k + 1.0f);
			Ak = Ak + apk;
			tauk = 2.0f / (k + 3.0f);

			f.x = apk;
			f.y = 0;

			//Scale vector pDfBuffer of complex-float elements and add to pWkBuffer.
			status = CLBlastCaxpy(rows * cols * slices * numFrames, f, pDfBuffer, 0, 1, pWkBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCaxpy() failed");

			//Copies complex-float elements from pXrefBuffer to pZkBuffer.
			pXrefBuffer = (*(getOutput()->getDeviceBuffer()))();
			status = CLBlastCcopy(rows * cols * slices * numFrames, pXrefBuffer, 0, 1, pZkBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCcopy() failed");

			//Scale vector pWkBuffer of complex-float elements and add to pZkBuffer.
			f.x = -Lmu1;
			f.y = 0;
			status = CLBlastCaxpy(rows * cols * slices * numFrames, f, pWkBuffer, 0, 1, pZkBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCaxpy() failed");

			//-------------------//
			//--- Updating xk ---//

			//Copies complex-float elements from pZkBuffer to pXkBuffer.
			status = CLBlastCcopy(rows * cols * slices * numFrames, pZkBuffer, 0, 1, pXkBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCcopy() failed");

			//Scales a complex-float pXkBuffer by a complex-float constant.
			f.x = tauk;
			f.y = 0;
			status = CLBlastCscal(rows * cols * slices * numFrames, f, pXkBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCscal() failed");

			f.x = (1.0f - tauk);
			f.y = 0;

			//Scale vector pYkBuffer of complex-float elements and add to pXkBuffer.
			status = CLBlastCaxpy(rows * cols * slices * numFrames, f, pYkBuffer, 0, 1, pXkBuffer, 0, 1, &queue, NULL);
			if(status != CL_SUCCESS)
		    	    BTTHROW(CLError(status),"NestaUp: CLBlastCaxpy() failed");

			if((k + 1) % (pLP->verbose) == 0) {
				NESTAUP_CERR("Iter: " << (k + 1) << " ~ fmu: " << fx[0] << " ~ Rel. Variation of fmu: " << qp << std::endl);
			}
		}

		/////////////////////////////////
		////   END CORE NESTEROV UP  ////
		/////////////////////////////////

		NESTAUP_CERR("----------------------------------------------------------------------------" << std::endl);

		//Copies complex-float elements from pXkBuffer to pXrefBuffer.
		status = CLBlastCcopy(rows * cols * slices * numFrames, pXkBuffer, 0, 1, pXrefBuffer, 0, 1, &queue, NULL);
		if(status != CL_SUCCESS)
	    	    BTTHROW(CLError(status),"NestaUp: CLBlastCcopy() failed");
	}

	if(pLP->argsMC != nullptr) {
		pAuxMC = nullptr;
	}

	//objetos cl
	clReleaseMemObject(fxObj);
	clReleaseMemObject(normResObj);
	clReleaseMemObject(normUkObj);

	delete[](fx);
	delete[](normRes);
	delete[](normUk);
	delete[](fmean);

	stopProfiling();

}
}
#undef NESTAUP_DEBUG
