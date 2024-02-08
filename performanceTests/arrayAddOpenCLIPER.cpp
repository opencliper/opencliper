
// Extraído y adaptado de
// http://developer.amd.com/tools-and-sdks/opencl-zone/opencl-resources/introductory-tutorial-to-opencl/

#include <LPISupport/InfoItems.hpp>
#include "vectorUtils.hpp"
#include "commonArrayMult.hpp"
#include <OpenCLIPER/processes/performanceTests/ArrayAddProcess.hpp>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include <vector>
#include <array>
#include <chrono> // Para medir tiempos de ejecución
using namespace std;
using namespace std::chrono;
#include <iomanip> // Para std::setprecision
#include <OpenCLIPER/defs.hpp>
#include <OpenCLIPER/XData.hpp>
#include "PerformanceTestArrayOpenCLIPER.hpp"
#include <utility>
#include <omp.h>
//#define __NO_STD_VECTOR // Use cl::vector instead of STL version

int main(int argc, char* argv[]) {

    std::shared_ptr<LPISupport::SampleCollection> pSamples = std::make_shared<LPISupport::SampleCollection>("execution time");

    try {
	unsigned long numberOfIterations ;
	unsigned int size, blockSize = 0;
	unsigned int RowsA;
	unsigned int ColsA;
	unsigned int RowsB;
	unsigned int ColsB;
	unsigned int RowsC;
	unsigned int ColsC;
	PerformanceTestArrayOpenCLIPER* pPerfTest = new PerformanceTestArrayOpenCLIPER(argc, argv);
	auto pConfigTraits = std::dynamic_pointer_cast<PerformanceTestArrayOpenCLIPER::ConfigTraits>(pPerfTest->getConfigTraits());
	size = pConfigTraits->size;
	numberOfIterations = pConfigTraits->repetitions;
	blockSize = pConfigTraits->dimBlockOrLocalSize;
	std::cout << "read size: " << size << std::endl;
	std::cout << "read blockSize: " << blockSize << std::endl;
	RowsA = size;
	ColsA = size;
	RowsB = ColsA;
	ColsB = size;
	RowsC = RowsA;
	ColsC = ColsB;

	// Número de operaciones es ColsA productos de dos números y ColsA-1 sumas
	// de dos números por cada elemento de la matriz resultado, que tiene
	// RowsC*ColsC elementos.
	unsigned long numOpsPerIteration = RowsC * ColsC;
	pConfigTraits->numOpsPerCalc = numOpsPerIteration;

	// Step 1: get a new OpenCLIPER app and initialize computing device
	CLapp::PlatformTraits platformTraits = pConfigTraits->platformTraits;
	CLapp::DeviceTraits deviceTraits = pConfigTraits->deviceTraits;
	deviceTraits.queueProperties = cl::QueueProperties(CL_QUEUE_PROFILING_ENABLE);
	auto pCLapp = CLapp::create(platformTraits, deviceTraits);

	const unsigned int SHOW_SIZE = 10;
	const unsigned int PRECISION_DIGITS = 10;

	cerr << argv[0] << " performance measurement" << std::endl;
	cout << "Starting program... " << flush;

	// Step 2: load input data
	cout << "Creating and filling arrays ... " << flush;
	std::shared_ptr<Data> XDataA(XData::genTestXData(pCLapp, ColsA, RowsA, 1, type_index(typeid(realType)), XData::CONSTANT));
	std::shared_ptr<Data> XDataB(XData::genTestXData(pCLapp, ColsB, RowsB, 1, type_index(typeid(realType)), XData::CONSTANT));

	// Step 3: create output with same size as input
	std::vector<dimIndexType>* pArrayDims = new std::vector<dimIndexType>({ColsC, RowsC});
	std::vector<std::vector<dimIndexType>*>* pArraysDims = new std::vector<std::vector<dimIndexType>*>;
	pArraysDims->push_back(pArrayDims);
	std::vector<dimIndexType>* pDynDims = new std::vector<dimIndexType>(); // Constructor value only sets vector size not value, that is 0 by default
	pDynDims->push_back(1); // Sets value
	std::shared_ptr<Data> XDataC(new XData(pCLapp, pArraysDims, pDynDims, type_index(typeid(realType))));
	//std::shared_ptr<Data> XDataC(new XData((dynamic_pointer_cast<XData>(XDataA)), false));

	// Set 4: register input and output in our CL app
	// automatic

	cout << "Done." << endl;

	/*
	cout << XDataA->getData()->at(0)->hostDataToString("a");
	cout << XDataB->getData()->at(0)->hostDataToString("b");
	cout << endl;
	*/


        // Step 5: create new process bound to our CL app, enabling profiling,
	// and set its input/output data sets
	auto pProcess = Process::create<ArrayAddProcess>(pCLapp, std::make_shared<ProcessCore::ProfileParameters> (true, numberOfIterations));
	pProcess->setInput(XDataA);
	pProcess->setOutput(XDataC);

	// Set parameters: handle of second array to be added
	auto launchParamsArrayAddProcess = make_shared<ArrayAddProcess::LaunchParameters>(XDataB->getHandle(), RowsA, ColsA, blockSize);
	pProcess->setLaunchParameters(launchParamsArrayAddProcess);

	// Step 6: load OpenCL kernel(s)
	pCLapp->loadKernels();

	// Step 7: initialize process
	pProcess->init();

	cerr << "Executing " << numberOfIterations << " iteration(s)\n";
	cerr << "- Matrix [1 .. " << size << ", 1 .. " << size << "]" << endl;
	LPISupport::InfoItems infoItems;
	cerr << pCLapp->getHWSWInfo().to_string(pConfigTraits->outputFormat);
	//TIME_DIFF_TYPE diffT2T1 = 0;
	std::stringstream strstr;
	strstr << setprecision(PRECISION_DIGITS);
	cout << "Starting product... " << endl;
	/*
	 for (unsigned long iteration = 0; iteration < numberOfIterations; iteration++) {
	    cout << "Iteration #" << iteration << std::endl;
	    BEGIN_TIME(t1);
	*/
	// Step 7.2 launch process
	pProcess->launch();

	/*
	    END_TIME(t2);
	    TIME_DIFF(diffT2T1, t1, t2);
	    pSamples->appendSample(diffT2T1);
	}
	*/
	// Step 8: get data back from computing device
	pCLapp->device2Host(XDataC);
	cout << "Product finished." << endl;
	if(size <= SHOW_SIZE) {
	    cout << XDataC->hostBufferToString("c", 0);
	}

	LPISupport::InfoItems infoItemsProfilingGPU;
	infoItemsProfilingGPU.append(pProcess->getSamplesGPUExecTime()->to_infoItems(PRECISION_DIGITS));
	/*
	 for (unsigned int i = 0; i < infoItemsProfilingGPU.size(); i++) {
	    infoItems.push_back(infoItemsProfilingGPU.at(i));
	}
	*/
	cerr << infoItemsProfilingGPU.to_string(LPISupport::InfoItems::OutputFormat::HUMAN);

	LPISupport::InfoItems infoItemsProfilingGPUAndCPU;
	infoItemsProfilingGPUAndCPU.append(pProcess->getSamplesGPU_CPUExecTime()->to_infoItems(PRECISION_DIGITS));
	cerr << infoItemsProfilingGPUAndCPU.to_string(LPISupport::InfoItems::OutputFormat::HUMAN);

	pConfigTraits->deviceType = pCLapp->getDeviceTypeAsString();
	pConfigTraits->deviceName = pCLapp->getDeviceVendor() +  " " + pCLapp->getDeviceName();
#if USE_GPU
	pPerfTest->buildTestInfo(pProcess->getSamplesGPUExecTime());
#else
	//pPerfTest->buildTestInfo(pProcess->getSamplesGPU_CPUExecTime());
	pPerfTest->buildTestInfo(pProcess->getSamplesGPUExecTime());
#endif
	pPerfTest->saveOrPrint();
    }
    catch(cl::BuildError& e) {
	CLapp::dumpBuildError(e);
    }
    catch(CLError& e) {
	std::cerr << CLapp::getOpenCLErrorInfoStr(e, argv[0]);
    }
    catch(std::exception& e) {
	LPISupport::Utils::showExceptionInfo(e, argv[0]);
    }
}
