
// Extraído y adaptado de
// http://developer.amd.com/tools-and-sdks/opencl-zone/opencl-resources/introductory-tutorial-to-opencl/

#include "vectorUtils.hpp"
#include "commonArrayMult.hpp"
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
#include <utility>
#include <omp.h>
#include <LPISupport/Utils.hpp>
#include "PerformanceTestArrayOpParallel.hpp"
//#define __NO_STD_VECTOR // Use cl::vector instead of STL version

int main(int argc, char* argv[]) {

    string outputFileName = "", deviceName;
    std::shared_ptr<LPISupport::SampleCollection> pSamples = make_shared<LPISupport::SampleCollection>("execution time");

    try {
	float* A;
	float* B;
	float* C;
	unsigned long numberOfIterations;
	unsigned int size;
	unsigned int RowsA;
	unsigned int ColsA;
	unsigned int RowsB;
	unsigned int ColsB;
	unsigned int RowsC;
	unsigned int ColsC;
	PerformanceTestArrayOpParallel* pPerfTest = new PerformanceTestArrayOpParallel(argc, argv);
	auto pConfigTraits = std::dynamic_pointer_cast<PerformanceTestArrayOpParallel::ConfigTraits>(pPerfTest->getConfigTraits());
	size = pConfigTraits->size;
	numberOfIterations = pConfigTraits->repetitions;
	std::cout << "read size: " << size << std::endl;
	RowsA = size;
	ColsA = size;
	RowsB = ColsA;
	ColsB = size;
	RowsC = RowsA;
	ColsC = ColsB;

	const unsigned int SHOW_SIZE = 10;
	// Número de operaciones es ColsA productos de dos números y ColsA-1 sumas
	// de dos números por cada elemento de la matriz resultado, que tiene
	// RowsC*ColsC elementos.
	unsigned long numOpsPerIteration = RowsC * ColsC;
	pConfigTraits->numOpsPerCalc = numOpsPerIteration;

	cerr << argv[0] << " performance measurement" << std::endl;
	cout << "Starting program... " << flush;
	cout << "Creating and filling arrays ... " << flush;
	initArray(A, RowsA, ColsA, 2.0);
	initArray(B, RowsB, ColsB, 2.0);
	initArray(C, RowsC, ColsC, 0.0);
	cout << "Done." << endl;
	print_array("a", A, RowsA, ColsA, SHOW_SIZE);
	print_array("b", B, RowsB, ColsB, SHOW_SIZE);
	cout << endl;

	cerr << "Executing " << numberOfIterations << " iteration(s)\n";
	cerr << "- Matrix [1 .. " << size << ", 1 .. " << size << "]" << endl;

	TIME_DIFF_TYPE diffT2T1 = 0;
	cout << "Starting product... " << endl;
	for(unsigned int iteration = 0; iteration < numberOfIterations; iteration++) {
	    cout << "Iteration #" << iteration << std::endl;
	    BEGIN_TIME(t1);
	    unsigned int row;
#ifdef USE_OPENMP_GPU
	    cerr << "OpenMP enabled" << std::endl;
	    cerr << "OpenMP execution on GPU enabled" << std::endl;
	    //#pragma omp target map(to:A[0:RowsA*ColsA],B[0:RowsB*ColsB]) map(from:C[0:RowsC*ColsC])
	    #pragma omp target teams distribute
#endif //USE_OPENMP_GPU
#ifdef USE_OPENMP_CPU
	    cerr << "OpenMP execution on GPU disabled" << std::endl;
	    cerr << "Number of threads for CPU: " << omp_get_max_threads() << std::endl;
	    #pragma omp parallel for
#endif /* USE_OPENMP_CPU */
#ifdef USE_OPENACC_GPU
	    cerr << "OpenACC enabled" << std::endl;
	    cerr << "OpenACC execution on GPU enabled" << std::endl;
#pragma acc target teams distribute parallel for
#endif
#ifdef USE_OPENACC_CPU
	    cerr << "OpenACC execution on GPU disabled" << std::endl;
	    cerr << "Number of threads for CPU: " << omp_get_max_threads() << std::endl;
	    //#pragma acc kernels
	    //{
#pragma acc loop
//#pragma acc parallel  //#pragma acc kernels
#endif //USE_OPENACC_CPU
	    for(row = 0; row < RowsA; row ++) {
		unsigned col;

#ifdef USE_OPENMP_GPU /* OpenMP on GPU */
		#pragma omp target parallel for
#endif // USE_OPENMP_GPU
#ifdef USE_OPENMP_CPU /* OpenMP on CPU */
		#pragma omp parallel for
#endif //USE_OPENMP_GPU
#ifdef USE_OPENACC_GPU /* OpenACC on GPU */
#pragma acc target teams distribute parallel for
#endif // OpenACC on GPU
#ifdef USE_OPENACC_CPU
		//#pragma acc kernels
		//#pragma acc parallel loop gang vector //#pragma acc kernels
#pragma acc loop
#endif //USE_OPENACC_CPU
		for(col = 0; col < ColsA; col ++) {
		    //float res = 0.0;
		    C[row * ColsC + col] = A[row * ColsC + col] + B[row * ColsC + col];
		}
	    }
#ifdef USE_OPENACC_CPU
	    //}
#endif
	    END_TIME(t2);
	    TIME_DIFF(diffT2T1, t1, t2);
	    pSamples->appendSample(diffT2T1);
	}
	cout << "Product finished." << endl;
	print_array<float>("c", C, RowsC, ColsC, SHOW_SIZE);

#if defined(USE_OPENMP_GPU) || defined(USE_OPENACC_GPU)
	pConfigTraits->deviceType = "GPU";
#else
	pConfigTraits->deviceType = "CPU";
#endif
	pPerfTest->buildTestInfo(pSamples);
	pPerfTest->saveOrPrint();
	freeArrays(A, B, C);
	pSamples = nullptr;
    }
    catch(std::exception& e) {
	LPISupport::Utils::showExceptionInfo(e, argv[0]);
    }
}
