#include "arrayMultCUDA.cuh"

int main(int argc, char** argv) {
    MatrixForCUDA A, B, C;
    std::shared_ptr<LPISupport::SampleCollection> pSamples = make_shared<LPISupport::SampleCollection>("execution time");

    //printf("CUDA Runtime API reference performance measurement\n");
    try {
	unsigned long numberOfIterations;
	unsigned int size;
	unsigned int block_size;
	unsigned int SHOW_SIZE = 10, PRECISION_DIGITS = 10;
	PerformanceTestArrayOpParallel* pPerfTest = new PerformanceTestArrayOpParallel(argc, argv);
	auto pConfigTraits = std::dynamic_pointer_cast<PerformanceTestArrayOpParallel::ConfigTraits>(pPerfTest->getConfigTraits());
	size = pConfigTraits->size;
	numberOfIterations = pConfigTraits->repetitions;

	//count, N

	/*
	if (argc >= 4) {
	    if ((block_size = atoi(argv[3])) < 1) {
	    fprintf(stderr, "Invalid block size\n");
	    return -1;
	    }
	}
	*/
	block_size = pConfigTraits->dimBlockOrLocalSize;
	if(block_size > size)
	    block_size = size;

	dim3 dimBlock(block_size, block_size);
	dim3 dimGrid(ceil((float)(size) / (float)(block_size)),
		     ceil((float)(size) / (float)(block_size)));

	const unsigned int mem_size = sizeof(float) * size * size;
	A.width = A.height = size;
	B.width = B.height = size;
	C.width = C.height = size;
	A.elements = (float*)malloc(mem_size);
	B.elements = (float*)malloc(mem_size);
	C.elements = (float*)malloc(mem_size);
	cerr << argv[0] << " performance measurement" << std::endl;
	cout << "Starting program... " << flush;
	cout << "Creating and filling arrays ... " << flush;
	initArray(A.elements, A.height, A.width, 2.0);
	initArray(B.elements, B.height, B.width, 2.0);
	initArray(C.elements, C.height, C.width, 0.0);
	cout << "Done." << endl;
	print_array("a", A.elements, A.height, A.width, SHOW_SIZE);
	print_array("b", B.elements, B.height, B.width, SHOW_SIZE);
	cout << endl;

	cerr << "Executing " << numberOfIterations << " iteration(s)\n";
	cerr << "- Matrix [1 .. " << size << ", 1 .. " << size << "]" << endl;

	printf("Executing %lu iteration(s)\n", numberOfIterations);
	printf("- Matrix [1 .. %u, 1 .. %u]\n", size, size);
	printf("- Grid size %u\n", dimGrid.x);
	printf("- Block size %u\n", dimBlock.x);
	MatMul(A, B, C, block_size, dimGrid, dimBlock, pSamples, numberOfIterations);
	cout << "Product finished." << endl;

	// Number of simple calculations for array multiplication is
	// ColsA products of two float numbers plus ColsA-1 sums of two float numbers
	// for every element of output array (which has RowsC*ColsC elements)
	// IMPORTANT!!: at least one expression operand must be casted to unsigned long type for the result being unsigned long;
	// otherwise, in spite of operand values being less than integer type maximum value, the result may exceed unsigned int maximum value and be truncated
	pConfigTraits->numOpsPerCalc = ((unsigned long) C.height) * C.width * (A.width * 2 - 1);
	cerr << "Number of operations: " << ((unsigned long) C.height * C.width * (A.width * 2 - 1)) << std::endl;
	cerr << "Number of operations stored in pConfigTraits->numOpsPerCalc: " << pConfigTraits->numOpsPerCalc << std::endl;
	print_array<float>("c", C.elements, C.height, C.width, SHOW_SIZE);

	pConfigTraits->deviceType = "GPU";
	pConfigTraits->numDigitsPrec = PRECISION_DIGITS;
	pPerfTest->buildTestInfo(pSamples);
	pPerfTest->saveOrPrint();

	free(A.elements);
	free(B.elements);
	free(C.elements);
	pSamples = nullptr;
	return 0;
    }
    catch(std::exception& e) {
	LPISupport::Utils::showExceptionInfo(e, argv[0]);
    }
}
