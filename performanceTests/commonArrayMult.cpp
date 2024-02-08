#include "commonArrayMult.hpp"
#include <stdexcept>

void readParamsIterationsSize(int argc, char* argv[], unsigned long& iterations, unsigned int& size, unsigned int& blockSize,
			      unsigned char& csvMode, std::string &usageString, std::string& outputFileName,
			      std::string& deviceName) {
    try {
	int opt;
	while((opt = getopt(argc, argv, "i:s:b:v:o:d:")) != -1) {
	    switch(opt) {
		case 'i':
		    iterations = stoul(optarg);
		    break;
		case 's':
		    size = stoul(optarg);
		    break;
		case 'b':
		    blockSize = stoul(optarg);
		    break;
		case 'v':
		    csvMode = stoul(optarg);
		    break;
		case 'o':
		    outputFileName = optarg;
		    break;
		case 'd':
		    deviceName = optarg;
		    break;
		default:
		    cerr << usageString << endl;
	    }
	}
    }
    catch(const std::invalid_argument& ex) {
	throw std::invalid_argument(std::string(__FILE__) + ": " + std::to_string(__LINE__) + ": " +
				    ex.what() + ": error en los par�metros de entrada (deben ser num�ricos salvo csvMode)");
    }
}


void initArray(float*& A, const unsigned long Rows, const unsigned long Cols,
	       const float value) {
    A = new float [Rows * Cols];
    for(unsigned long r = 0; r < Rows; r++)
	for(unsigned long c = 0; c < Cols; c++)
	    A[r * Cols + c] = value;
}

void rowMult(unsigned long row, float* A, float* B, float*& C,
	     const unsigned long RowsA,  const unsigned long ColsA,
	     const unsigned long RowsB,  const unsigned long ColsB) {
    unsigned long col, ColsC;
    ColsC = ColsB;
    for(col = 0; col < ColsC; col++) {
	C[row * ColsC + col] = 0;
	/* No protegemos la escritura de C porque ning�n otro hilo acceder�
	   a la misma fila */
	C[row * ColsC + col] = calcElemMult(A, B, ColsA, ColsB, row, col);
    }
}

void mulMatrices(float* A, float* B, float*& C,
		 const unsigned long RowsA,  const unsigned long ColsA,
		 const unsigned long RowsB,  const unsigned long ColsB) {
    unsigned long row, col, ColsC = ColsB;
    for(row = 0; row < RowsA; row++)
	for(col = 0; col < ColsB; col++)
	    C[row * ColsC + col] = calcElemMult(A, B, ColsA, ColsB, row, col);
}

void setSummaryInfo(const std::string &testName, const std::string &deviceTypeName, const std::string &deviceInfo, const std::string &sampleName,
		    LPISupport::InfoItems* pInfoItems, unsigned int dimGridOrGlobalSize, unsigned int dimBlockOrLocalSize, double dataSize,
		    double numOpsPerIteration, LPISupport::SampleCollection* pSamples, unsigned int numDigitsPrec) {
    stringstream strstr;
    strstr << setprecision(numDigitsPrec);
    pInfoItems->addInfoItem("Test name", testName);
    pInfoItems->addInfoItem("Device type", deviceTypeName);
    pInfoItems->addInfoItem("Device info", deviceInfo);
    pInfoItems->addInfoItem("Dim grid / global size", dimGridOrGlobalSize);
    pInfoItems->addInfoItem("Dim block / local size", dimBlockOrLocalSize);
    pInfoItems->addInfoItem("Data size", std::to_string(dataSize));
    pInfoItems->addInfoItem("Number of iterations", std::to_string(pSamples->getNumOfSamples()));
    pInfoItems->append(pSamples->to_infoItems(numDigitsPrec));
    pInfoItems->addInfoItem("Number of operations (per iteration)", std::to_string(numOpsPerIteration));
    double MFLOPS = numOpsPerIteration / 1e6 / pSamples->getMean();
    strstr.str(std::string()); // Emtpy string associated to string stream
    strstr << MFLOPS;
    pInfoItems->addInfoItem("Throughput (MFLOPS)", strstr.str());
}


void freeArrays(float* A, float* B, float* C) {
    free(A);
    free(B);
    free(C);
}
