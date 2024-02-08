// $Id: comunMulMatrices.hpp,v 1.9 2016/09/28 19:42:10 manrod Exp $
/**
   \brief  Fichero con las firmas de funciones comunes para las distintas versiones 
   de programas que realizan el cálculo del producto de dos matrices.
 */
#include <iostream> // std::cout
#include <string> // std::string, std::stoi
#include <sstream> // string stream
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono> // Para medir tiempos de ejecución
#include <vector>
#include <iomanip> // Para std::setprecision
using namespace std;
using namespace std::chrono;
#include <unistd.h> // for getopt
#include <stdlib.h>
#include <OpenCLIPER/defs.hpp>
#include <LPISupport/SampleCollection.hpp>
#include <LPISupport/InfoItems.hpp>

using namespace OpenCLIPER;

#define INIT_MEASURE_TIME(t1) high_resolution_clock::time_point t1 = \
    high_resolution_clock::now();
#define END_MEASURE_TIME(t2) high_resolution_clock::time_point t2 = \
    high_resolution_clock::now();

/**
   Extrae de los parámetros de llamada del programa el número de iteraciones, 
   tamaño de la matriz (cuadrada) y modo csv
   @param[in] argc contador de argumentos de llamada
   @param[in] argv matriz con las cadenas de texto de los argumentos de llamada
   @param[in] iterations número de iteraciones 
   @param[in] size tamaño de la matriz (cuadrada)
   @param[in] csvMode modo csv
 */
void readParamsIterationsSize(int argc, char *argv[], unsigned long &iterations, unsigned int &size, unsigned int &blockSize, 
                              unsigned char &csvMode, std::string usageString, std::string &outputFileName, 
                              std::string &deviceName);

void initArray(float *&A, const unsigned long Rows, const unsigned long Cols,
	       const float value);

// inline float calcElemMult(float *a, float *b,
// 		   const unsigned long ColsA, const unsigned long ColsB,
// 		   const unsigned long row, const unsigned long col);
// Funciones inline deben ir siempre en un fichero de cabecera, si no,
// el enlazador no es capaz de encontrar la función desde el fichero fuente desde
// donde es llamada

inline float calcElemMult(float *a, float *b,
		   const unsigned long ColsA, const unsigned long ColsB,
		   const unsigned long row, const unsigned long col) {
  float res = 0.0;
  for (unsigned long k = 0; k < ColsA; k ++) {
    res += a[row*ColsA+k]*b[k*ColsB+col];
  }
  return res;
}

void rowMult(unsigned long row, float *A, float *B, float *&C,
	     const unsigned long RowsA,  const unsigned long ColsA,
	     const unsigned long RowsB,  const unsigned long ColsB);

/**
 * Función para obtener una cadena de texto con información sobre el programa (nombres 
 * de campos de información y sus valroes), la duración de los cálculos, el número de 
 * cálculos realizados y  el número de cálculos por unidad de tiempo (en MFLOPS).
 */
void setSummaryInfo(std::string testName, std::string deviceTypeName, std::string deviceInfo, std::string sampleName, 
                     LPISupport::InfoItems *pInfoItems, unsigned int dimGridOrGlobalSize, unsigned int dimBlockOrLocalSize, double dataSize, 
                    double numOpsPerIteration, LPISupport::SampleCollection* pSamples, unsigned int numDigitsPrec);
 
void freeArrays(float *A, float *B, float *C);
