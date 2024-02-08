// Para control de versiones con RCS/CVS
// $Id: vector_utils.hpp,v 1.5 2016/09/28 17:20:38 manrod Exp $
const char vector_utils_hpp_RCSId[] =
  "$Id: vector_utils.hpp,v 1.5 2016/09/28 17:20:38 manrod Exp $";

#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <sstream>

using namespace std;

/**
 * Funci�n para mostrar un vector de un tipo base gen�rico.
*/
template <typename vector_type>
void print_vector(const string &name, std::vector<vector_type> v,
			 unsigned long show_size);

/**
 * Funci�n para mostrar una fila de una matriz de un tipo base gen�rico.
 */
template <typename array_type>
void print_array_row (array_type array[], int columns,
		      unsigned long show_size);

/**
 * Funci�n para mostrar una matriz de un tipo base gen�rico.
 */
template <typename array_type>
void print_array (const string &name, array_type array[], int rows, int columns,
		    int show_size);

/**
 * Clase que representa a una matriz donde sus elementos se almacenan fila a 
 * fila en un vector (para que no haya que hacer conversiones al usarlo como 
 * fuente de datos para matrices opencl, que se representan como vectores
 */
template <class elementType> class Matrix {
public:
  // Constructores
  Matrix(unsigned long rows, unsigned long cols);
  Matrix(unsigned long rows, unsigned long cols, elementType value);
  //Matrix(const Matrix <elementType>&);
  Matrix(unsigned long rows, unsigned long cols, elementType *initialArray);
  // Destructor
  ~Matrix();
  // Devuelve el n�mero de filas de la matriz
  unsigned long getNumberOfRows(void);
  // Devuelve el n�mero de columnas de la matriz
  unsigned long getNumberOfCols(void);
  // Devuelve la matriz con los elementos (almacenados fila a fila como un vector)
 elementType * data(void);
  // Muestra el contenido de la matriz,  pudiendo limitar el n�mero de elementos
  // mostrado por fila y/o por clumunas
  string toString(const std::string& name, unsigned long rowShowSize,
		  unsigned long colShowSize);
  void copy(Matrix<elementType> sourceMatrix);
private:
  // Vector con los elementos de la matriz almacenados fila a fila
  vector<elementType> elements;
  // N�mero de filas y de columnas
  unsigned long rows, cols;
};

