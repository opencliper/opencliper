// Para control de versiones con RCS/CVS
// $Id: vector_utils.cpp,v 1.6 2016/09/28 17:21:45 manrod Exp $
const char vector_utils_cpp_RCSId[] =
    "$Id: vector_utils.cpp,v 1.6 2016/09/28 17:21:45 manrod Exp $";

#include "vectorUtils.hpp"

template <typename vector_type>
void print_vector(const string &name, std::vector<vector_type> v,
		  unsigned long show_size) {
    cout << name << ": " << endl;
    if(show_size >= v.size())
	std::copy(v.begin(), v.end(),
		  std::ostream_iterator<vector_type>(std::cout, " "));
    else {
	std::copy(v.begin(), v.begin() + show_size / 2,
		  std::ostream_iterator<vector_type>(std::cout, " "));
	std::cout << " ... ";
	std::copy(v.end() - show_size / 2, v.end(),
		  std::ostream_iterator<vector_type>(std::cout, " "));
    }
    std::cout << std::endl;
}

// Para evitar errores de enlazado porque no se encuentra la versi�n
// con el par�metro int al estar esta llamada a la funci�n en otro fichero .cpp
template void print_vector<int>(const string &name, std::vector<int> v,
				unsigned long show_size);
template void print_vector<float>(const string &name, std::vector<float> v,
				  unsigned long show_size);

template <typename array_type>
void print_array_row(array_type array[], int columns,
		     int show_size) {
    if(show_size >= columns)
	for(int c = 0; c < columns; c++)
	    std::cout << array[c] << " ";
    else {
	for(int c = 0; c < show_size / 2; c++)
	    std::cout << array[c] << " ";
	std::cout << " ... ";
	for(int c = columns - show_size / 2; c < columns; c++)
	    std::cout << array[c] << " ";
    }
    cout << endl;
}

template <typename array_type>
void print_array(const string &name, array_type array[], int rows, int columns,
		 int show_size) {
    cout << name << "(" << rows << "," << columns << "):" << endl;
    if(show_size >= rows)
	for(int r = 0; r < rows; r++) {
	    print_array_row(&array[r * columns], columns, show_size);
	}
    else {
	for(int r = 0; r < show_size / 2; r++)
	    print_array_row(&array[r * columns], columns, show_size);
	std::cout << " ... " << endl;
	for(int r = rows - show_size / 2; r < rows; r++)
	    print_array_row(&array[r * columns], columns, show_size);
    }
    std::cout << std::endl;
}

// Para evitar errores de enlazado porque no se encuentra la versi�n
// con el par�metro int al estar esta llamada a la funci�n en otro fichero .cpp
template void print_array<int>(const string &name,
			       int array[],
			       int rows,
			       int columns,
			       int show_size);

template void print_array<float>(const string &name,
				 float array[],
				 int rows,
				 int columns,
				 int show_size);

template void print_array<double>(const string &name,
				  double array[],
				  int rows,
				  int columns,
				  int show_size);



template <class elementType> Matrix<elementType>::Matrix(unsigned long paramRows,
	unsigned long paramCols) {
    // Crea el vector de elementos
    //&elements = new vector<elementType>(rows*cols);
    // Asigna los valores de n�mero de filas y de n�mero de columnas
    rows = paramRows;
    cols = paramCols;
    elements.resize(rows * cols);
}

template <class elementType> Matrix<elementType>::Matrix(
    unsigned long paramRows,
    unsigned long paramCols,
    elementType value) {
    // Crea el vector de elementos
    //Matrix(paramRows,paramCols);
    rows = paramRows;
    cols = paramCols;
    elements.resize(rows * cols);
#ifdef DEBUG
    cerr << "rows: " << rows << "\tcols: " << cols
	 << "\tsize: " << elements.size() << endl;
#endif
    for(unsigned long it = 0; it  <  elements.size(); it++) {
	elements[it] = value;
    }
#ifdef DEBUG
    cerr << "Fin asignaci�n valores " << "(" << rows << "," << cols << ")" << endl;
#endif
}

template <class elementType> Matrix<elementType>::Matrix(
    unsigned long paramRows,
    unsigned long paramCols,
    elementType* initialArray) {
    rows = paramRows;
    cols = paramCols;
    elements.resize(rows * cols);
    for(unsigned long it = 0; it  <  elements.size(); it++) {
	elements[it] = initialArray[it];
    }
}

// template <class elementType> Matrix<elementType>::Matrix(
// 							 Matrix<elementType> *initialMatrix) {
//   rows = initialMatrix.getNumberOfRows();
//   cols = initialMatrix.getNumberOfCols();
//   elements.resize (rows*cols);
//    for (unsigned long it = 0; it  <  elements.size(); it++) {
//      initialMatrix.data()[it] = initialArray[it];
//   }
// }

template <class elementType> Matrix<elementType>::~Matrix() {
    // Liberamos la memoria asociada al vector
    //delete &elements;
    elements.resize(0);
}

template <class elementType> unsigned long Matrix<elementType>::getNumberOfRows() {
    return rows;
}

template <class elementType> unsigned long Matrix<elementType>::getNumberOfCols() {
    return cols;
}

template <class elementType> elementType* Matrix<elementType>::data(void) {
    return elements.data();
}

template <typename array_type>
string arrayRowToString(array_type array[], int columns,
			int show_size) {
    string s;
    stringstream ss;
    if(show_size >= columns)
	for(int c = 0; c < columns; c++)
	    ss << array[c] << " ";
    else {
	for(int c = 0; c < show_size / 2; c++)
	    ss << array[c] << " ";
	ss << " ... ";
	for(int c = columns - show_size / 2; c < columns; c++)
	    ss << array[c] << " ";
    }
    ss << endl;
    s = ss.str();
    return s;
}



template <class elementType> string Matrix<elementType>::toString(const std::string& name, unsigned long rowShowSize,
	unsigned long colShowSize) {
    // Flujo que almacena cadenas de caracteres
    string s;
    stringstream ss;
    ss << name << "(" << rows << "," << cols << "):" << endl;
    if(rowShowSize >= rows)
	for(unsigned long r = 0; r < elements.size() / cols; r++) {
	    ss  << arrayRowToString(&elements[r * cols], cols, colShowSize);
	}
    else {
	for(unsigned long r = 0; r < rowShowSize / 2; r++)
	    ss << arrayRowToString(&elements[r * cols], cols, colShowSize);
	ss << " ... " << endl;
	for(unsigned long r = (elements.size() / cols) - (rowShowSize / 2);
		r < elements.size() / cols; r++)
	    ss << arrayRowToString(&elements[r * cols], cols, colShowSize);
    }
    ss << endl;
    s = ss.str();
    return s;
}

template <class elementType> void Matrix<elementType>::copy(
    Matrix<elementType> sourceMatrix) {

    rows = sourceMatrix.getNumberOfRows();
    cols = sourceMatrix.getNumberOfCols();
    elements.resize(rows * cols);
    for(unsigned long it = 0; it  <  elements.size(); it++) {
	elements[it] = sourceMatrix.data()[it];
    }
}

//template Matrix<float>;
//template Matrix<float>::Matrix(unsigned long rows, unsigned long cols);
template Matrix<float>::Matrix(unsigned long rows, unsigned long cols);
template Matrix<float>::Matrix(unsigned long rows, unsigned long cols, float);
//template Matrix<float>::Matrix(Matrix<float> initialMatrix);
template Matrix<float>::Matrix(unsigned long rows, unsigned long cols, float* initialArray);
template Matrix<float>::~Matrix();
template unsigned long Matrix<float>::getNumberOfRows();
template unsigned long Matrix<float>::getNumberOfCols();
template string Matrix<float>::toString(const std::string& name, unsigned long rowShowSize,
					unsigned long colShowSize);
template float* Matrix<float>::data();
template void Matrix<float>::copy(Matrix<float> sourceMatrix);
// template void print_array<double>(string name,
// 				  double array[],
// 				  int rows,
// 				  int columns,
// 				  int show_size);

