#include <fstream>
#include <iostream>
#include <complex>

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << argv[0] << " <fileName> <numOfElements> {<floatValue>|<complexRealPart> <complexImaginaryPart>}" << std::endl;
    exit(-1);
  }
  std::ofstream out;
  out.open(argv[1], std::ios::out | std::ios::binary);
  if (argc==4) {// float value
    float f = atof(argv[3]);
    for (int i=0; i < atoi(argv[2]); i++) {
      out.write( reinterpret_cast<const char*>( &f ), sizeof( float ));
    }
  } else {
    float real = atof(argv[3]);
    float img = atof(argv[4]);
    std::complex<float> c(real, img);
    for (int i=0; i < atoi(argv[2]); i++) {
      out.write( reinterpret_cast<const char*>( &c ), sizeof(std::complex<float>));
    }
  }
  out.close();
  return 0;
}
