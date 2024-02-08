#include <cuda_runtime.h>
#include "commonArrayMult.hpp"
#include "vectorUtils.hpp"
#include "PerformanceTestArrayOpParallel.hpp"
typedef struct {
    int width;
    int height;
    float* elements;
} MatrixForCUDA;

// MatrixForCUDA addition - Host code
// MatrixForCUDA dimensions are assumed to be multiples of BLOCK_SIZE
void MatAdd(const MatrixForCUDA A, const MatrixForCUDA B, MatrixForCUDA C, int block_size, 
            dim3 dimGrid, dim3 dimBlock, std::shared_ptr<LPISupport::SampleCollection> pSamples, unsigned int numberOfIterations);
