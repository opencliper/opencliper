// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
#include <stdio.h>
#include <sys/time.h>
#include <string>
#include "arrayMultCUDA.cuh"

void check_result(cudaError code, const char* message)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr, "%s (%d): %s\n", message, (int) code, 
	      cudaGetErrorString(code));
      exit(-1);
    }
}

// Forward declaration of the matrix addition kernel
__global__ void MatAddKernel(const MatrixForCUDA, const MatrixForCUDA, MatrixForCUDA);

// MatrixForCUDA add - Host code
// MatrixForCUDA dimensions are assumed to be multiples of BLOCK_SIZE
void MatAdd(const MatrixForCUDA A, const MatrixForCUDA B, MatrixForCUDA C, int block_size, 
	    dim3 dimGrid, dim3 dimBlock, std::shared_ptr<LPISupport::SampleCollection> pSamples, unsigned int numberOfIterations)
{
    // Load A and B to device memory
    MatrixForCUDA d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    check_result(cudaMalloc(&d_A.elements, size),
		 "Unable to allocate device memory");
    check_result(cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice),
		 "Unable to copy variable to device");
    MatrixForCUDA d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    check_result(cudaMalloc(&d_B.elements, size),
		 "Unable to allocate device memory");
    check_result(cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice),
		 "Unable to copy variable to device");

    // Allocate C in device memory
    MatrixForCUDA d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    check_result(cudaMalloc(&d_C.elements, size),
		 "Unable to allocate device memory");

    // Invoke kernel: Mal dimGrid si B.width y A.height no son múltiplos de 
    // block_size
    // dim3 dimBlock(block_size, block_size);
    // dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (unsigned int iteration = 0; iteration < numberOfIterations; iteration++) {
        cout << "Iteration #" << iteration << std::endl;
        //gettimeofday(&t1, NULL);
        cudaEventRecord(start, 0);
        MatAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float gpuElapsedTime;
        cudaEventElapsedTime(&gpuElapsedTime, start, stop); // in ms
        /*  ((t2.tv_sec - t1.tv_sec) * 1000.0) +
        ((t2.tv_usec - t1.tv_usec) / 1000.0); */
        pSamples->appendSample(gpuElapsedTime / 1000);
        //gettimeofday(&t2, NULL);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);  

    // Read C from device memory
    check_result(cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost),
		 "Unable to copy output variable from device");

    // Free device memory
    check_result(cudaFree(d_A.elements), "Error freeing device memory");
    check_result(cudaFree(d_B.elements), "Error freeing device memory");
    check_result(cudaFree(d_C.elements), "Error freeing device memory");
}

// MatrixForCUDA add kernel called by MatMul()
__global__ void MatAddKernel(MatrixForCUDA A, MatrixForCUDA B, MatrixForCUDA C)
{
    // Each thread computes one element of C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row > A.height-1) /* Nos salimos del rango de índices válidos */
      return; 
    if(A.height <= 32) // Son demasiados índices a mostrar con N grande
      printf("threadIdx.y: %d, blockIdx.y: %d, row: %d\n", 
	     threadIdx.y, blockIdx.y, row); 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col > B.width-1) /* Nos salimos del rango de índices válidos */
      return; 
    if(B.width <= 32) // Son demasiados índices a mostrar con N grande
      printf("threadIdx.x: %d, blockIdx.x: %d, col: %d\n", 
	     threadIdx.x, blockIdx.x, col); 

        C.elements[row * C.width + col] = A.elements[row * A.width + col] + B.elements[row * B.width + col];
}

