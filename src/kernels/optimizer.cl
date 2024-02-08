/* Copyright (C) 2018 Federico Simmross Wattenberg,
 *                    Manuel Rodríguez Cayetano,
 *                    Javier Royuela del Val,
 *                    Elena Martín González,
 *                    Elisa Moya Sáez,
 *                    Marcos Martín Fernández and
 *                    Carlos Alberola López
 *
 * This file is part of OpenCLIPER.
 *
 * OpenCLIPER is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; version 3 of the License.
 *
 * OpenCLIPER is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with OpenCLIPER; If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *  Contact:
 *
 *  Federico Simmross Wattenberg
 *  E.T.S.I. Telecomunicación
 *  Universidad de Valladolid
 *  Paseo de Belén 15
 *  47011 Valladolid, Spain.
 *  fedsim@tel.uva.es
 */

/*! \file optimizer.cl
 *	\brief File containing every kernel being executed during optimization
 *  \date May 24, 2017
 *  \author Elena Martin Gonzalez
 */

#include <OpenCLIPER/kernels/hostKernelFunctions.h>

inline float mean3D(__global float2* I, int col, int row) {
  float sum = 0.0;
  
  uint width = getSpatialDimSize(I, COLUMNS, 0);
  uint height = getSpatialDimSize(I, ROWS, 0);
  uint numFrames = getTemporalDimSize(I, 0);
  
  uint idx = col + row*width;
  
  for(uint frame=0; frame<numFrames; frame++){
    sum += I[idx].x;
    idx += getTemporalDimStride(I, 0, 0);
  }
  return sum/numFrames;
}

inline float mean4D(__global float* dH, int i, int j, int l) {
    float sum = 0.0f;

    uint dim0 = getSpatialDimSize(dH, 0, 0);
    uint dim1 = getSpatialDimSize(dH, 1, 0);
    uint dim2 = getSpatialDimSize(dH, 2, 0);
    uint dim3 = getSpatialDimSize(dH, 3, 0);

    uint idx = j + i * dim1 + l * dim0 * dim1 * dim2;

    for(uint k = 0; k < dim2; k++) { // Bucle para hallar la media en frames
	sum += dH[idx];
	idx += getSpatialDimStride(dH, 2, 0);
    }
    return sum / dim2;
}

// https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
inline void atomicAdd_g_f(volatile __global float* addr, float val) {
    union {
	unsigned int u32;
	float        f32;
    } next, expected, current;
    current.f32    = *addr;
    do {
	expected.f32 = current.f32;
	next.f32     = expected.f32 + val;
	current.u32  = atomic_cmpxchg((volatile __global unsigned int*)addr, expected.u32, next.u32);
    }
    while(current.u32 != expected.u32);
}



/**
 * Metric calculation for a given image set
 * @param[out] V metric at each pixel
 * @param[in] I images for metric calculation
 * @param[in] tam2 third dimension size (number of images)
 * @param[in] mean I mean
 */
__kernel void metric(__global float *V,
		     __global float2 *I){
  
  uint height = getSpatialDimSize(I, ROWS, 0);
  uint width = getSpatialDimSize(I, COLUMNS, 0);
  uint numFrames = getTemporalDimSize(I, 0);
  
  int row = get_global_id(0); // row index
  int col = get_global_id(1); // column index
  
  uint idx1 = col + row*width;
  uint idx2 = idx1; // indice para 2D
  
  // Performs the mean on the third dimension of a 3D object
  float mean = mean3D(I,col,row);
  
  V[idx2] = 0.0;
  for(uint frame=0; frame<numFrames; frame++){
    V[idx2] += (I[idx1].x-mean)*(I[idx1].x-mean);
    idx1 += getTemporalDimStride(I, 0, 0);
  }
  V[idx2] = V[idx2]/numFrames;
}

/**
 * Addition of smooth terms to the metric calculation
 * @param[out] V metric at each pixel (smooth terms added)
 * @param[in] tam2 third dimension size (number of images)
 * @param[in] nt number of dimensions
 * @param[in] dtaux spatial derivatives
 * @param[in] dtaux2 spatial derivatives
 * @param[in] dtauxy spatial derivatives
 * @param[in] dtaut temporal derivatives
 * @param[in] dtaut2 temporal derivatives
 * @param[in] lambda0 weight associated to first spatial derivative
 * @param[in] lambda1 weight associated to second spatial derivative
 * @param[in] lambda2 weight associated to first temporal derivative
 * @param[in] lambda2 weight associated to second spatial derivative
 */
__kernel void sumLambdaMetric(__global float* V,
			      __const uint numFrames,
			      __const uint nt,
			      __global float* dtaux,
			      __global float* dtaux2,
			      __global float* dtauxy,
			      __global float* dtaut,
			      __global float* dtaut2,
			      __global int* r1,
			      __global int* r2,
			      __const float lambda0,
			      __const float lambda1,
			      __const float lambda2,
			      __const float lambda3) {


    int i = get_global_id(0);
    int j = get_global_id(1);

    uint tam0 = getSpatialDimSize(V, ROWS, 0);
    uint tam1 = getSpatialDimSize(V, COLUMNS, 0);

    int idx2 = r2[j] + r1[i] * tam1; // indice para 2D

    float sumando0 = 0.0, sumando1 = 0.0, sumando2 = 0.0, sumando3 = 0.0, sumando4 = 0.0;

    for(uint k = 0; k < numFrames; k++) { // Frames
	for(int l = 0; l < nt; l++) { // Dimensiones
	    int idx4 = r2[j] + r1[i] * tam1 + k * tam0 * tam1 + l * tam0 * tam1 * numFrames; // indice para 4D
	    sumando2 += lambda1 * 2 * (dtauxy[idx4] * dtauxy[idx4]);
	    sumando3 += lambda3 * (dtaut2[idx4] * dtaut2[idx4]);
	    sumando4 += lambda2 * (dtaut[idx4] * dtaut[idx4]);
	    for(int m = 0; m < 2; m++) {
		int idx5 = r2[j] + r1[i] * tam1 + k * tam0 * tam1 + l * tam0 * tam1 * numFrames + m * tam0 * tam1 * numFrames * nt; // indice para 5D
		sumando0 += lambda0 * (dtaux[idx5] * dtaux[idx5]);
		sumando1 += lambda1 * (dtaux2[idx5] * dtaux2[idx5]);
	    }
	}
    }
    V[idx2] += sumando0 + sumando1 + sumando2 + sumando3 + sumando4;
}


/**
 * Aproximate gradient. Returns the numerical gradient of the matrix I. I is a 2D sequence. d1 corresponds to dI/dx, the
 * differences in x (horizontal) direction. d2 corresponds to dI/dy, the differenies in y (vertical direction). The spacing
 * between points in each direction is assumed to be one.
 * @param[out] d1 gradient in horizontal dimension
 * @param[out] d2 gradient in vertical dimension
 * @param[in] I images
 */
__kernel void gradient(__global float* d1,
		       __global float* d2,
		       __global float2* I) {

    uint height = getSpatialDimSize(I, ROWS, 0);
    uint width = getSpatialDimSize(I, COLUMNS, 0);
    uint numFrames = getTemporalDimSize(I, 0);

    int row = get_global_id(0);
    int col = get_global_id(1);
    int frame = get_global_id(2);

    // Horizontal dimension
    if(col == 0) { // First column
	d1[0 + row * width + frame * width * height] = I[1 + row * width + frame * width * height].x - I[0 + row * width + frame * width * height].x;
    }
    else if(col == height - 1) { // Last column
	d1[(height - 1) + row * width + frame * width * height] = I[(height - 1) + row * width + frame * width * height].x - I[(height - 2) + row * width + frame * width * height].x;
    }
    else { // Rest of the columns
	d1[col + row * width + frame * width * height] = (I[(col + 1) + row * width + frame * width * height].x - I[(col - 1) + row * width + frame * width * height].x) / 2;
    }

    // Vertical dimension
    if(row == 0) { // First row
	d2[col + 0 * width + frame * width * height] = I[col + 1 * width + frame * width * height].x - I[col + 0 * width + frame * width * height].x;
    }
    else if(row == width - 1) { // Last row
	d2[col + (width - 1)*width + frame * width * height] = I[col + (width - 1) * width + frame * width * height].x - I[col + (width - 2) * width + frame * width * height].x;
    }
    else {  // Rest of the rows
	d2[col + row * width + frame * width * height] = (I[col + (row + 1) * width + frame * width * height].x - I[col + (row - 1) * width + frame * width * height].x) / 2;
    }
}

/**
 * Interpolate a mesh over the images to generate new images corresponding to the new mesh (bilinear interpolation)
 * @param[out] dx transformed gradient (contains horizontal dimension and vertical dimension)
 * @param[in] d1 gradient in horizontal dimension
 * @param[in] d2 gradient in vertical dimension
 * @param[in] xn new control point mesh
 * @param[in] x original mesh
 * @param[in] r1margin row indices
 * @param[in] r2margin column indices
 * @param[in] tam0 dx first dimension size
 * @param[in] tam1 dx second dimension size
 */
__kernel void gradientInterpolator(__global float* dx,
				   __global float* d1,
				   __global float* d2,
				   __global float* xn,
				   __global int* x,
				   __global int* r1margin,
				   __global int* r2margin) { // cols

    uint tam0 = getSpatialDimSize(dx, ROWS, 0);
    uint tam1 = getSpatialDimSize(dx, COLUMNS, 0);
    uint tam2 = getSpatialDimSize(dx, 2, 0);

    int i = get_global_id(0); // irow
    int j = get_global_id(1); // icol
    int frame = get_global_id(2); // frame

    int hor = x[r1margin[i] + r2margin[j] * tam0 + 0 * tam0 * tam1] - 1;
    int ver = x[r1margin[i] + r2margin[j] * tam0 + 1 * tam0 * tam1] - 1;

    float nhor = xn[r2margin[j] + r1margin[i] * tam1 + frame * tam0 * tam1 + 0 * tam0 * tam1 * tam2] - 1;
    float nver = xn[r2margin[j] + r1margin[i] * tam1 + frame * tam0 * tam1 + 1 * tam0 * tam1 * tam2] - 1;

    int prehor = floor(nhor);
    int prever = floor(nver);

    int posthor = prehor + 1;
    int postver = prever + 1;

    float i1, i2, i3, i4;
    float aux1, aux2;

    // Horizontal dimension
    i1 = d1[prehor + prever * tam1 + frame * tam0 * tam1];
    i2 = d1[posthor + prever * tam1 + frame * tam0 * tam1];
    i3 = d1[prehor + postver * tam1 + frame * tam0 * tam1];
    i4 = d1[posthor + postver * tam1 + frame * tam0 * tam1];

    aux1 = (posthor - nhor) * i1 + (nhor - prehor) * i2;
    aux2 = (posthor - nhor) * i3 + (nhor - prehor) * i4;

    dx[hor + ver * tam1 + frame * tam0 * tam1 + 0 * tam0 * tam1 * tam2] = (postver - nver) * aux1 + (nver - prever) * aux2;

    // Vertical dimension
    i1 = d2[prehor + prever * tam1 + frame * tam0 * tam1];
    i2 = d2[posthor + prever * tam1 + frame * tam0 * tam1];
    i3 = d2[prehor + postver * tam1 + frame * tam0 * tam1];
    i4 = d2[posthor + postver * tam1 + frame * tam0 * tam1];

    aux1 = (posthor - nhor) * i1 + (nhor - prehor) * i2;
    aux2 = (posthor - nhor) * i3 + (nhor - prehor) * i4;

    dx[hor + ver * tam1 + frame * tam0 * tam1 + 1 * tam0 * tam1 * tam2] = (postver - nver) * aux1 + (nver - prever) * aux2;
}


/**
 * Calculates the gradient of the metric over the images
 * @param[out] dy gradient of the metric
 * @param[in] I images
 * @param[in] mean I mean
 */
__kernel void gradientMetric(__global float* dy,
			     __global float2* I,
			     __const float factor) { // factor = 2/numFrames

    uint height = getSpatialDimSize(I, ROWS, 0);
    uint width = getSpatialDimSize(I, COLUMNS, 0);
    uint numFrames = getTemporalDimSize(I, 0);

    int row = get_global_id(0);
    int col = get_global_id(1);
    int frame = get_global_id(2);

    int idx = col + row * width + frame * width * height;

    float mean = mean3D(I, col, row);
    dy[idx] = (I[idx].x - mean) * factor;
}



__kernel void gradientRegularization(__global float* dthetax,
				     __global float* dthetax2,
				     __global float* dthetaxy,
				     __global float* dthetat,
				     __global float* dthetat2,
				     __global float* dtaux,
				     __global float* dtaux2,
				     __global float* dtauxy,
				     __global float* dtaut,
				     __global float* dtaut2,
				     __global realType* coefg,
				     __global float* BBg,
				     __global float* BB1g,
				     __global float* BB2g,
				     __global float* BB11g,
				     __const uint dim0, // rows
				     __const uint dim2,
				     __const uint dim4,
				     __const uint dimdtau0,
				     __const uint dimdtau1) {

    uint dim1 = get_global_size(0) / dim0; // cols
    uint dim3 = get_global_size(1) / dim2;
    uint dim5 = get_global_size(2) / dim4;
    uint dimdtau2 = dim2;

    int i = get_global_id(0) % dim0; // irow
    int j = get_global_id(0) / dim0; // icol
    int n = get_global_id(1) % dim2;
    int k = get_global_id(1) / dim2;
    int l = get_global_id(2) % dim4;
    int o = get_global_id(2) / dim4;

    int range1 = (int)(coefg[k + 0 * dim3 + 0 * dim3 * 2] - 1);
    int range2 = (int)(coefg[l + 0 * dim3 + 1 * dim3 * 2] - 1);

    int idx, idxy, idxdtaux, idxdtauxy;

    for(uint m = 0; m < 2; m++) {
	idx = j + i * dim1 + n * dim0 * dim1 + k * dim0 * dim1 * dim2 + l * dim0 * dim1 * dim2 * dim3 + o * dim0 * dim1 * dim2 * dim3 * dim4 + m * dim0 * dim1 * dim2 * dim3 * dim4 * dim5;
	idxdtaux = (range2 + i) + (range1 + j) * dimdtau0 + n * dimdtau0 * dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2 + m * dimdtau0 * dimdtau1 * dimdtau2 * 2;
	dthetax[idx] = dtaux[idxdtaux] * BB1g[idx];
	dthetax2[idx] = dtaux2[idxdtaux] * BB2g[idx];
    }

    idxy = j + i * dim1 + n * dim0 * dim1 + k * dim0 * dim1 * dim2 + l * dim0 * dim1 * dim2 * dim3 + o * dim0 * dim1 * dim2 * dim3 * dim4;
    idxdtauxy = (range2 + i) + (range1 + j) * dimdtau0 + n * dimdtau0 * dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2;
    dthetaxy[idxy] = dtauxy[idxdtauxy] * BB11g[idxy];

    idx = j + i * dim1 + n * dim0 * dim1 + k * dim0 * dim1 * dim2 + l * dim0 * dim1 * dim2 * dim3 + o * dim0 * dim1 * dim2 * dim3 * dim4;

    if(n == dim2 - 2) {
	dthetat[idx] = 2 * (dtaut[range1 + range2 * dimdtau1 + n * dimdtau0 * dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2] - dtaut[range1 + range2 * dimdtau1 +
			    (n + 1) * dimdtau0 * dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2]) * BBg[j + i * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim3];
	dthetat2[idx] = 2 * (dtaut2[range1 + range2 * dimdtau1 + n * dimdtau0 * dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2] - 2 * dtaut2[range1 + range2 * dimdtau1 +
			     (n + 1) * dimdtau0 * dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2] + dtaut2[range1 + range2 * dimdtau1 + 0 * dimdtau0 * dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2]) * BBg[j + i
				     * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim3];
    }
    else if(n == dim2 - 1) {
	dthetat[idx] = 2 * (dtaut[range1 + range2 * dimdtau1 + n * dimdtau0 * dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2] - dtaut[range1 + range2 * dimdtau1 + 0 * dimdtau0 * dimdtau1 +
			    o * dimdtau0 * dimdtau1 * dimdtau2]) * BBg[i + j * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim3];
	dthetat2[idx] = 2 * (dtaut2[range1 + range2 * dimdtau1 + n * dimdtau0 * dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2] - 2 * dtaut2[range1 + range2 * dimdtau1 + 0 * dimdtau0 *
			     dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2] + dtaut2[range1 + range2 * dimdtau1 + 1 * dimdtau0 * dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2]) * BBg[j + i * dim1 + k * dim0 *
				     dim1 + l * dim0 * dim1 * dim3];
    }
    else {
	dthetat[idx] = 2 * (dtaut[range1 + range2 * dimdtau1 + n * dimdtau0 * dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2] - dtaut[range1 + range2 * dimdtau1 +
			    (n + 1) * dimdtau0 * dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2]) * BBg[i + j * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim3];
	dthetat2[idx] = 2 * (dtaut2[range1 + range2 * dimdtau1 + n * dimdtau0 * dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2] - 2 * dtaut2[range1 + range2 * dimdtau1 +
			     (n + 1) * dimdtau0 * dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2] + dtaut2[range1 + range2 * dimdtau1 + (n + 2) * dimdtau0 * dimdtau1 + o * dimdtau0 * dimdtau1 * dimdtau2]) *
			BBg[j + i * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim3];
    }
}



/**
 * Precomputing the regularization term
 * @param[out] Aux auxiliar matrix
 * @param[in] dthetax spatial derivative
 * @param[in] dthetax2 spatial derivative
 * @param[in] dthetaxy spatial derivative
 * @param[in] dthetat temporal derivative
 * @param[in] dthetat2 temporal derivative
 * @param[in] lambda0 weight associated to first spatial derivative
 * @param[in] lambda1 weight associated to second spatial derivative
 * @param[in] lambda2 weight associated to first temporal derivative
 * @param[in] lambda2 weight associated to second spatial derivative
 * @param[in] dim0 Aux first dimension size
 * @param[in] dim3 Aux fourth dimension size
 */
__kernel void gradientJointAux(__global float* Aux,
			       __global float* aux2) {

    uint dim0 = getSpatialDimSize(Aux, 0, 0);
    uint dim1 = getSpatialDimSize(Aux, 1, 0);
    uint dim2 = getSpatialDimSize(Aux, 2, 0);
    uint dim3 = getSpatialDimSize(Aux, 3, 0);

    int i = get_global_id(0) % dim0; // irow
    int j = get_global_id(0) / dim0; // icol
    int k = get_global_id(1);
    int l = get_global_id(2) % dim3;
    int m = get_global_id(2) / dim3;

    int idx4 = j + i * dim1 + l * dim0 * dim1 + m * dim0 * dim1 * dim3;
    int idx5 = j + i * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim2 + m * dim0 * dim1 * dim2 * dim3;
    Aux[idx5] = aux2[idx4];
}


/**
 * Assembles each part of the gradient calculation: image gradient, metric gradient and transformation gradient with smoothness
 * @param[out] dV gradient obtained using gradient descent method
 * @param[in] dy gradient of the metric
 * @param[in] dx gradient of the images
 * @param[in] dtheta gradient of the transformation
 * @param[in] Aux auxiliar matrix containing the regularization
 * @param[in] coefg coefficients for the gradient
 * @param[in] dim0 dV first dimension size
 * @param[in] dim2 dV third dimension size
 * @param[in] dim4 dV fifth dimension size
 * @param[in] tam0 dx and dy first dimension size
 * @param[in] tam1 dx and dy second dimension size
 */
__kernel void gradientJoint(__global float* dV,
			    __global float* dy,
			    __global float* dx,
			    __global float* BBg,
			    __global float* Aux,
			    __global realType* coefg,
			    __const uint dim0,
			    __const uint dim2,
			    __const uint dim4,
			    __const uint tam0,
			    __const uint tam1) {

    uint dim1 = get_global_size(0) / dim0;
    uint dim3 = get_global_size(1) / dim2;
    uint dim5 = get_global_size(2) / dim4;
    uint tam2 = dim2;

    int i = get_global_id(0) % dim0;
    int j = get_global_id(0) / dim0;
    int n = get_global_id(1) % dim2;
    int k = get_global_id(1) / dim2;
    int l = get_global_id(2) % dim4;
    int m = get_global_id(2) / dim4;

    int range1 = (int)(coefg[k + 0 * dim3 + 0 * dim3 * 2] - 1) + j;
    int range2 = (int)(coefg[l + 0 * dim3 + 1 * dim3 * 2] - 1) + i;
    int idxdy = range1 + range2 * tam1 + n * tam0 * tam1;

    int idxdV = j + i * dim1 + n * dim0 * dim1 + k * dim0 * dim1 * dim2 + l * dim0 * dim1 * dim2 * dim3 + m * dim0 * dim1 * dim2 * dim3 * dim4;
    int idxdx = range1 + range2 * tam1 + n * tam0 * tam1 + m * tam0 * tam1 * tam2;
    int idxAux = j + i * dim1 + n * dim0 * dim1 + k * dim0 * dim1 * dim2 + l * dim0 * dim1 * dim2 * dim3;

    // gradientTransformation
    float dtheta = BBg[j + i * dim1 + l * dim0 * dim1 + m * dim0 * dim1 * dim3];
    dV[idxdV] = dy[idxdy] * dx[idxdx] * dtheta + Aux[idxAux];
}


/**
 * Permutes dVpermute = permute(dV,[1 2 4 5 3 6])
 * @param[out] dVpermute permutation of dV
 * @param[in] dV gradient obtained using gradient descent method
 * @param[in] dim0 dV first dimension size (dVpermute first dimension size)
 * @param[in] dim2 dV third dimension size (dVpermute fifth dimension size)
 * @param[in] dim4 dV fifth dimension size (dVpermute fourth dimension size)
 */
__kernel void permutedV(__global float* dVpermute,
			__global float* dV,
			__const uint dim0,
			__const uint dim2,
			__const uint dim4) {

    uint dim1 = get_global_size(0) / dim0; // dV second dimension size (dVpermute second dimension size)
    uint dim3 = get_global_size(1) / dim2; // dV fourth dimension size (dVpermute third dimension size)
    uint dim5 = get_global_size(2) / dim4; // dV sixth dimension size (dVpermute sixth dimension size)

    int i = get_global_id(0) % dim0;
    int j = get_global_id(0) / dim0;
    int k = get_global_id(1) % dim2;
    int l = get_global_id(1) / dim2;
    int m = get_global_id(2) % dim4;
    int n = get_global_id(2) / dim4;

    int idxIn = j + i * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim2 + m * dim0 * dim1 * dim2 * dim3 + n * dim0 * dim1 * dim2 * dim3 * dim4;
    int idxOut = j + i * dim1 + l * dim0 * dim1 + m * dim0 * dim1 * dim3 + k * dim0 * dim1 * dim3 * dim4 + n * dim0 * dim1 * dim3 * dim4 * dim2;

    dVpermute[idxOut] = dV[idxIn];
}


/**
 * Computes the product Wn*proydH
 * @param[out] Dif weighted projection
 * @param[in] proydH projection
 * @param[in] Wn normalized weight
 * @param[in] dim2 proydH third dimension size
 */
__kernel void updateTransformation(__global float* T,
				   __global float* Dif,
				   __global float* dH,
				   __global float* Wn) {


    uint dim0 = getSpatialDimSize(dH, 0, 0);
    uint dim1 = getSpatialDimSize(dH, 1, 0);
    uint dim2 = getSpatialDimSize(dH, 2, 0);
    uint dim3 = getSpatialDimSize(dH, 3, 0);

    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2) % dim2;
    int l = get_global_id(2) / dim2;

    // Index
    int idx = j + i * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim2;

    // Performs the mean on the third dimension of a 4D object
    //float mean = mean4D(dH, i, j, l, dim0, dim1, dim2, dim3);
    float mean = mean4D(dH, i, j, l);

    // Projection of gradient cost
    float projdH = dH[idx] - mean;

    Dif[idx] = (*Wn) * projdH; // Need Dif outside the kernel

    // New T
    T[idx] = T[idx] - Dif[idx]; // Need T outside the kernel
}


/**
 * Produces the new transformation matrix. Difference between previous T matrix and Diff
 * @param[out] TAux auxiliar variable of size BB with the values of T. Matrix with new values reshaped to fit
 * @param[in] T transformation matrix
 * @param[in] coef coefficients for transformation
 * @param[in] r1 column indices of T
 * @param[in] r2 row indices of T
 * @param[in] dim0 TAux first dimension size
 * @param[in] dim2 TAux third dimension size
 * @param[in] dim4 TAux fifth dimension size
 * @param[in] C1 T first dimension size
 * @param[in] C0 T second dimension size
 */
__kernel void transformationAux(__global float* TAux,
				__global float* T,
				__global realType* coef,
				__global int* r1,
				__global int* r2,
				__const uint dim0, // rows
				__const uint dim2,
				__const uint dim4,
				__const uint tam0,
				__const uint C1,
				__const uint C0) {

    uint dim1 = get_global_size(0) / dim0; // cols
    uint dim3 = get_global_size(1) / dim2;
    uint dim5 = get_global_size(2) / dim4;
    uint tam2 = dim2;

    int i = get_global_id(0) % dim0; // row
    int j = get_global_id(0) / dim0; // col
    int m = get_global_id(1) % dim2;
    int n = get_global_id(1) / dim2;
    int l = get_global_id(2) % dim4;
    int k = get_global_id(2) / dim4;

    int idxTAux = j + i * dim1 + m * dim0 * dim1 + n * dim0 * dim1 * dim2 + l * dim0 * dim1 * dim2 * dim3 + k * dim0 * dim1 * dim2 * dim3 * dim4;
    int idxT = (l + (int)coef[r1[j] + 0 * tam0 + 0 * tam0 * 2] - 1) + (k + (int)coef[r2[i] + 0 * tam0 + 1 * tam0 * 2] - 1) * C0 + m * C1 * C0 + n * C1 * C0 * tam2;
    TAux[idxTAux] = T[idxT];
}


/**
 * Generates a new control point mesh from the original according to their displacements
 * @param[out] xn new mesh generated from transformation
 * @param[in] TAux matrix with new values reshaped to fit
 * @param[in] BB bspline matrix
 * @param[in] r1 column indices of T
 * @param[in] r2 row indices of T
 * @param[in] dim2 TAux third dimension size
 * @param[in] tam0 xn first dimension size
 * @param[in] tam1 xn second dimension size
 * @param[in] nt number of dimensions
 * @param[in] tsEplus1 bspline order plus one
 */
__kernel void transformation(__global float* xn,
			     __global float* TAux,
			     __global float* BB,
			     __global int* r1,
			     __global int* r2,
			     __const uint dimTAux2,
			     __const uint dimBB0,
			     __const uint dimBB1,
			     __const uint nt,
			     __const uint tsEplus1) {

    uint dimTAux0 = get_global_size(0);
    uint dimTAux1 = get_global_size(1);
    uint dimTAux3 = get_global_size(2) / dimTAux2;
    uint tam2 = dimTAux2;

    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2) % dimTAux2;
    int l = get_global_id(2) / dimTAux2;

    int idx = r1[j] + r2[i] * dimBB1 + k * dimBB0 * dimBB1 + l * dimBB0 * dimBB1 * tam2;

    for(uint m = 0; m < tsEplus1; m++) {
	for(uint n = 0; n < tsEplus1; n++) {
	    int idxTAux = j + i * dimTAux1 + k * dimTAux0 * dimTAux1 + l * dimTAux0 * dimTAux1 * dimTAux2 + m * dimTAux0 * dimTAux1 * dimTAux2 * dimTAux3 + n * dimTAux0 * dimTAux1 * dimTAux2 *
			  dimTAux3 * tsEplus1;
	    int idxBB = r1[j] + r2[i] * dimBB1 + k * dimBB0 * dimBB1 + l * dimBB0 * dimBB1 * tam2 + m * dimBB0 * dimBB1 * tam2 * nt + n * dimBB0 * dimBB1 * tam2 * nt * tsEplus1;
	    xn[idx] += TAux[idxTAux] * BB[idxBB];
	}
    }
}


/**
 * Generates a new control point mesh from the original according to their displacements
 * @param[out] xn new mesh generated from transformation adjusted (xn=permute(repmat(x,[1 1 1 N]),[1 2 4 3])+xn)
 * @param[in] x initial mesh
 * @param[in] tam2 xn third dimension size
 */
__kernel void transformationAdjust(__global float* xn,
				   __global int* x) {

    uint tam0 = getSpatialDimSize(xn, ROWS, 0);
    uint tam1 = getSpatialDimSize(xn, COLUMNS, 0);
    uint tam2 = getSpatialDimSize(xn, 2, 0);

    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2) % tam2;
    int l = get_global_id(2) / tam2;

    int idx = j + i * tam1 + k * tam0 * tam1 + l * tam0 * tam1 * tam2;
    xn[idx] += x[i + j * tam0 + l * tam0 * tam1];
}


/**
 * Permutes TAux=permute(TAux,[2 1 3 4 5 6 7 8]);
 * @param[out] TAuxPerm TAux permuted
 * @param[in] TAux initial TAux
 * @param[in] dim0 TAux second dimension size (TAuxPerm first dimension size)
 * @param[in] dim2 TAux third dimension size (TAuxPerm third dimension size)
 * @param[in] dim4 TAux fifth dimension size (TAuxPerm fifth dimension size)
 */
__kernel void permuteTAux(__global float* TAuxPerm,
			  __global float* TAux,
			  __const uint dim0,
			  __const uint dim2,
			  __const uint dim4) {

    uint dim1 = get_global_size(0) / dim0; // TAux first dimension size (TAuxPerm second dimension size)
    uint dim3 = get_global_size(1) / dim2; // TAux fourth dimension size (TAuxPerm fourth dimension size)
    uint dim5 = get_global_size(2) / dim4; // TAux sixth dimension size (TAuxPerm sixth dimension size)

    int i = get_global_id(0) % dim0;
    int j = get_global_id(0) / dim0;
    int k = get_global_id(1) % dim2;
    int l = get_global_id(1) / dim2;
    int m = get_global_id(2) % dim4;
    int n = get_global_id(2) / dim4;

    int idx = i + j * dim0 + k * dim1 * dim0 + l * dim1 * dim0 * dim2 + m * dim1 * dim0 * dim2 * dim3 + n * dim1 * dim0 * dim2 * dim3 * dim4;
    int idxPerm = j + i * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim2 + m * dim0 * dim1 * dim2 * dim3 + n * dim0 * dim1 * dim2 * dim3 * dim4;

    TAuxPerm[idxPerm] = TAux[idx];
}


/**
 * Circularly shifts TAuxPerm along frames (third dimension)
 * @param[out] TAuxShift1 TAuxPerm 1 frame circularly shifted
 * @param[out] TAuxShift2 TAuxPerm 2 frames circularly shifted
 * @param[in] TAuxPerm matrix to shift
 * @param[in] dim0 TAuxPerm first dimension size
 * @param[in] dim2 TAuxPerm third dimension size
 * @param[in] dim4 TAuxPerm fifth dimension size
 */
__kernel void shiftTAux(__global float* TAuxShift1,
			__global float* TAuxShift2,
			__global float* TAuxPerm,
			__const uint dim0,
			__const uint dim2,
			__const uint dim4) {

    uint dim1 = get_global_size(0) / dim0;
    uint dim3 = get_global_size(1) / dim2;
    uint dim5 = get_global_size(2) / dim4;

    int i = get_global_id(0) % dim0;
    int j = get_global_id(0) / dim0;
    int k = get_global_id(1) % dim2;
    int l = get_global_id(1) / dim2;
    int m = get_global_id(2) % dim4;
    int n = get_global_id(2) / dim4;

    int idx = j + i * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim2 + m * dim0 * dim1 * dim2 * dim3 + n * dim0 * dim1 * dim2 * dim3 * dim4;
    int idxShift1, idxShift2;

    if(k == 0) {
	idxShift1 = j + i * dim1 + (dim2 - 1) * dim0 * dim1 + l * dim0 * dim1 * dim2 + m * dim0 * dim1 * dim2 * dim3 + n * dim0 * dim1 * dim2 * dim3 * dim4;
	idxShift2 = j + i * dim1 + (dim2 - 2) * dim0 * dim1 + l * dim0 * dim1 * dim2 + m * dim0 * dim1 * dim2 * dim3 + n * dim0 * dim1 * dim2 * dim3 * dim4;
    }
    else if(k == 1) {
	idxShift1 = j + i * dim1 + (k - 1) * dim0 * dim1 + l * dim0 * dim1 * dim2 + m * dim0 * dim1 * dim2 * dim3 + n * dim0 * dim1 * dim2 * dim3 * dim4;
	idxShift2 = j + i * dim1 + (dim2 - 1) * dim0 * dim1 + l * dim0 * dim1 * dim2 + m * dim0 * dim1 * dim2 * dim3 + n * dim0 * dim1 * dim2 * dim3 * dim4;
    }
    else {
	idxShift1 = j + i * dim1 + (k - 1) * dim0 * dim1 + l * dim0 * dim1 * dim2 + m * dim0 * dim1 * dim2 * dim3 + n * dim0 * dim1 * dim2 * dim3 * dim4;
	idxShift2 = j + i * dim1 + (k - 2) * dim0 * dim1 + l * dim0 * dim1 * dim2 + m * dim0 * dim1 * dim2 * dim3 + n * dim0 * dim1 * dim2 * dim3 * dim4;
    }

    TAuxShift1[idx] = TAuxPerm[idxShift1];
    TAuxShift2[idx] = TAuxPerm[idxShift2];
}



__kernel void regularization(__global float* dtaux,
			     __global float* dtaux2,
			     __global float* dtauxy,
			     __global float* dtaut,
			     __global float* dtaut2,
			     __global float* TAuxShift1,
			     __global float* TAuxShift2,
			     __global float* TAuxPerm,
			     __global float* BB,
			     __global float* BB1,
			     __global float* BB2,
			     __global float* BB11,
			     __global int* r1,
			     __global int* r2,
			     __const uint dimdtau0,
			     __const uint dimdtau1,
			     __const uint dimdtau2,
			     __const uint dimTAux4) {

    uint dimTAux0 = get_global_size(0);
    uint dimTAux1 = get_global_size(1);
    uint dimdtau3 = get_global_size(2) / dimdtau2;

    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2) % dimdtau2;
    int l = get_global_id(2) / dimdtau2;
    int idx, idxBB, idxTAux;

    // Spatial derivative
    for(uint p = 0; p < 2; p++) {
	idx = r2[j] + r1[i] * dimdtau1 + k * dimdtau0 * dimdtau1 + l * dimdtau0 * dimdtau1 * dimdtau2 + p * dimdtau0 * dimdtau1 * dimdtau2 * dimdtau3;
	for(uint m = 0; m < dimTAux4; m++) {
	    for(uint n = 0; n < dimTAux4; n++) {
		idxTAux = j + i * dimTAux1 + k * dimTAux0 * dimTAux1 + l * dimTAux0 * dimTAux1 * dimdtau2 + m * dimTAux0 * dimTAux1 * dimdtau2 * dimdtau3 + n * dimTAux0 * dimTAux1 * dimdtau2 *
			  dimdtau3 * dimTAux4;
		idxBB = r2[j] + r1[i] * dimdtau1 + k * dimdtau0 * dimdtau1 + l * dimdtau0 * dimdtau1 * dimdtau2 + m * dimdtau0 * dimdtau1 * dimdtau2 * dimdtau3 + n * dimdtau0 * dimdtau1 * dimdtau2
			* dimdtau3 * dimTAux4 + p * dimdtau0 * dimdtau1 * dimdtau2 * dimdtau3 * dimTAux4 * dimTAux4;
		dtaux[idx] += TAuxPerm[idxTAux] * BB1[idxBB];
		dtaux2[idx] += TAuxPerm[idxTAux] * BB2[idxBB];
	    }
	}
	if(p == l) {
	    idx = r2[j] + r1[i] * dimdtau1 + k * dimdtau0 * dimdtau1 + p * dimdtau0 * dimdtau1 * dimdtau2 + p * dimdtau0 * dimdtau1 * dimdtau2 * dimdtau3;
	    dtaux[idx] = dtaux[idx] + 1;
	}
    }

    idx = r2[j] + r1[i] * dimdtau1 + k * dimdtau0 * dimdtau1 + l * dimdtau0 * dimdtau1 * dimdtau2;
    for(uint m = 0; m < dimTAux4; m++) {
	for(uint n = 0; n < dimTAux4; n++) {
	    idxTAux = j + i * dimTAux1 + k * dimTAux0 * dimTAux1 + l * dimTAux0 * dimTAux1 * dimdtau2 + m * dimTAux0 * dimTAux1 * dimdtau2 * dimdtau3 + n * dimTAux0 * dimTAux1 * dimdtau2 *
		      dimdtau3 * dimTAux4;
	    idxBB = r2[j] + r1[i] * dimdtau1 + k * dimdtau0 * dimdtau1 + l * dimdtau0 * dimdtau1 * dimdtau2 + m * dimdtau0 * dimdtau1 * dimdtau2 * dimdtau3 + n * dimdtau0 * dimdtau1 * dimdtau2
		    * dimdtau3 * dimTAux4;
	    dtauxy[idx] += TAuxPerm[idxTAux] * BB11[idxBB];
	}
    }

    idx = r2[j] + r1[i] * dimdtau1 + k * dimdtau0 * dimdtau1 + l * dimdtau0 * dimdtau1 * dimdtau2;
    // Temporal derivative
    for(uint m = 0; m < dimTAux4; m++) {
	for(uint n = 0; n < dimTAux4; n++) {
	    int idxBB = r2[j] + r1[i] * dimdtau1 + k * dimdtau0 * dimdtau1 + l * dimdtau0 * dimdtau1 * dimdtau2 + m * dimdtau0 * dimdtau1 * dimdtau2 * dimdtau3 + n * dimdtau0 * dimdtau1 *
			dimdtau2 * dimdtau3 * dimTAux4;
	    int idxTAux = j + i * dimTAux1 + k * dimTAux0 * dimTAux1 + l * dimTAux0 * dimTAux1 * dimdtau2 + m * dimTAux0 * dimTAux1 * dimdtau2 * dimdtau3 + n * dimTAux0 * dimTAux1 * dimdtau2 *
			  dimdtau3 * dimTAux4;
	    dtaut[idx] += (TAuxPerm[idxTAux] - TAuxShift1[idxTAux]) * BB[idxBB];
	    dtaut2[idx] += (TAuxPerm[idxTAux] - 2.0 * TAuxShift1[idxTAux] + TAuxShift2[idxTAux]) * BB[idxBB];
	}
    }
}


/**
 * Interpolate a mesh over the images to generate new images corresponding to the new mesh (bilinear interpolation).
 * Performs the motion compensation.
 * @param[out] output transformed images accordign to xn
 * @param[in] input original images
 * @param[in] xn new control point mesh (deformation fields)
 * @param[in] x original mesh
 * @param[in] r1margin row indices
 * @param[in] r2margin column indices
 * @param[in] tam0 output first dimension size
 * @param[in] tam1 output second dimension size
 */
__kernel void interpolator(__global float2* output,
			   __global float2* input,
			   __global float* xn,
			   __global int* x,
			   __global int* r1margin,
			   __global int* r2margin) {

    uint tam0 = getSpatialDimSize(input, ROWS, 0);
    uint tam1 = getSpatialDimSize(input, COLUMNS, 0);
    uint tam2 = getTemporalDimSize(input, 0);

    uint i = get_global_id(0); // irow
    uint j = get_global_id(1); // icol
    uint k = get_global_id(2); // frame

    int hor = x[r1margin[i] + r2margin[j] * tam0 + 0 * tam0 * tam1] - 1; // Original mesh - horizontal dimension
    int ver = x[r1margin[i] + r2margin[j] * tam0 + 1 * tam0 * tam1] - 1; // Original mesh - vertical dimension

    float nhor = xn[r2margin[j] + r1margin[i] * tam1 + k * tam0 * tam1 + 0 * tam0 * tam1 * tam2] - 1; // New mesh - horizontal dimension
    float nver = xn[r2margin[j] + r1margin[i] * tam1 + k * tam0 * tam1 + 1 * tam0 * tam1 * tam2] - 1; // New mesh - vertical dimension

    int prehor = floor(nhor);
    int prever = floor(nver);
    int posthor = prehor + 1;
    int postver = prever + 1;

    float2 i1 = input[prehor + prever * tam1 + k * tam0 * tam1]; // Pixel left-up
    float2 i2 = input[posthor + prever * tam1 + k * tam0 * tam1]; // Pixel right-up
    float2 i3 = input[prehor + postver * tam1 + k * tam0 * tam1]; // Pixel left-down
    float2 i4 = input[posthor + postver * tam1 + k * tam0 * tam1]; // Pixel rigth-down

    float2 aux1 = (posthor - nhor) * i1 + (nhor - prehor) * i2;
    float2 aux2 = (posthor - nhor) * i3 + (nhor - prehor) * i4;

    output[hor + ver * tam1 + k * tam0 * tam1] = (postver - nver) * aux1 + (nver - prever) * aux2;
}

/**
 * Implementation of the adjoint for the deformation (adjoint of motion compensation)
 * @param[out] output transformed images accordign to xn
 * @param[in] input images
 * @param[in] xn new control point mesh
 * @param[in] x original mesh
 * @param[in] r1margin row indices
 * @param[in] r2margin column indices
 * @param[in] tam0 output first dimension size
 * @param[in] tam1 output second dimension size
 */
__kernel void adjointInterpolator(__global float2* output,
				  __global float2* input,
				  __global float* xn,
				  __global int* x,
				  __global int* r1margin,
				  __global int* r2margin) {

    uint tam0 = getSpatialDimSize(input, ROWS, 0);
    uint tam1 = getSpatialDimSize(input, COLUMNS, 0);
    uint tam2 = getTemporalDimSize(input, 0);

    uint i = get_global_id(0); // irow
    uint j = get_global_id(1); // icol
    uint k = get_global_id(2); // frame

    int hor = x[r1margin[i] + r2margin[j] * tam0 + 0 * tam0 * tam1] - 1; // Original mesh - horizontal dimension
    int ver = x[r1margin[i] + r2margin[j] * tam0 + 1 * tam0 * tam1] - 1; // Original mesh - vertical dimension

    float nhor = xn[r2margin[j] + r1margin[i] * tam1 + k * tam0 * tam1 + 0 * tam0 * tam1 * tam2] - 1; // New mesh - horizontal dimension
    float nver = xn[r2margin[j] + r1margin[i] * tam1 + k * tam0 * tam1 + 1 * tam0 * tam1 * tam2] - 1; // New mesh - vertical dimension

    int prehor = floor(nhor);
    int prever = floor(nver);
    int posthor = prehor + 1;
    int postver = prever + 1;

    float Wvv = nver - prever;
    float Whh = nhor - prehor;

    float2 i0 = input[hor + ver * tam1 + k * tam0 * tam1];

    global float* ptr = (global float*)output;

    float aux1 = i0.x * (1 - Whh) * (1 - Wvv); // Weight to pixel left-up
    float aux2 = i0.x * Whh * (1 - Wvv); 	// Weight to pixel right-up
    float aux3 = i0.x * (1 - Whh) * Wvv;	// Weight to pixel left-down
    float aux4 = i0.x * Whh * Wvv; 	// Weight to pixel right-down

    // Atomic operations in real part of images
    atomicAdd_g_f(&ptr[2 * (prehor + prever * tam1 + k * tam0 * tam1)], aux1); // Pixel left-up
    atomicAdd_g_f(&ptr[2 * (posthor + prever * tam1 + k * tam0 * tam1)], aux2); // Pixel right-up
    atomicAdd_g_f(&ptr[2 * (prehor + postver * tam1 + k * tam0 * tam1)], aux3); // Pixel left-down
    atomicAdd_g_f(&ptr[2 * (posthor + postver * tam1 + k * tam0 * tam1)], aux4); // Pixel rigth-down

    aux1 = i0.y * (1 - Whh) * (1 - Wvv); // Weight to pixel left-up
    aux2 = i0.y * Whh * (1 - Wvv); 	// Weight to pixel right-up
    aux3 = i0.y * (1 - Whh) * Wvv;	// Weight to pixel left-down
    aux4 = i0.y * Whh * Wvv; 	// Weight to pixel right-down

    // Atomic operations in imaginary part of images
    atomicAdd_g_f(&ptr[2 * (prehor + prever * tam1 + k * tam0 * tam1) + 1], aux1); // Pixel left-up
    atomicAdd_g_f(&ptr[2 * (posthor + prever * tam1 + k * tam0 * tam1) + 1], aux2); // Pixel right-up
    atomicAdd_g_f(&ptr[2 * (prehor + postver * tam1 + k * tam0 * tam1) + 1], aux3); // Pixel left-down
    atomicAdd_g_f(&ptr[2 * (posthor + postver * tam1 + k * tam0 * tam1) + 1], aux4); // Pixel rigth-down
}



__kernel void auxiliarMask(__global float* Xaux,
			   __global uint* X,
			   __global realType* coefg,
			   __const int l,
			   __const int k,
			   __const uint dim0,
			   __const uint dimcoefg0,
			   __const uint dimcoefg1) {

    uint dim1 = get_global_size(0) / dim0;
    uint dim2 = get_global_size(1);
    uint dim5 = get_global_size(2);

    int i = get_global_id(0) % dim0;
    int j = get_global_id(0) / dim0;
    int m = get_global_id(1);
    int n = get_global_id(2);
    
    uint dimX0 = getSpatialDimSize(X,0,0);


    int idxX = (i + (int)coefg[k + 0 * dimcoefg0 + 1 * dimcoefg0 * dimcoefg1] - 1) + (j + (int)coefg[l + 0 * dimcoefg0 + 0 * dimcoefg0 * dimcoefg1] - 1) * dimX0;
    int idxAux = j + i * dim0 + m * dim0 * dim1 + n * dim0 * dim1 * dim2;
    Xaux[idxAux] = (float)X[idxX];
}



__kernel void auxiliarCost(__global float* cost1,
			   __global float* dVpermute,
			   __global float* Xaux,
			   __const int l,
			   __const int k,
			   __const uint dim0, // rows
			   __const uint dim3,
			   __const uint dim4) {

    uint dim1 = get_global_size(0) / dim0; // cols
    uint dim2 = get_global_size(1);
    uint dim5 = get_global_size(2);

    int i = get_global_id(0) % dim0; // irow
    int j = get_global_id(0) / dim0; // icol
    int m = get_global_id(1);
    int n = get_global_id(2);

    int idxAux = j + i * dim1 + m * dim0 * dim1 + n * dim0 * dim1 * dim2;
    int idxdVperm = j + i * dim1 + l * dim0 * dim1 + k * dim0 * dim1 * dim3 + m * dim0 * dim1 * dim3 * dim4 + n * dim0 * dim1 * dim3 * dim4 * dim2;
    int idxcost1 = j + i * dim1 + m * dim0 * dim1 + n * dim0 * dim1 * dim2;
    cost1[idxcost1] = dVpermute[idxdVperm] * Xaux[idxAux];
}


/**
 * Sum of the metric values only in the ROI for each image and point of the mesh
 * @param[out] costTotal total cost for each frame with 4 dimensiones
 * @param[in] dV gradient obtained using gradient descent method
 * @param[in] X mask determining the ROI
 * @param[in] coefg coefficients for the gradient
 * @param[in] dim2 costTotal third dimension size
 * @param[in] dim4 dV first dimension size
 * @param[in] dim5 dV second dimension size
 * @param[in] tam0 X first dimension size
 */
__kernel void cost6D(__global float* dH,
		     __global float* cost2,
		     __const int l,
		     __const int k) {

    uint dim0 = getSpatialDimSize(dH, 0, 0);
    uint dim1 = getSpatialDimSize(dH, 1, 0);
    uint dim2 = getSpatialDimSize(dH, 2, 0);
    uint dim3 = getSpatialDimSize(dH, 3, 0);

    uint dim00 = getSpatialDimSize(cost2, 0, 0);
    uint dim11 = getSpatialDimSize(cost2, 1, 0);

    int m = get_global_id(0);
    int n = get_global_id(1);

    int idx = l + k * dim1 + m * dim0 * dim1 + n * dim0 * dim1 * dim2;
    float suma = 0.0;
    int idxcost2;
    for(int i = 0; i < dim00; i++) {
	idxcost2 = m + i * dim11 + n * dim00 * dim11;
	suma += cost2[idxcost2];
    }
    dH[idx] = suma;
}
