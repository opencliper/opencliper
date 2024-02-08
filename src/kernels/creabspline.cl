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
/*! \file creabspline.cl
 *	\brief File containing every kernel being executed during Bsplines creation
 *  \date May 24, 2017
 *  \author Elena Martin Gonzalez
 */

// ROW MAJOR - index = icol + irow*cols + ...

#include <OpenCLIPER/kernels/hostKernelFunctions.h>

/** Setting the location of every control point in the original mesh
 * @param[out] pu location of each control point in the mesh
 * @param[in] C00 minimum limit to guarantee influence only in ROI
 * @param[in] C10 minimum limit to guarantee influence only in ROI
 * @param[in] c center
 * @param[in] Dp density point
 */
__kernel void controlPointsLocation(__global float* pu,
				    __const int C00,
				    __const int C10,
				    __global float* c,
				    __global int* Dp) {


    int i = get_global_id(0); // irow
    int j = get_global_id(1); // icol

    uint height = getSpatialDimSize(pu, ROWS, 0);
    uint width = getSpatialDimSize(pu, COLUMNS, 0);

    pu[j + i * width + 0 * width * height] = c[0] + Dp[0] * (-C00 + j);
    pu[j + i * width + 1 * width * height] = c[1] + Dp[1] * (-C10 + i);
}


/**
 * Computes matricial products
 * @param[out] auxBB matrix of bspline products
 * @param[out] auxBB1 matrix of bspline products Bprimax*By y Bprimay*Bx
 * @param[out] auxBB2 matrix of bspline products B2primax*By y B2primay*B
 * @param[out] auxBB11 matrix of bspline products Bprimax*Bprimay
 * @param[in] BBAux auxiliar matrix for the coefficients
 * @param[in] dim2 third dimension size
 */
__kernel void createBB(__global float* auxBB,
		       __global float* auxBB1,
		       __global float* auxBB2,
		       __global float* auxBB11,
		       __global float* BBAux,
		       __const uint dim2) { // rows BBAux

    uint dim0 = get_global_size(0); // rows auxBB
    uint dim1 = get_global_size(1); // cols auxBB
    uint dim3 = get_global_size(2) / dim2; // cols BBAux

    int i = get_global_id(0); // irow auxBB
    int j = get_global_id(1); // icol auxBB
    int m = get_global_id(2) % dim2; // irow BBAux
    int n = get_global_id(2) / dim2; // irow BBAux
    
    int idx = j + i * dim1 + m * dim1 * dim0 + n * dim1 * dim0 * dim2;

    auxBB[idx] = BBAux[i + n * dim0 + 0 * dim0 * dim2 + 1 * dim0 * dim2 * 3] * BBAux[j + m * dim0 + 0 * dim0 * dim2 + 0 * dim0 * dim2 * 3];

    auxBB1[idx + 0 * dim1 * dim0 * dim2 * dim3] = BBAux[i + n * dim0 + 0 * dim0 * dim2 + 1 * dim0 * dim2 * 3] * BBAux[j + m * dim0 + 1 * dim0 * dim2 + 0 * dim0 * dim2 * 3];
    
    auxBB1[idx + 1 * dim1 * dim0 * dim2 * dim3] = BBAux[i + n * dim0 + 1 * dim0 * dim2 + 1 * dim0 * dim2 * 3] * BBAux[j + m * dim0 + 0 * dim0 * dim2 + 0 * dim0 * dim2 * 3];

    auxBB2[idx + 0 * dim1 * dim0 * dim2 * dim3] = BBAux[i + n * dim0 + 0 * dim0 * dim2 + 1 * dim0 * dim2 * 3] * BBAux[j + m * dim0 + 2 * dim0 * dim2 + 0 * dim0 * dim2 * 3];
    
    auxBB2[idx + 1 * dim1 * dim0 * dim2 * dim3] = BBAux[i + n * dim0 + 2 * dim0 * dim2 + 1 * dim0 * dim2 * 3] * BBAux[j + m * dim0 + 0 * dim0 * dim2 + 0 * dim0 * dim2 * 3];

    auxBB11[idx] = BBAux[i + n * dim0 + 1 * dim0 * dim2 + 1 * dim0 * dim2 * 3] * BBAux[j + m * dim0 + 1 * dim0 * dim2 + 0 * dim0 * dim2 * 3];
}


/**
 * Permutes bspline products matrices and performs the product with density point
 * @param[out] BB matrix of bspline products after permutation
 * @param[out] BB1 matrix of bspline products Bprimax*By y Bprimay*Bx after permutation
 * @param[out] BB2 matrix of bspline products B2primax*By y B2primay*B after permutation
 * @param[out] BB11 matrix of bspline products Bprimax*Bprimay after permutation
 * @param[out] BB11 matrix of bspline products Bprimax*Bprimay after permutation
 * @param[in] auxBB matrix of bspline products before permutation
 * @param[in] auxBB1 matrix of bspline products Bprimax*By y Bprimay*Bx before permutation
 * @param[in] auxBB2 matrix of bspline products B2primax*By y B2primay*B before permutation
 * @param[in] auxBB11 matrix of bspline products Bprimax*Bprimay before permutation
 * @param[in] Dp density point
 * @param[in] dim0 BB first dimension size (auxBB third dimension size)
 * @param[in] dim2 BB third dimension size
 * @param[in] dim4 BB fifth dimension size (auxBB first dimension size)
 */
__kernel void permuteBB(__global float* BB,
			__global float* BB1,
			__global float* BB2,
			__global float* BB11,
			__global float* auxBB,
			__global float* auxBB1,
			__global float* auxBB2,
			__global float* auxBB11,
			__global int* Dp,
			__const uint dim0, // rows
			__const uint dim2,
			__const uint dim4) {

    uint dim1 = get_global_size(0) / dim0; // cols
    uint dim3 = get_global_size(1) / dim2;
    uint dim5 = get_global_size(2) / dim4;

    int i = get_global_id(0) % dim0; // irow
    int j = get_global_id(0) / dim0; // icol
    int k = get_global_id(1) % dim2;
    int m = get_global_id(1) / dim2;
    int n = get_global_id(2) % dim4;
    int o = get_global_id(2) / dim4;

    int idxpermute, idx;

    for(int l = 0; l < 2; l++) {
	idxpermute = n + o * dim5 + j * dim5 * dim4 + i * dim5 * dim4 * dim0 + l * dim5 * dim4 * dim0 * dim1;
	idx = j + i * dim1 + k * dim1 * dim0 + m * dim1 * dim0 * dim2 + n * dim1 * dim0 * dim2 * dim3 + o * dim1 * dim0 * dim2 * dim3 * dim4 +  l * dim1 * dim0 * dim2 * dim3 * dim4 * dim5;
	BB1[idx] = Dp[l] * auxBB1[idxpermute];
	BB2[idx] = Dp[l] * Dp[l] * auxBB2[idxpermute];
    }

    idxpermute = n + o * dim5 + j * dim5 * dim4 + i * dim5 * dim4 * dim0;
    idx = j + i * dim1 + k * dim1 * dim0 + m * dim1 * dim0 * dim2 + n * dim1 * dim0 * dim2 * dim3 + o * dim1 * dim0 * dim2 * dim3 * dim4;
    BB11[idx] = Dp[0] * Dp[1] * auxBB11[idxpermute];
    BB[idx] = auxBB[idxpermute];
}

/**
 * Computes matricial products
 * @param[out] BBg matrix of bspline products
 * @param[out] auxBB1g matrix of bspline products Bprimax*By y Bprimay*Bx
 * @param[out] auxBB2g matrix of bspline products B2primax*By y B2primay*B
 * @param[out] auxBB11g matrix of bspline products Bprimax*Bprimay
 * @param[in] BBAux1 auxiliar matrix for the coefficients
 * @param[in] Dp density points
 * @param[in] dim2 third dimension size
 */
__kernel void createBBg(__global float* BBg,
			__global float* auxBB1g,
			__global float* auxBB2g,
			__global float* auxBB11g,
			__global float* BBAux1,
			__global int* Dp,
			__const uint dim2) {

    uint dim0 = get_global_size(0); // rows
    uint dim1 = get_global_size(1); // cols
    uint dim3 = get_global_size(2) / dim2;

    int i = get_global_id(0); // irow
    int j = get_global_id(1); // icol
    int n = get_global_id(2) % dim2;
    int m = get_global_id(2) / dim2;
    
    int idx = j + i * dim1 + n * dim1 * dim0 + m * dim1 * dim0 * dim2;

    BBg[idx] = BBAux1[i + m * dim1 + 0 * dim1 * dim2 + 1 * dim1 * dim2 * 3] * BBAux1[j + n * dim1 + 0 * dim1 * dim2 + 0 * dim1 * dim2 * 3];

    auxBB11g[idx] = 2 * Dp[0] * Dp[1] * BBAux1[i + m * dim1 + 1 * dim1 * dim2 + 1 * dim1 * dim2 * 3] * BBAux1[j + n * dim1 + 1 * dim1 * dim2 + 0 * dim1 * dim2 * 3];

    auxBB1g[idx + 0 * dim1 * dim0 * dim2 * dim3] = 2 * Dp[0] * BBAux1[i + m * dim1 + 0 * dim2 * dim1 + 1 * dim2 * dim1 * 3] * BBAux1[j + n * dim1 + 1 * dim2 * dim1 + 0 * dim2 * dim1 * 3];
    auxBB1g[idx + 1 * dim1 * dim0 * dim2 * dim3] = 2 * Dp[1] * BBAux1[i + m * dim1 + 1 * dim2 * dim1 + 1 * dim2 * dim1 * 3] * BBAux1[j + n * dim1 + 0 * dim2 * dim1 + 0 * dim2 * dim1 * 3];

    auxBB2g[idx + 0 * dim1 * dim0 * dim2 * dim3] = 2 * Dp[0] * Dp[0] * BBAux1[i + m * dim1 + 0 * dim2 * dim1 + 1 * dim2 * dim1 * 3] * BBAux1[j + n * dim1 + 2 * dim2 * dim1 + 0 * dim2 * dim1 * 3];
    auxBB2g[idx + 1 * dim1 * dim0 * dim2 * dim3] = 2 * Dp[1] * Dp[1] * BBAux1[i + m * dim1 + 2 * dim2 * dim1 + 1 * dim2 * dim1 * 3] * BBAux1[j + n * dim1 + 0 * dim2 * dim1 + 0 * dim2 * dim1 * 3];
}

/**
 * Repeats dimensions
 * @param[out] aux1BB1g
 * @param[out] aux1BB2g
 * @param[out] aux1BB11g
 * @param[in] auxBB1g
 * @param[in] auxBB2g
 * @param[in] auxBB11g
 * @param[in] dim0
 * @param[in] dim2
 * @param[in] dim4
 * @param[in] dim6
 */
__kernel void repmatBBg(__global float* aux1BB1g,
			__global float* aux1BB2g,
			__global float* aux1BB11g,
			__global float* auxBB1g,
			__global float* auxBB2g,
			__global float* auxBB11g,
			__const uint dim0, // rows
			__const uint dim2,
			__const uint dim4,
			__const uint dim6) {

    uint dim1 = get_global_size(0) / dim0; // cols
    uint dim3 = get_global_size(1) / dim2;
    uint dim5 = get_global_size(2) / dim4;

    int i = get_global_id(0) % dim0; // irow
    int j = get_global_id(0) / dim0; // icol
    int k = get_global_id(1) % dim2;
    int l = get_global_id(1) / dim2;
    int m = get_global_id(2) % dim4;
    int n = get_global_id(2) / dim4;

    int idx;
    int idxaux;

    for(int o = 0; o < dim6; o++) {
	idxaux = j + i * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim2 + m * dim0 * dim1 * dim2 * dim3;
	idx = j + i * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim2 + m * dim0 * dim1 * dim2 * dim3 + n * dim0 * dim1 * dim2 * dim3 * dim4 + o * dim0 * dim1 * dim2 * dim3 * dim4 * dim5;
	aux1BB1g[idx] = auxBB1g[idxaux];
	aux1BB2g[idx] = auxBB2g[idxaux];

	idxaux = j + i * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim2;
	idx = j + i * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim2 + n * dim0 * dim1 * dim2 * dim3 + o * dim0 * dim1 * dim2 * dim3 * dim5;
	aux1BB11g[idx] = auxBB11g[idxaux];
    }
}

/**
 * Permutes dimensions
 * @param[out] BB1g
 * @param[out] BB2g
 * @param[out] BB11g
 * @param[in] aux1BB1g
 * @param[in] aux1BB2g
 * @param[in] aux1BB11g
 * @param[in] dim0
 * @param[in] dim2
 * @param[in] dim4
 * @param[in] dim6
 */
__kernel void permuteBBg(__global float* BB1g,
			 __global float* BB2g,
			 __global float* BB11g,
			 __global float* aux1BB1g,
			 __global float* aux1BB2g,
			 __global float* aux1BB11g,
			 __const uint dim0, // rows
			 __const uint dim2,
			 __const uint dim4,
			 __const uint dim6) {

    uint dim1 = get_global_size(0) / dim0; // cols
    uint dim3 = get_global_size(1) / dim2;
    uint dim5 = get_global_size(2) / dim4;

    int i = get_global_id(0) % dim0; // irow
    int j = get_global_id(0) / dim0; // icol
    int k = get_global_id(1) % dim2;
    int l = get_global_id(1) / dim2;
    int m = get_global_id(2) % dim4;
    int n = get_global_id(2) / dim4;

    int idx;
    int idxaux;

    for(int o = 0; o < dim6; o++) {
	idxaux = j + i * dim1 + l * dim0 * dim1 + m * dim0 * dim1 * dim3 + o * dim0 * dim1 * dim3 * dim4 + k * dim0 * dim1 * dim3 * dim4 * dim6 + n * dim0 * dim1 * dim3 * dim4 * dim6 *
		 dim2;
	idx = j + i * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim2 + m * dim0 * dim1 * dim2 * dim3 + n * dim0 * dim1 * dim2 * dim3 * dim4 + o * dim0 * dim1 * dim2 * dim3 * dim4 * dim5;
	BB1g[idx] = aux1BB1g[idxaux];
	BB2g[idx] = aux1BB2g[idxaux];
    }

    idxaux = j + i * dim1 + l * dim0 * dim1 + m * dim0 * dim1 * dim3 + k * dim0 * dim1 * dim3 * dim4 + n * dim0 * dim1 * dim3 * dim4 * dim2;
    idx = j + i * dim1 + k * dim0 * dim1 + l * dim0 * dim1 * dim2 + m * dim0 * dim1 * dim2 * dim3 + n * dim0 * dim1 * dim2 * dim3 * dim4;
    BB11g[idx] = aux1BB11g[idxaux];
}
