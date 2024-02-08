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

/*! \file reductions.cl
 *	\brief File containing kernels which executes a sum
 *  \date Nov 23, 2017
 *  \author Elena Martin Gonzalez
 */

#include <OpenCLIPER/kernels/hostKernelFunctions.h>


/**
 * Reduces dimensionality performing a sum in 7-th dimension and the corresponding products by the regularization weights 
 * @param[out] aux
 * @param[in] dthetax gradient transformation - first order spatial
 * @param[in] dthetax2 gradient transformation - second order spatial
 * @param[in] dthetaxy gradient transformation - cross spatial
 * @param[in] dthetat gradient transformation - first order temporal
 * @param[in] dthetat2 gradient transformation - second order temporal
 * @param[in] lambda0 weight for first order spatial derivatives
 * @param[in] lambda1 weight for second order spatial derivatives
 * @param[in] lambda2 weight for first order temporal derivatives
 * @param[in] lambda3 weight for second order temporal derivatives
 */
__kernel void reduction7Dto6D(__global float* aux,
			      __global float* dthetax,
			      __global float* dthetax2,
			      __global float* dthetaxy,
			      __global float* dthetat,
			      __global float* dthetat2,
			      __const float lambda0,
			      __const float lambda1,
			      __const float lambda2,
			      __const float lambda3) {

    int idx6 = get_global_id(0);
    int idx7 = idx6;
    uint dim6 = getSpatialDimSize(dthetax, 6, 0);
    float sumando1 = 0.0f;
    float sumando2 = 0.0f;
    float sumando3 = dthetaxy[idx6];
    float sumando4 = dthetat[idx6];
    float sumando5 = dthetat2[idx6];

    for(int p = 0; p < dim6; p++) { // 7D
	sumando1 += dthetax[idx7];
	sumando2 += dthetax2[idx7];
	idx7 += getSpatialDimStride(dthetax, 6, 0);
    }
    aux[idx6] = lambda0 * sumando1 + lambda1 * sumando2 + lambda1 * 2 * sumando3 + lambda2 * sumando4 + lambda3 * sumando5;
}


/**
 * Reduces dimensionality performing a sum in the selected dimension
 * @param[out] output array after sum
 * @param[in] input array to reduce
 * @param[in] dim2sum dimension in which perform the sum
 */
__kernel void reductionsum(__global float* output,
			   __global float* input,
			   __const uint dim2sum) {

    int i    = get_global_id(0);
    int idx  = i;
    uint dim = getSpatialDimSize(input, dim2sum, 0);

    float suma = 0.0;
    for(int k = 0; k < dim; k++) {
	suma += input[idx];
	idx += getSpatialDimStride(input, dim2sum, 0);
    }
    output[i] = suma;
}


/**
 * Reduces dimensionality performing a sum
 * @param[out] output array after sum
 * @param[in] input array to reduce
 */
__kernel void cost6DReduction(__global float* cost2,
			      __global float* cost1) {

    uint dim0 = getSpatialDimSize(cost1, 0, 0);
    uint dim1 = getSpatialDimSize(cost1, 1, 0);
    uint dim2 = getSpatialDimSize(cost1, 2, 0);
    uint dim5 = getSpatialDimSize(cost1, 3, 0);

    int i = get_global_id(0);
    int m = get_global_id(1);
    int n = get_global_id(2);

    int idx = m + i * dim2 + n * dim0 * dim2;
    int idxcost1;
    float suma = 0.0f;
    for(int j = 0; j < dim1; j++) {
	idxcost1 = i + j * dim1 + m * dim0 * dim1 + n * dim0 * dim1 * dim2;
	suma += cost1[idxcost1];
    }
    cost2[idx] = suma;
}


/**
* Determines the total cost of a given metric array.
* Sum the metric values only in the Region Of Interest (ROI)
* @param[out] cost total cost
* @param[in] V metric array
* @param[in] X ROI mask, determines the subset to take into account
* @param[in] iter current iteration in optimization loop
*/
__kernel void costReduction(__global float* cost,
			    __global float* V,
			    __global int* X,
			    __const int iter) {

    uint dims = get_global_size(0);

    cost[iter] = 0.0f;
    // Sum of the metric values only in the ROI
    for(int i = 0; i < dims; i++) {
	cost[iter] += V[i] * X[i];
    }
}
