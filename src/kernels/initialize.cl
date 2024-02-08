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

/*! \file initialize.cl
 *	\brief File containing every kernel being executed for initialization
 *  \date May 24, 2017
 *  \author Elena Martin Gonzalez
 */

#include <OpenCLIPER/kernels/hostKernelFunctions.h>

/**
 * Initializes a realType array - every element to zero
 * @param[out] a array initialized to zero
 */
__kernel void initZero(__global float* a) {
    int id = get_global_id(0);
    a[id] = 0.0;
}

/**
 * Initializes a region of a complex array
 * @param[out] a array initialized to zero
 */
__kernel void initZeroRect(__global float2* a,
			   __global int* rows2zero,
			   __global int* cols2zero) {

    uint i = get_global_id(0); // irow
    uint j = get_global_id(1); // icol
    uint k = get_global_id(2); // frame

    uint width = getSpatialDimSize(a, COLUMNS, 0);
    uint height = getSpatialDimSize(a, ROWS, 0);

    int id = cols2zero[j] + rows2zero[i] * width + k * width * height;

    a[id] = 0.0;
}

/**
 * Initializes a complexType array from another complexType array - every element in a equal to b
 * @param[out] a array being initialized
 * @param[in] b array used to initialize
 */
__kernel void copyDataGPU(__global float2* a,
			  __global float2* b) {
    int id = get_global_id(0);
    a[id] = b[id];
}

/**
 * Initializes the array - every element in a equal to b
 * @param[out] a array being initialized
 * @param[in] b array used to initialize
 */
__kernel void dataNormalization(__global float2* a,
				__global float2* b,
				__const float oldmax,
				__const float newmax) {
    int id = get_global_id(0);
    float factor = newmax / oldmax;
    //a[id].x = factor*sqrt(pow(b[id].x,2)+pow(b[id].y,2));
    a[id].x = factor * hypot(b[id].x, b[id].y);
    a[id].y = 0.0;
}

/**
 * Initializes dx (gradient of the image)
 * @param[out] dx array being initialized
 * @param[in] d1 array used to fill the first 'element' of dx fourth dimension
 * @param[in] d2 array used to fill the second 'element' of dx fourth dimension
 */
__kernel void initdx(__global float* dx,
		     __global float* d1,
		     __global float* d2) {

    int row = get_global_id(0);
    int col = get_global_id(1);
    int frame = get_global_id(2);

    uint width = getSpatialDimSize(d1, COLUMNS, 0);
    uint height = getSpatialDimSize(d1, ROWS, 0);
    uint numFrames = getTemporalDimSize(d1, 0);

    uint idx = col + row * width + frame * width * height;

    dx[idx] = d1[idx];
    dx[idx + width * height * numFrames] = d2[idx];
}
