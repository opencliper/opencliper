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
#include <OpenCLIPER/kernels/hostKernelFunctions.h>

__kernel void arrayMult_kernel(__global realType* arrayA, __global realType* arrayB, __global realType* arrayC,
			       unsigned int RowsA, unsigned int ColsA, unsigned int ColsB) {
    int row = get_global_id(0) / ColsA;
    int col = get_global_id(0) - row * ColsA;

    if((row < RowsA) && (col < ColsA)) {
	unsigned int ColsC = ColsB;
	float res = 0.0;
	for(unsigned int k = 0; k < ColsA; k ++) {
	    res += arrayA[row * ColsA + k] * arrayB[k * ColsB + col];
	}
	arrayC[row * ColsC + col] = res;
    }
    else {
	return;
    }
}
