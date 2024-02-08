
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
//#define KERNEL_DEBUG
__kernel void arrayAdd_kernel(__global realType* arrayA, __global realType* arrayB, __global realType* arrayC,
			      unsigned int RowsA, unsigned int ColsA) {
    // int row = get_global_id(0);
    int row = get_global_id(0) / ColsA;
    // int col = get_global_id(1);
    int col = get_global_id(0) - row * ColsA;
    if((row < RowsA) && (col < ColsA)) {
	arrayC[row * ColsA + col] = arrayA[row * ColsA + col] + arrayB[row * ColsA + col];
    }
    else {
	return;
    }
}
