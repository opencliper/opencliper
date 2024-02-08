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
/*
 * vectorNormalization.cl
 *
 *  Modified on: 29 de oct. de 2021
 *      Author: Emilio López-Ales
 */

#include <OpenCLIPER/kernels/hostKernelFunctions.h>

__kernel void vectorNormalization(__global float2* in, __global float2* out,  __const float mu) {

	int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);

	uint cols = getSpatialDimSize(in, COLUMNS, 0);
	uint rows = getSpatialDimSize(in, ROWS, 0);
	uint idx =  k * rows * cols + j * cols + i;

	float factor = fmax(mu, length(in[idx]));

	out[idx] = in[idx] / factor;

}


