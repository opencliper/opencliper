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

/*! \file temporalTV.cl
 *	\brief File containing every kernel about temporal total variation
 *  \date May 15, 2018
 *  \author Elisa Moya Saez
 */
/*
 * temporalTV.cl
 *
 *  Modified on: 29 de oct. de 2021
 *      Author: Emilio López-Ales
 */

#include <OpenCLIPER/kernels/hostKernelFunctions.h>


__kernel void operator_tTV(__global float2* in, __global float2* out, __const uint numFrames) {

	int i = get_global_id(0); // rowID
	int j = get_global_id(1); // colID
	int k = get_global_id(2); // sliceID

	uint cols = getSpatialDimSize(in, COLUMNS, 0);
	uint rows = getSpatialDimSize(in, ROWS, 0);
	uint slices = getSpatialDimSize(in, SLICES, 0);
	if(slices == 0)
		slices = 1;

	uint idx1 = (k * cols * rows) + (j * cols) + i;
	uint idx2 = idx1;
	uint idxlength = 0;

	//First frame different
	out[idx1]=in[idx1]-in[idx1+(numFrames-1)*cols*rows*slices];

	for(uint f=1; f<numFrames; f++){
		idxlength=getTemporalDimStride(in, 0, f);
		idx1+=idxlength;

		out[idx1]=in[idx1]-in[idx2];

		idx2+=idxlength;
	}
}

__kernel void operator_tTVadj(__global float2* in, __global float2* out, __const uint numFrames) {

	int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);

	uint cols = getSpatialDimSize(in, COLUMNS, 0);
	uint rows = getSpatialDimSize(in, ROWS, 0);
	uint slices = getSpatialDimSize(in, SLICES, 0);
	if(slices == 0)
		slices = 1;

	uint idx1 = (k * cols * rows) + (j * cols) + i;
	uint idx2 = idx1;
	uint idxlength = 0;

	for(uint f=0; f<numFrames-1; f++){
		idxlength = getTemporalDimStride(in, 0, f);
		idx2+=idxlength;
        
		out[idx1]=in[idx2]-in[idx1];
        
		idx1+=idxlength;
	}

	//Last frame different
	uint idxFirst = (k * cols * rows) + (j * cols) + i;
	uint idxLast = idxFirst+(numFrames-1)*cols*rows*slices;
	out[idxLast] = in[idxFirst]-in[idxLast];
}

