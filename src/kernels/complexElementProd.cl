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
 * complexElementProd.cl
 * 
 *  Modified on: 29 de oct. de 2021
 *      Author: Emilio López-Ales
 */

#include <OpenCLIPER/kernels/hostKernelFunctions.h>

kernel void complexElementProd_kernel(global complexType* inBuffer, global complexType* sensMaps, global complexType* outBuffer, uint conjugateMask)  {
	uint inOffset = get_global_id(0);
	uint outOffset = get_global_id(0);
	uint sensMapsOffset = get_global_id(0);

	uint inCoilStride = getCoilStride(inBuffer,0);
	uint outCoilStride = getCoilStride(outBuffer,0);
	uint sensMapsCoilStride = getCoilStride(sensMaps,0);

	// Note that the output is ALWAYS separated in coils whereas the input may consist of several coils or be the X-space image (and hence have no coils)
	uint nCoils = getNumCoils(outBuffer);
    
	uint nFrames = getTemporalDimSize(inBuffer, 0);

	for(uint frame = 0; frame < nFrames; frame++) {
		for(uint coil = 0; coil < nCoils; coil++) {
			complexType in = inBuffer[inOffset];
			complexType sm = sensMaps[sensMapsOffset];
			sm.y = as_float(as_uint(sm.y) ^ conjugateMask);

			outBuffer[outOffset].x = in.x * sm.x - in.y * sm.y;
			outBuffer[outOffset].y = in.x * sm.y + in.y * sm.x;

			inOffset += inCoilStride;
			outOffset += outCoilStride;
			sensMapsOffset += sensMapsCoilStride;
		}
		// If inCoilStride==0, then outCoilStride equals inFrameStride, so inOffset+=inFrameStride
		// If inCoilStride!=0, then outCoilStride equals inCoilStride, so inOffset+=0
		inOffset += (outCoilStride - inCoilStride);

		// go back to first sensitivity map for next frame
		sensMapsOffset = get_global_id(0);
	}
}
