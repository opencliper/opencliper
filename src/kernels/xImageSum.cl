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
 * RCS/CVS version control info
 * $Id: reduce_kernel.cl,v 1.2 2016/11/02 12:34:19 manrod Exp $
 * $Revision: 1.2 $
 * $Date: 2016/11/02 12:34:19 $
 */

#include <OpenCLIPER/kernels/hostKernelFunctions.h>

kernel void xImageSum_kernel(global complexType* pInBuffer, global complexType* pOutBuffer) {
	uint inOffset = get_global_id(0);
	uint outOffset = get_global_id(0);
	uint inCoilStride = getCoilStride(pInBuffer,0);	
	uint outFrameStride = getTemporalDimStride(pOutBuffer,0,0);
	uint nInFrames = getTemporalDimSize(pInBuffer, 0);
	uint nCoils = getNumCoils(pInBuffer);

	complexType acum;
	for(uint frame = 0; frame < nInFrames; frame++) {
		acum = 0;
		for(uint coil = 0; coil < nCoils; coil++) {
			acum += pInBuffer[inOffset];
			inOffset += inCoilStride;
		}

		pOutBuffer[outOffset] = acum;
		outOffset += outFrameStride;
	}
}

