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

__kernel void rss_kernel(__global float2* pInputBuffer, __global float2* pOutputBuffer) {


    int i = get_global_id(0);
    int j = get_global_id(1);
    int z = get_global_id(2);

    //int cols=get_global_size(0);
    //int rows=get_global_size(1);
    //int numFrames=get_global_size(2);

    uint cols = getSpatialDimSize(pInputBuffer, COLUMNS, 0);
    uint rows = getSpatialDimSize(pInputBuffer, ROWS, 0);
    uint numFrames = getTemporalDimSize(pInputBuffer, 0);
    uint numCoils = getNumCoils(pInputBuffer);


    int idxIn = z * cols * rows * numCoils + j * cols + i;
    int idxOut = z * cols * rows + j * cols + i;
    unsigned int k;

    pOutputBuffer[idxOut].x = 0;

    for(k = 0; k < numCoils; k++) {

	pOutputBuffer[idxOut].x += pInputBuffer[idxIn + k * rows * cols].x * pInputBuffer[idxIn + k * rows * cols].x + pInputBuffer[idxIn + k * rows * cols].y * pInputBuffer[idxIn + k *
				   rows * cols].y;

    }

    pOutputBuffer[idxOut].x = sqrt(pOutputBuffer[idxOut].x);
}
