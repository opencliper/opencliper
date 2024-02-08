/* Copyright (C) 2018 Federico Simmross Wattenberg,
 *                    Manuel Rodríguez Cayetano,
 *                    Javier Royuela del Val,
 *                    Elena Martín González,
 *                    Elisa Moya Sáez,
 *                    Marcos Martín Fernández and
 *                    Carlos Alberola López
 *                    Emilio López-Ales
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
 * complexPow.cl
 *
 *  Created on: 29 de sep. de 2021
 *      Author: Emilio López-Ales
 */

#include <OpenCLIPER/kernels/hostKernelFunctions.h>

__kernel void complexPow(__global float2* in, __global float2* out) {

    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    uint cols = getSpatialDimSize(in, COLUMNS, 0);
    uint rows = getSpatialDimSize(in, ROWS, 0);
//     uint idx =  k * rows * cols + j * cols + i;
//     uint idx = get_global_id(0);
    
    uint nCoils = getNumCoils(in);
    uint nFrames = getTemporalDimSize(in, 0);
    uint inCoilStride = getCoilStride(in, 0);
    uint outCoilStride = getCoilStride(out, 0);
    uint inFrameStride = getTemporalDimStride(in, 0, 0);
//     uint idxlength = 0;
    
    for(uint frame = 0; frame < nFrames; frame++){
        uint idx = (k * rows * cols + j * cols + i) + (frame * inFrameStride);
        for(uint coil = 0; coil < nCoils; coil++) {
            complexType inC = in[idx];
            complexType outC = out[idx];
            outC.x = (inC.x * inC.x) - (inC.y * inC.y);
            outC.y = 2 * inC.x * inC.y;
        
            out[idx] = outC;
            
            idx += inCoilStride;
        }
    }
    
}

