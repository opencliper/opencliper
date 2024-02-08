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

//--------------------------------------------------------------------------------------------------------------------
//                                            Atomics
//--------------------------------------------------------------------------------------------------------------------
void atomic_add_local(volatile local float* source, const float operand) {
    float prevVal, newVal;

    do {
	prevVal = *source;
	newVal = prevVal + operand;
    }
    while(atomic_cmpxchg((volatile local unsigned int*)source, as_int(prevVal), as_int(newVal)) != as_int(prevVal));
}

//--------------------------------------------------------------------------------------------------------------------
//                                            Complex to real
//--------------------------------------------------------------------------------------------------------------------
kernel void complex2real_abs(global complexType* input, global realType* output, uint batchSize, uint inBatchDistance, uint outBatchDistance) {
    size_t inOffset = get_global_id(0);
    size_t outOffset = inOffset;

    for(uint i = 0; i < batchSize; i++) {
	output[outOffset] = hypot(input[inOffset].x, input[inOffset].y);
	inOffset += inBatchDistance;
	outOffset += outBatchDistance;
    }
}

kernel void complex2real_real(global complexType* input, global realType* output, uint batchSize, uint inBatchDistance, uint outBatchDistance) {
    size_t inOffset = get_global_id(0);
    size_t outOffset = inOffset;

    for(uint i = 0; i < batchSize; i++) {
	output[outOffset] = input[inOffset].x;
	inOffset += inBatchDistance;
	outOffset += outBatchDistance;
    }
}

kernel void complex2real_imag(global complexType* input, global realType* output, uint batchSize, uint inBatchDistance, uint outBatchDistance) {
    size_t inOffset = get_global_id(0);
    size_t outOffset = inOffset;

    for(uint i = 0; i < batchSize; i++) {
	output[outOffset] = input[inOffset].y;
	inOffset += inBatchDistance;
	outOffset += outBatchDistance;
    }
}

kernel void complex2real_arg(global complexType* input, global realType* output, uint batchSize, uint inBatchDistance, uint outBatchDistance) {
    size_t inOffset = get_global_id(0);
    size_t outOffset = inOffset;

    for(uint i = 0; i < batchSize; i++) {
	output[outOffset] = atan2(input[inOffset].y, input[inOffset].x);
	inOffset += inBatchDistance;
	outOffset += outBatchDistance;
    }
}


//--------------------------------------------------------------------------------------------------------------------
//                                            Reductions
//--------------------------------------------------------------------------------------------------------------------
kernel void reduce_sum(global realType* input, global realType* partialOutput, local realType* scratch, uint batchSize, uint batchDistance, uint realGlobalSize) {
    size_t offset = get_global_id(0);

    // If globalSize % warp size !=0, some of the last work items will lie outside input buffer
    if(offset >= realGlobalSize)
	return;

    size_t lid = get_local_id(0);
    realType workgroupSum = 0;

    for(uint nBatch = 0; nBatch < batchSize; nBatch++) {
	// Fetch input into this workgroup's local mem
	scratch[lid] = input[offset];

	uint localStride;
	for(size_t currentLocalSize = get_local_size(0); currentLocalSize >= 2; currentLocalSize /= 2) {
	    localStride = currentLocalSize / 2;
	    barrier(CLK_LOCAL_MEM_FENCE);

	    // Only the lower half of the workgroup actually works, sadly (there are half operations than operands)
	    // Maybe load the next batch into local memory?
	    // If we used OpenCL 2.0, we could do this in a single work_group_reduce_add(scratch) operation
	    if(lid < localStride)
		scratch[lid] += scratch[lid + localStride];

	    // Use one of the unused work items to check if the last element was left behind due to currentLocalSize being odd
	    else if((lid == localStride) && (currentLocalSize & 1))
		atomic_add_local(&scratch[0], scratch[currentLocalSize - 1]);
	}

	// Accumulate this batch to previous ones
	barrier(CLK_LOCAL_MEM_FENCE);
	if(lid == 0)
	    workgroupSum += scratch[0];

	offset += batchDistance;
    }

    // Store output from this workgroup for next reduce iteration
    if(lid == 0) {
	partialOutput[get_group_id(0)] = workgroupSum;
	//printf("wg %lu: wgsize=%lu, sum=%f\n", get_group_id(0), get_local_size(0), partialOutput[get_group_id(0)]);
    }
}

//--------------------------------------------------------------------------------------------------------------------
//                                            Normalization
//--------------------------------------------------------------------------------------------------------------------
kernel void normalize_show(global realType* dataSum, global realType* input, global realType* output, uint batchSize, uint batchDistance) {
    size_t offset = get_global_id(0);

    realType factor = 0.5 * get_global_size(0) * batchSize / *dataSum;

    for(uint i = 0; i < batchSize; i++) {
	output[offset] = factor * input[offset];
	offset += batchDistance;
    }
}

kernel void scalarMultiply(realType factor, global realType* input, global realType* output, uint batchSize, uint batchDistance) {
    size_t offset = get_global_id(0);

    for(uint i = 0; i < batchSize; i++) {
	output[offset] = factor * input[offset];
	offset += batchDistance;
    }
}

//--------------------------------------------------------------------------------------------------------------------
//                                            HIP-OpenCL interop.
//--------------------------------------------------------------------------------------------------------------------
kernel void getDevicePointer(global void* p, global ulong* destBuffer) {
    // destBuffer must be long enough to hold a global pointer. We assume pointers are always 64bit words, BTW...

    *destBuffer = (ulong)p;
}

//--------------------------------------------------------------------------------------------------------------------
//                                            Painting
//--------------------------------------------------------------------------------------------------------------------
kernel void reshape_show(global realType* canvas, global realType* source, uint winWidth, uint timeDim) {
    // We need X and Y coords to select the correct coil (cant't do with just offsets)

    // Get parameters of input/output data
    uint sliceWidth = getSpatialDimSize(source, 0, 0);
    uint sliceHeight = getSpatialDimSize(source, 1, 0);
    uint inFrameStride = getTemporalDimStride(source, timeDim, 0);
    uint outFrameStride = getTemporalDimStride(canvas, 0, 0);
    uint nFrames = getTemporalDimSize(canvas, 0);

    // Calculate output offset within a given frame
    uint outX = get_global_id(0);
    uint outY = get_global_id(1);
    uint outOffset_withinFrame = outY * winWidth + outX;

    // Calculate input offset within a given frame
    uint inSlice=(outX/sliceWidth) + (outY/sliceHeight) * (winWidth/sliceWidth);
    uint inY = outY % sliceHeight;
    uint inX = outX % sliceWidth;
    uint inOffset_withinFrame = inSlice * getDimStride(source, 2, 0) + inY * getSpatialDimStride(source, 1, 0) + inX;
    uint numCoils = getNumCoils(source);
    uint inOffset_withinFrame_max = getNDArrayTotalSize(source, 0) * (numCoils != 0 ? numCoils: 1);

    // Paint! Each work item paints a single output pixel per frame
    uint inFrameStart = 0;
    uint outFrameStart = 0;
    for(uint frame = 0; frame < nFrames; frame++) {
	uint outOffset = outFrameStart + outOffset_withinFrame;

	// Paint the canvas where there is actual input data. Fill with 0 otherwise
	if(inOffset_withinFrame < inOffset_withinFrame_max) {
            uint inOffset = inFrameStart + inOffset_withinFrame;
	    canvas[outOffset] = source[inOffset];
        }
	else
	    canvas[outOffset] = 0;

	inFrameStart += inFrameStride;
	outFrameStart += outFrameStride;
    }
}

//--------------------------------------------------------------------------------------------------------------------
//                                 Setting elements to initial value
//--------------------------------------------------------------------------------------------------------------------

kernel void memset_complex(global complexType* output, complexType value)  {
    size_t offset = get_global_id(0);
    printf ("kernel data, offset: %lu\toriginal real: %f\toriginal imag: %f\n", offset, output[offset].x, output[offset].y);
    output[offset] = value;
    printf ("kernel data, offset: %lu\tnew real: %f\tnew imag: %f\n", offset, output[offset].x, output[offset].y);
}

kernel void memset_real(global realType* output, realType value)  {
    size_t offset = get_global_id(0);
    output[offset] = value;
}

kernel void memset_uint(global uint* output, uint value)  {
    size_t offset = get_global_id(0);
    output[offset] = value;
}
