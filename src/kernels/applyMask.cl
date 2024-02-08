#include <OpenCLIPER/kernels/hostKernelFunctions.h>

//#define DEBUG

kernel void applyMask_complex(global complexType* input, global const uchar* mask) {
    dimIndexType dataOffset = get_global_id(0);
    dimIndexType maskOffset = get_global_id(0);
    dimIndexType numCoils = getNumCoils(input);
    dimIndexType numFrames = getTemporalDimSize(input, 0); // of input for temporal dimension 0
    dimIndexType coilStride = getCoilStride(input, 0); // of input for NDArray 0
    dimIndexType dataFrameStride = getTemporalDimStride(input, 0, 0); // of input for temporal dimension 0, NDArray 0
    dimIndexType maskFrameStride = getTemporalDimStride(mask, 0, 0); // of input for temporal dimension 0, NDArray 0
    uint2 maskWord;
#ifdef DEBUG
    if (get_global_id(0) == 0) {
		printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! applyMask_kernel !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
		printf("dataOffset: %u\t", dataOffset);
		printf("maskOffset: %u\t", maskOffset);
		printf("numCoils: %u\t", numCoils);
		printf("numFrames: %u\t", numFrames);
		printf("coilStride: %u\t", coilStride);
		printf("dataFrameStride: %u\t", dataFrameStride);
		printf("maskFrameStride: %u\n", maskFrameStride);
		printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! end applyMask_kernel !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
		printf("\n");
    }
#endif
    for (int frame = 0; frame < numFrames; frame++) {
    	dataOffset = get_global_id(0) + frame * dataFrameStride; // increment by frameStride is not valid, coilOffset must be reset to 0
		for (int coil = 0; coil < numCoils; coil ++) {
			//maskWord = select((uint2)0, (uint2)-1, (uint2)mask[maskOffset]);
			//input[dataOffset] = as_float2(as_uint2(input[dataOffset]) & maskWord);
			if (mask[maskOffset] == 0) {
				input[dataOffset] = 0;
			}
			dataOffset += coilStride;
		}
		maskOffset += maskFrameStride;
    }
}

kernel void applyMask_real(global realType* input, global const uchar* mask) {
    dimIndexType dataOffset = get_global_id(0);
    dimIndexType maskOffset = get_global_id(0);
    dimIndexType numCoils = getNumCoils(input);
    dimIndexType numFrames = getTemporalDimSize(input, 0); // of input for temporal dimension 0
    dimIndexType coilStride = getCoilStride(input, 0); // of input for NDArray 0
    dimIndexType dataFrameStride = getTemporalDimStride(input, 0, 0); // of input for temporal dimension 0, NDArray 0
    dimIndexType maskFrameStride = getTemporalDimStride(mask, 0, 0); // of input for temporal dimension 0, NDArray 0
    uint2 maskWord;
#ifdef DEBUG
    if (get_global_id(0) == 0) {
		printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! applyMask_kernel !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
		printf("dataOffset: %u\t", dataOffset);
		printf("maskOffset: %u\t", maskOffset);
		printf("numCoils: %u\t", numCoils);
		printf("numFrames: %u\t", numFrames);
		printf("coilStride: %u\t", coilStride);
		printf("dataFrameStride: %u\t", dataFrameStride);
		printf("maskFrameStride: %u\n", maskFrameStride);
		printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! end applyMask_kernel !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
		printf("\n");
    }
#endif
    for (int frame = 0; frame < numFrames; frame++) {
		dataOffset = get_global_id(0) + frame * dataFrameStride; // increment by frameStride is not valid, coilOffset must be reset to 0
		for (int coil = 0; coil < numCoils; coil ++) {
			//maskWord = select((uint2)0, (uint2)-1, (uint2)mask[maskOffset]);
			//input[dataOffset] = as_float2(as_uint2(input[dataOffset]) & maskWord);
			if (mask[maskOffset] == 0) {
				input[dataOffset] = 0;
			}
			dataOffset += coilStride;
		}
		maskOffset += maskFrameStride;
    }
}