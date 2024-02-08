/*
 * RCS/CVS version control info
 * $Id: reduce_kernel.cl,v 1.2 2016/11/02 12:34:19 manrod Exp $
 * $Revision: 1.2 $
 * $Date: 2016/11/02 12:34:19 $
 */

#include <OpenCLIPER/kernels/hostKernelFunctions.h>

kernel void blankDataRows(global complexType* input, global const uint* samplingMask, uint frameOffset_data, uint frameOffset_samplingMask) {
    // Point to the beginning of this frame
    uint offset=frameOffset_data;

    // Shift to our coil
    offset += get_global_id(2) * getCoilStride(input, 0);

    // Shift to our line
    offset += samplingMask[frameOffset_samplingMask + get_global_id(1)] * getSpatialDimStride(input, 1, 0);

    // Shift to our column
    offset += get_global_id(0);

    // Blank!
    input[offset] = 0;
}
