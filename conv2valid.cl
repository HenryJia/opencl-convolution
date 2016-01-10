/*
 * A 2D Valid convolution kernel.
 */

// For Row Major
#define IDX2R(i,j,ld) (i * ld + j) // i is column, j is row, ld is total number of columns

__kernel void conv2_valid(__global float* in, __global float* filter, __global float* out, int M, int N, int filterM, int filterN,
                          int outM, int outN) {

    /*
     * The vectorization length can be adjusted by adjusting the number behind float and vload. Supported values are 2, 4, 6, 8 or 16.
     * The resulting sum at the end should also be adjusted along with the vectorization length.
     */
    int row = get_global_id(1);
    int col = get_global_id(0);
    float8 CValue = 0;

    for(int i = 0; i < filterM; i++) {
        int k = 0;
        int img_idx = IDX2R((row + i), (col + 0), N);
        int fltr_idx = IDX2R(i, 0, filterN);
        for(int j = 0; k <= filterN - 8; j++, k += 8)
            CValue += vload8(j, in + img_idx) * vload8(j, filter + fltr_idx);

        for (; k < filterN; k++)
            CValue.s0 += in[img_idx + k] * filter[fltr_idx + k];
    }
    out[IDX2R(row, col, outN)] = CValue.s0 + CValue.s1 + CValue.s2 + CValue.s3 + CValue.s4 + CValue.s5 + CValue.s6 + CValue.s7;
}

// Note, OpenCL kernel code must end with newline
