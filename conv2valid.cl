/*
 * A 2D Valid convolution kernel.
 */

/*
void conv2ValidGPU(const float* in, const float* filter, float* out, int M, int N, int filterM, int filterN)
{
	int outM = M - filterM + 1;
	int outN = N - filterN + 1;
	int o_group_M = BLOCK_DIM2 - filterM + 1;
	int o_group_N = BLOCK_DIM2 - filterN + 1;
	dim3 gridDim(((outN - 1) / o_group_N + 1), ((outM - 1) / o_group_M + 1));
	dim3 blockDim(BLOCK_DIM2, BLOCK_DIM2);
	kernelConv2Valid<<<gridDim, blockDim>>>(in, filter, out, M, N, filterM, filterN, outM, outN, o_group_M, o_group_N);
}
*/

// For Row Major
#define IDX2R(i,j,ld) (i * ld + j) // i is column, j is row, ld is total number of columns

#define GROUP_SIZE 8

__kernel void conv2_valid(__global float* in, __global float* filter, __global float* out, int M, int N, int filterM, int filterN,
                               int outM, int outN, int o_group_M, int o_group_N)
{
    //int bx = blockIdx.x;
    /*const int bx = get_group_id(0);
    //int by = blockIdx.y;
    const int by = get_group_id(1);
    //int tx = threadIdx.x;
    const int tx = get_local_id(0);
    //int ty = threadIdx.y;
    const int ty = get_local_id(1);
    const int row = by * o_group_M + ty;
    const int col = bx * o_group_N + tx;
    float CValue = 0;

    __local float dl_in[GROUP_SIZE][GROUP_SIZE];
    __local float dl_filter[GROUP_SIZE][GROUP_SIZE];

    if(row < M && col < N) {
        dl_in[ty][tx] = in[IDX2R(row, col, N)];
        if(ty < filterM && tx < filterN)
            dl_filter[ty][tx] = filter[IDX2R(ty, tx, filterN)];
    } else
        dl_in[ty][tx] = 0.0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(ty < o_group_M && tx < o_group_N && row < outM && col < outN) {
        for(int i = 0; i < filterM; i++)
            for(int j = 0; j < filterN; j++)
                CValue += dl_in[ty + i][tx + j] * dl_filter[i][j];
        out[IDX2R(row, col, outN)] = CValue;
    }*/

    int row = get_global_id(1);
    int col = get_global_id(0);
    float4 CValue = 0;

    //if(row < outM && col < outN) {
        for(int i = 0; i < filterM; i++) {
            int k = 0;
            int img_idx = IDX2R((row + i), (col + 0), N);
            int fltr_idx = IDX2R(i, 0, filterN);
            for(int j = 0; k <= filterN - 4; j++, k += 4)
                CValue += vload4(0, &in[img_idx]) * vload4(0, &filter[fltr_idx]);

            for (; k < filterN; k++)
                CValue.x += in[img_idx + k] * filter[fltr_idx + k];
        }
        out[IDX2R(row, col, outN)] = CValue.x + CValue.y + CValue.z + CValue.w;
    //}
}

// Note, OpenCL kernel code must end with newline
