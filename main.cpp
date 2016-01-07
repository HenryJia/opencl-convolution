#include <iostream>
#include <random>
#include <vector>
#include <chrono>

#include <stdio.h>

#include "cl.hpp"

#include "cl_get_error.hpp"

#define GROUP_SIZE 32

int image_dim = 32;
int filter_dim = 8;
int out_dim = image_dim - filter_dim + 1;

// For loading the OpenCL kernel code
char* load_src(const char *filename) {
    FILE *fp;
    char* source;
    size_t len;

    fp = fopen(filename, "r");
    std::fseek(fp, 0, SEEK_END);
    len = std::ftell(fp);
    source = new char [len];
    std::rewind(fp);

    std::fread(source, 1, len, fp);
    std::fclose(fp);

    return source;
}

// A wrapper for setting kernel arguments
cl_int cl_set_kernel_args(cl_kernel kernel, std::vector<size_t> args_size, std::vector<void*> args) {
    cl_int cl_err = CL_SUCCESS;

    for (int i = 0; i < args.size(); i++)
        cl_err += clSetKernelArg(kernel, i, args_size[i], args[i]);
    return cl_err;
}

int main(int argc, char **argv) {

    // Load the OpenCL kernel code as a char string
    char* conv2_valid_src = load_src("../conv2valid.cl");

    // Display it
    /*for (int i = 0; conv2_valid_src[i] != NULL; i++)
        std::cout << conv2_valid_src[i];
    std::cout << "Source Printed\n";*/

    // Setup OpenCL and compile the kernel
    cl_platform_id cl_plat;           // OpenCL platform
    cl_device_id cl_dev;              // device ID
    cl_context cl_ctx;                // context
    cl_command_queue cl_queue;        // command queue
    cl_program conv2_valid;           // program
    cl_kernel conv2_valid_kernel;     // kernel

    cl_int cl_err = CL_SUCCESS;

    cl_err = clGetPlatformIDs(1, &cl_plat, NULL);
    if (cl_err != CL_SUCCESS)
        std::cerr << "clGetPlatformIDs failed: " << cl_get_error(cl_err) << std::endl;

    cl_err = clGetDeviceIDs(cl_plat, CL_DEVICE_TYPE_GPU, 1, &cl_dev, NULL);
    if (cl_err != CL_SUCCESS)
        std::cerr << "clGetDeviceIDs failed: " << cl_get_error(cl_err) << std::endl;

    cl_ctx = clCreateContext(0, 1, &cl_dev, NULL, NULL, &cl_err);
    if (cl_err != CL_SUCCESS)
        std::cerr << "clCreateContext failed: " << cl_get_error(cl_err) << std::endl;

    cl_queue = clCreateCommandQueue(cl_ctx, cl_dev, CL_QUEUE_PROFILING_ENABLE, &cl_err);
    if (cl_err != CL_SUCCESS)
        std::cerr << "clCreateCommandQueue failed: " << cl_get_error(cl_err) << std::endl;

    // Create the compute program from the source buffer
    conv2_valid = clCreateProgramWithSource(cl_ctx, 1, (const char **)&conv2_valid_src, NULL, &cl_err);
    if (cl_err != CL_SUCCESS)
        std::cerr << "clCreateProgramWithSource failed: " << cl_get_error(cl_err) << std::endl;

    cl_err = clBuildProgram(conv2_valid, 0, NULL, "-Dfilter_dim=5", NULL, NULL);
    if (cl_err != CL_SUCCESS)
        std::cerr << "clBuildProgram failed: " << cl_get_error(cl_err) << std::endl;

    // Create the compute kernel in the program we wish to run
    conv2_valid_kernel = clCreateKernel(conv2_valid, "conv2_valid", &cl_err);
    if (cl_err != CL_SUCCESS)
        std::cerr << "clCreateKernel failed: " << cl_get_error(cl_err) << std::endl;

    // Generate some random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);

    std::vector<float> img(image_dim * image_dim);
    std::vector<float> filter(filter_dim * filter_dim);
    std::vector<float> output_device(out_dim * out_dim);
    std::vector<float> output_host(out_dim * out_dim);

    for (int i = 0; i < image_dim * image_dim; i++)
        img[i] = dis(gen);
    for (int i = 0; i < filter_dim * filter_dim; i++)
        filter[i] = dis(gen);

    // Load the data into OpenCL device
    // Device input buffer
    cl_mem d_img;
    cl_mem d_filter;
    // Device output buffer
    cl_mem d_output;

    const size_t img_size = image_dim * image_dim * sizeof(float);
    const size_t filter_size = filter_dim * filter_dim * sizeof(float);
    const size_t out_size = out_dim * out_dim * sizeof(float);

    // Allocate memory in OpenCL device
    d_img = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY, img_size, NULL, &cl_err);
    if (cl_err != CL_SUCCESS)
        std::cerr << "clCreateBuffer d_img failed: " << cl_get_error(cl_err) << std::endl;
    d_filter = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY, filter_size, NULL, &cl_err);
    if (cl_err != CL_SUCCESS)
        std::cerr << "clCreateBuffer d_filter failed: " << cl_get_error(cl_err) << std::endl;
    d_output = clCreateBuffer(cl_ctx, CL_MEM_WRITE_ONLY, out_size, NULL, &cl_err);
    if (cl_err != CL_SUCCESS)
        std::cerr << "clCreateBuffer d_output failed: " << cl_get_error(cl_err) << std::endl;

    cl_err += clEnqueueWriteBuffer(cl_queue, d_img, CL_TRUE, 0, img_size, img.data(), 0, NULL, NULL);
    cl_err = clEnqueueWriteBuffer(cl_queue, d_filter, CL_TRUE, 0, filter_size, filter.data(), 0, NULL, NULL);
    if (cl_err != CL_SUCCESS)
        std::cerr << "clEnqueueWriteBuffer d_filter failed: " << cl_get_error(cl_err) << std::endl;

    int o_group_dim = GROUP_SIZE - filter_dim + 1;
    //cl_err = clSetKernelArg(conv2_valid_kernel, 0, sizeof(cl_mem), (void*)&d_img);
    void* args [] = {(void*)&d_img, (void*)&d_filter, (void*)&d_output,
                     &image_dim, &image_dim, &filter_dim, &filter_dim, &out_dim, &out_dim, &o_group_dim, &o_group_dim};
    const size_t args_size [] = {sizeof(cl_mem), sizeof(cl_mem), sizeof(cl_mem),
                     sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int)};

    cl_err = cl_set_kernel_args(conv2_valid_kernel,
                                std::vector<size_t>(args_size, args_size + sizeof(args_size) / sizeof(args_size[0])),
                                std::vector<void*>(args, args + sizeof(args) / sizeof(args[0])));
    if (cl_err != CL_SUCCESS)
        std::cerr << "cl_set_kernel_args failed: " << cl_get_error(cl_err) << std::endl;

    size_t global_size[] = {((out_dim - 1) / o_group_dim + 1) * GROUP_SIZE, ((out_dim - 1) / o_group_dim + 1) * GROUP_SIZE};
    size_t group_size[] = {GROUP_SIZE, GROUP_SIZE};
    // Execute the kernel
    // Warm it up first
    cl_err = clEnqueueNDRangeKernel(cl_queue, conv2_valid_kernel, 2, NULL, global_size, group_size, 0, NULL, NULL);
    clFinish(cl_queue);
    cl_event event;

    double total_time = 0;
    for (int i = 0; i < 1000; i++) {
        cl_err = clEnqueueNDRangeKernel(cl_queue, conv2_valid_kernel, 2, NULL, global_size, group_size, 0, NULL, &event);
        clWaitForEvents(1 , &event);
        if (cl_err != CL_SUCCESS)
            std::cerr << "clEnqueueNDRangeKernel failed: " << cl_get_error(cl_err) << std::endl;
        cl_ulong time_start, time_end;
    
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        total_time += time_end - time_start;
    }
    std::cout << "Execution time in microseconds = " << (total_time / 1000 / 1.0e3) << " us" << std::endl;

    clFinish(cl_queue);
    cl_err = clEnqueueReadBuffer(cl_queue, d_output, CL_TRUE, 0, out_size, output_device.data(), 0, NULL, NULL );
    if (cl_err != CL_SUCCESS)
        std::cerr << "clEnqueueWriteBuffer d_filter failed: " << cl_get_error(cl_err) << std::endl;

    // CPU convolution
    for (int i = 0; i < out_dim; i++) {
        for (int j = 0; j < out_dim; j++){
            float CValue = 0;
            for (int m = 0; m < filter_dim; m++)
                for (int n = 0; n < filter_dim; n++)
                    CValue += filter[m * filter_dim + n] * img[(i + m) * image_dim + (j + n)];
            output_host[i * out_dim + j] = CValue;
        }
    }

    // Display the first 5 rows and columns to check the OpenCL kernel is working correctly.
    /*std::cout << std::endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++)
            std::cout << output_device[i * out_dim + j] << '\t';
        std::cout << std::endl;
    }

    std::cout << std::endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++)
            std::cout << output_host[i * out_dim + j] << '\t';
        std::cout << std::endl;
    }*/

    float diff = 0;
    for (int i = 0; i < out_dim * out_dim; i++)
        diff += std::abs(output_host[i] - output_device[i]);

    // Check the GPU's output is the same as the CPU. There is a slight difference which should be due to accuracy differences (i.e. GPU code not wrong)
    std::cout << "Difference between OpenCL output and Naive CPU output = " << diff << std::endl;

    clReleaseMemObject(d_img);
    clReleaseMemObject(d_filter);
    clReleaseMemObject(d_output);

    getchar();
    return 0;
}