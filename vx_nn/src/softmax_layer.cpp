/*
Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "kernels.h"

struct SoftmaxLayerLocalData {
    NeuralNetworkCommonHandle * handle;
    float alpha;
    float beta;
    miopenTensorDescriptor_t input_desc;
    cl_mem input_mem;
    miopenTensorDescriptor_t output_desc;
    cl_mem output_mem;
    int dim_in;
    int dim_out;
};

static vx_status VX_CALLBACK validateSoftmaxLayer(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check tensor dimensions
    vx_enum type;
    vx_size num_dims;
    vx_size input_dims[4] = { 1, 1, 1, 1 }, output_dims[4] = { 1, 1, 1, 1 };
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if(num_dims != 2 && num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: softmax: #0 num_dims=%ld (must be 2 or 4)\n", num_dims);
    if(type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: softmax: #0 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &input_dims[4-num_dims], num_dims*sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if(num_dims != 2 && num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: softmax: #1 num_dims=%ld (must be 2 or 4)\n", num_dims);
    if(type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: softmax: #1 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &output_dims[4-num_dims], num_dims*sizeof(vx_size)));
    if (output_dims[3] != input_dims[3] || output_dims[2] != input_dims[2] ||
        output_dims[1] != input_dims[1] || output_dims[0] != input_dims[0])
        return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: softmax: dims input[%ld,%ld,%ld,%ld] != output[%ld,%ld,%ld,%ld]\n",
                    input_dims[0], input_dims[1], input_dims[2], input_dims[3],
                    output_dims[0], output_dims[1], output_dims[2], output_dims[3]);

    // output tensor configuration
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &output_dims[4-num_dims], num_dims*sizeof(vx_size)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processSoftmaxLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    SoftmaxLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    miopenHandle_t miopenHandle = data->handle->miopen_handle;

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));

    ERROR_CHECK_STATUS(miopenSoftmaxForward(miopenHandle, &data->alpha, data->input_desc, data->input_mem, &data->beta, data->output_desc, data->output_mem));

#if ENABLE_DUMP_LAYERS
    clFinish(data->handle->cmdq);
    vx_size num_dims;
    vx_size input_dims[4] = { 1, 1, 1, 1 }, output_dims[4] = { 1, 1, 1, 1 };
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &input_dims[4-num_dims], num_dims*sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &output_dims[4-num_dims], num_dims*sizeof(vx_size)));

    long input_count = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];
    long output_count = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];

    std::string input_file_name = "out/" + std::to_string(counter) + "_ann_softmax_layer_input";
    std::string output_file_name = "out/" + std::to_string(counter) + "_ann_softmax_layer_output";
    FILE * fs_inputs = fopen(input_file_name.c_str(), "wb");
    FILE * fs_outputs = fopen(output_file_name.c_str(), "wb");

    float * inputs = new float[input_count];
    float * outputs = new float[output_count];

    cl_int err = clEnqueueReadBuffer(data->handle->cmdq, data->input_mem, CL_TRUE, 0, sizeof(float) * input_count, inputs, 0, NULL, NULL);
    if (err != CL_SUCCESS) std::cout << "ERROR in reading input buffer softmax." << std::endl;
    err = clEnqueueReadBuffer(data->handle->cmdq, data->output_mem, CL_TRUE,0, sizeof(float) * output_count, outputs, 0, NULL, NULL);
    if (err != CL_SUCCESS) std::cout << "ERROR in reading outpt buffer softmax." << std::endl;

    fwrite(inputs, sizeof(float), input_count, fs_inputs);
    fclose(fs_inputs);
    delete inputs;

    fwrite(outputs, sizeof(float),output_count, fs_outputs);
    fclose(fs_outputs);
    delete outputs;

#endif

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeSoftmaxLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    SoftmaxLayerLocalData * data = new SoftmaxLayerLocalData;
    memset(data, 0, sizeof(*data));
    ERROR_CHECK_STATUS(createGraphHandle(node, &data->handle));

    //Parameters input and output.
    vx_size num_dims, input_dims[4] = { 1, 1, 1, 1 }, output_dims[4] = { 1, 1, 1, 1 };
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &input_dims[4-num_dims], num_dims * sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &output_dims[4-num_dims], num_dims * sizeof(vx_size)));

    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->input_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->output_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->input_desc, miopenFloat, input_dims[3], input_dims[2], input_dims[1], input_dims[0]));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->output_desc, miopenFloat, output_dims[3], output_dims[2], output_dims[1], output_dims[0]));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
    
    data->alpha = 1;
    data->beta = 0;

#if ENABLE_DEBUG_PRINT_DIMS
    std::cout << "softmax input " << input_dims[3] << " " << input_dims[2] << " " << input_dims[1] << " " << input_dims[0] << " ";
    std::cout << "output " << output_dims[3] << " " << output_dims[2] << " " << output_dims[1] << " " << output_dims[0] << std::endl;
#endif

    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeSoftmaxLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    SoftmaxLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->input_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->output_desc));
    if (data) {
        ERROR_CHECK_STATUS(releaseGraphHandle(node, data->handle));
        delete data;
    }
    return VX_SUCCESS;
}

vx_status publishSoftmaxLayer(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "org.khronos.nn_extension.softmax_layer", VX_KERNEL_SOFTMAX_LAYER, processSoftmaxLayer, 2, validateSoftmaxLayer, initializeSoftmaxLayer, uninitializeSoftmaxLayer);
    ERROR_CHECK_OBJECT(kernel);

    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxSoftmaxLayer(vx_graph graph, vx_tensor inputs, vx_tensor outputs)
{
    vx_reference params[] = {
        (vx_reference)inputs,
        (vx_reference)outputs
    };
    return createNode(graph, VX_KERNEL_SOFTMAX_LAYER, params, sizeof(params)/sizeof(params[0]));
}
