/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
#include <vector>
#include <array>

struct RNNLayerLocalData
{
    NeuralNetworkCommonHandle * handle;
    cl_mem input_mem;
    cl_mem output_mem;
    cl_mem weights_mem;
    cl_mem hidden_input_mem;
    cl_mem hidden_output_mem;
    cl_mem cell_input_mem;
    cl_mem cell_output_mem;
    cl_mem bias_mem;
    cl_mem workspace;
    size_t workspace_size;
    std::vector<miopenTensorDescriptor_t> input_desc_array;
    std::vector<miopenTensorDescriptor_t> output_desc_array;
    miopenTensorDescriptor_t weight_desc;
    miopenTensorDescriptor_t hidden_input_desc;
    miopenTensorDescriptor_t bias_desc;
    miopenTensorDescriptor_t output_desc;
    miopenRNNDescriptor_t rnnDesc;
    vx_size seq_len;
};

static vx_status VX_CALLBACK validateRNNLayer(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    //check scalar type for seq_length and mode.
    vx_enum type;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar) parameters[5], VX_SCALAR_TYPE, &type, sizeof(type)));
    if (type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: seqlen : #5 type = %d (must be VX_TYPE_FLOAT32) \n", type);
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar) parameters[4], VX_SCALAR_TYPE, &type, sizeof(type)));
    if (type != VX_TYPE_INT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: mode : #4 type = %d (must be VX_TYPE_INT32) \n", type);

    //check input, weights, hidden and output tensors.
    vx_size num_dims;
    //input tensor.
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    if(type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: input: #0 type=%d (must be VX_TYPE_FLOAT32)\n", type);

    //weights tensor.
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    if(type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: weights: #1 type=%d (must be VX_TYPE_FLOAT32)\n", type);

    //hidden input tensors.
    if (parameters[2]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[2], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        if(type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: hidden_input: #2 type=%d (must be VX_TYPE_FLOAT32)\n", type);
    }

    //bias tensor.
    if (parameters[3]) {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[3], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        if(type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: bias: #3 type=%d (must be VX_TYPE_FLOAT32)\n", type);
    }

    //output tensors.
    vx_size output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[6], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[6], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    if(type != VX_TYPE_FLOAT32) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: output: #7 type=%d (must be VX_TYPE_FLOAT32)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));


    //output tensor configuration.
    type = VX_TYPE_FLOAT32;
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[4], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[4], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[4], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processRNNLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
{

    RNNLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));

    //Run forward inference.
    if (parameters[2]) {
        ERROR_CHECK_MIOPEN_STATUS(miopenRNNForwardInference(data->handle->miopen_handle, data->rnnDesc, data->seq_len, data->input_desc_array.data(), data->input_mem, data->hidden_input_desc,
                                                            data->hidden_input_mem, nullptr, nullptr, data->weight_desc, data->weights_mem, data->output_desc_array.data(), data->output_mem,
                                                            nullptr, nullptr, nullptr, nullptr, data->workspace, data->workspace_size));
    }
    else {
        ERROR_CHECK_MIOPEN_STATUS(miopenRNNForwardInference(data->handle->miopen_handle, data->rnnDesc, data->seq_len, data->input_desc_array.data(), data->input_mem, nullptr,
                                                            nullptr, nullptr, nullptr, data->weight_desc, data->weights_mem, data->output_desc_array.data(), data->output_mem,
                                                            nullptr, nullptr, nullptr, nullptr, data->workspace, data->workspace_size));
    }


    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeRNNLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{

    RNNLayerLocalData * data = NULL;
    memset(data, 0, sizeof(*data));
    ERROR_CHECK_STATUS(createGraphHandle(node, &data->handle));

    //sequence length.
    vx_size mode = 0;
    data->seq_len = 0;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar) parameters[5], &data->seq_len, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar) parameters[4], &mode, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    //input, weights, hidden and output dimensions.
    vx_size input_dims[4] = {1,1,0, 0}, output_dims[4] = {1,1,0,0}, weight_dims[4] = {1,1,0,0}, hidden_dims[3], bias_dims[1];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[1], VX_TENSOR_DIMS, weight_dims, sizeof(weight_dims)));
    if (parameters[2]) ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[2], VX_TENSOR_DIMS, hidden_dims, sizeof(hidden_dims)));
    if (parameters[3]) ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[3], VX_TENSOR_DIMS, bias_dims, sizeof(bias_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[5], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    //tensor descriptors.
    miopenTensorDescriptor_t input_desc;
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&input_desc));
    data->input_desc_array.push_back(input_desc);
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->weight_desc));
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->output_desc));
    data->output_desc_array.push_back(data->output_desc);
    if (parameters[2]) ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->hidden_input_desc));
    if (parameters[3]) ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->bias_desc));

    //set tensor descriptors.
    for (int i=0; i < data->input_desc_array.size(); i++) {
        std::array<int, 2> in_lens = {(vx_int32) input_dims[2], (vx_int32) input_dims[3]};
        ERROR_CHECK_MIOPEN_STATUS(miopenSetTensorDescriptor(data->input_desc_array[i], miopenFloat, 2, in_lens.data(), nullptr));
    }

    for (int i=0; i < data->output_desc_array.size(); i++) {
        std::array<int, 2> out_lens = { (vx_int32) output_dims[2], (vx_int32) output_dims[3]};
        ERROR_CHECK_MIOPEN_STATUS(miopenSetTensorDescriptor(data->output_desc_array[i], miopenFloat, 2, out_lens.data(), nullptr));
    }

    if (parameters[2]) {
        vx_size hidden_l =  data->seq_len;
        vx_size hidden_h =  input_dims[2];
        std::array<int, 3> hidden_lens = {(vx_int32) hidden_l , (vx_int32) input_dims[2], (vx_int32) hidden_h};
        ERROR_CHECK_MIOPEN_STATUS(miopenSetTensorDescriptor(data->hidden_input_desc, miopenFloat, 3, hidden_lens.data(), nullptr));
    }

    //get rnn mode.
    miopenRNNMode_t rnn_mode;
    if (mode == 0) {
        rnn_mode = miopenRNNRELU;
    }
    else if (mode == 1) {
        rnn_mode = miopenRNNTANH;
    }
    else if (mode == 2) {
        rnn_mode = miopenLSTM;
    }
    else if (mode == 3) {
        rnn_mode = miopenGRU;
    }
    else {
        return VX_FAILURE;
    }

    //get rnn bias mode.
    miopenRNNBiasMode_t bias_mode = miopenRNNNoBias;
    if (parameters[3]) {
        bias_mode = miopenRNNwithBias;
    }

    //set RNN descriptor.
    ERROR_CHECK_MIOPEN_STATUS(miopenSetRNNDescriptor(data->rnnDesc, input_dims[2], data->seq_len, miopenRNNlinear, miopenRNNunidirection, rnn_mode, bias_mode, miopenRNNdefault, miopenFloat));

    //workspace.
    ERROR_CHECK_MIOPEN_STATUS(miopenGetRNNWorkspaceSize(data->handle->miopen_handle, data->rnnDesc, data->seq_len, data->input_desc_array.data(), &data->workspace_size));
    if (data->workspace_size > 0) {
        vx_context   vxContext = vxGetContext((vx_reference)node);
        cl_context context;
        ERROR_CHECK_STATUS(vxQueryContext(vxContext, VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT, &context, sizeof(context)));
        data->workspace_size = (data->workspace_size + 3) & ~3;
        data->workspace = clCreateBuffer(context, CL_MEM_READ_WRITE, data->workspace_size, NULL, NULL);
        if (!data->workspace) {
            return VX_FAILURE;
        }
        cl_float pattern= 0;
        cl_int err = clEnqueueFillBuffer(data->handle->cmdq, data->workspace, &pattern, sizeof(cl_float), 0, data->workspace_size, 0, NULL, NULL);
        if(err) return VX_FAILURE;
    }

    //create buffers.
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->weights_mem, sizeof(data->weights_mem)));
    if (parameters[2]) ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[2], VX_TENSOR_BUFFER_OPENCL, &data->hidden_input_mem, sizeof(data->hidden_input_mem)));
    if (parameters[3]) ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor) parameters[3], VX_TENSOR_BUFFER_OPENCL, &data->bias_mem, sizeof(data->bias_mem)));

    //add to node attribute.
    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeRNNLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    RNNLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if(data->workspace && clReleaseMemObject(data->workspace) != 0) return VX_FAILURE;
    for (int i=0; i < data->input_desc_array.size(); i++) {
        ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->input_desc_array[i]));
    }
    ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->hidden_input_desc));
    for (int i=0; i < data->output_desc_array.size(); i++) {
        ERROR_CHECK_MIOPEN_STATUS(miopenDestroyTensorDescriptor(data->output_desc_array[i]));
    }

    if (data) {
        ERROR_CHECK_STATUS(releaseGraphHandle(node, data->handle));
    }

    return VX_SUCCESS;
}

vx_status publishRNNLayer(vx_context context)
{
    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.rnn_layer", VX_KERNEL_RNN_LAYER_AMD, processRNNLayer, 7, validateRNNLayer, initializeRNNLayer, uninitializeRNNLayer);
    ERROR_CHECK_OBJECT(kernel);

    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));

    //set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    //finialize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxRNNLayer(vx_graph graph, vx_tensor input, vx_tensor weights, vx_tensor hidden, vx_tensor bias, vx_size mode, vx_size sequence_length, vx_tensor output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference) graph);
    if (vxGetStatus((vx_reference) context) == VX_SUCCESS) {
        vx_scalar seq_len = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &sequence_length, sizeof(sequence_length));
        vx_scalar rnn_mode = vxCreateScalarWithSize(context, VX_TYPE_INT32, &mode, sizeof(mode));
        if (vxGetStatus((vx_reference) seq_len) == VX_SUCCESS && (vxGetStatus((vx_reference) rnn_mode) == VX_SUCCESS)) {
            vx_reference params[] = {
                (vx_reference) input,
                (vx_reference) weights,
                (vx_reference) hidden,
                (vx_reference) bias,
                (vx_reference) rnn_mode,
                (vx_reference) seq_len,
                (vx_reference) output
            };
            node = createNode(graph, VX_KERNEL_RNN_LAYER_AMD, params, sizeof(params)/sizeof(params[0]));
            vxReleaseScalar(&seq_len);
            vxReleaseScalar(&rnn_mode);
        }
    }
    return node;
}
