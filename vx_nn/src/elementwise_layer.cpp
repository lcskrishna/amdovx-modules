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

struct ElementwiseLayerLocalData {
    NeuralNetworkCommonHandle * handle;
    miopenTensorOp_t operation;
    double alpha1;
    double alpha2;
    double beta;
    miopenTensorDescriptor_t input1;
    cl_mem input1_mem;
    miopenTensorDescriptor_t input2;
    cl_mem input2_mem;
    miopenTensorDescriptor_t output;
    cl_mem output_mem;
};



static vx_status VX_CALLBACK validateElementwiseLayer(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    //check scalar types.
    vx_enum type;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[2], VX_SCALAR_TYPE, &type, sizeof(type)));
    if (type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[3], VX_SCALAR_TYPE, &type, sizeof(type)));
    if (type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &type, sizeof(type)));
    if (type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;

    //check tensor dims.
    vx_size num_dims;
    vx_size input1_dims[4], input2_dims[4], output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input1_dims, sizeof(input1_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input2_dims, sizeof(input2_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if (num_dims != 4) return VX_ERROR_INVALID_DIMENSION;
    if (type != VX_TYPE_FLOAT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    if ((output_dims[3] != input1_dims[3]) && (input1_dims[3] != input2_dims[3])) return VX_ERROR_INVALID_DIMENSION;
    if ((output_dims[2] != input1_dims[2]) && (input1_dims[2] != input2_dims[2])) return VX_ERROR_INVALID_DIMENSION;
    if ((output_dims[1] != input1_dims[1]) && (input1_dims[1] != input2_dims[1])) return VX_ERROR_INVALID_DIMENSION;
    if ((output_dims[0] != input1_dims[0]) && (input1_dims[0] != input2_dims[0])) return VX_ERROR_INVALID_DIMENSION;

    //output tensor configuration.
    type = VX_TYPE_FLOAT32;
    num_dims = 4;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[5], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[5], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[5], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK processElementwiseLayer(vx_node node, const vx_reference * parameters, vx_uint32 num) {
    ElementwiseLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    miopenHandle_t miopenHandle = data->handle->miopen_handle;

    //miopen elementwise addition call.
    ERROR_CHECK_MIOPEN_STATUS(miopenOpTensor(miopenHandle, data->operation, &data->alpha1, data->input1, data->input1_mem, &data->alpha2, data->input2, data->input2_mem, &data->beta, data->output, data->output_mem));
    clFinish(data->handle->cmdq);

    return VX_SUCCESS;
}

static vx_status VX_CALLBACK initializeElementwiseLayer(vx_node node, const vx_reference *parameters, vx_uint32 num) {

    ElementwiseLayerLocalData * data = new ElementwiseLayerLocalData;
    memset(data, 0, sizeof(*data));
    ERROR_CHECK_STATUS(createGraphHandle(node, &data->handle));

    //initialize input and output tensor descriptors.
    vx_size input1_dims[4], input2_dims[4], output_dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input1_dims, sizeof(input1_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, input2_dims, sizeof(input2_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->input1));
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->input2));
    ERROR_CHECK_MIOPEN_STATUS(miopenCreateTensorDescriptor(&data->output));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->input1, miopenFloat, input1_dims[0], input1_dims[1], input1_dims[2], input1_dims[3]));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->input2, miopenFloat, input2_dims[0], input2_dims[1], input2_dims[2], input2_dims[3]));
    ERROR_CHECK_MIOPEN_STATUS(miopenSet4dTensorDescriptor(data->output, miopenFloat, output_dims[0], output_dims[1], output_dims[2], output_dims[3]));

    //scaling parameters.
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[2], &data->alpha1, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[3], &data->alpha2, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[4], &data->beta, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    data->operation = miopenTensorOpAdd;

    //input and output memory.
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->input1_mem, sizeof(data->input1_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->input2_mem, sizeof(data->input2_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[5], VX_TENSOR_BUFFER_OPENCL, &data->output_mem, sizeof(data->output_mem)));

    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeElementwiseLayer(vx_node node, const vx_reference *parameters, vx_uint32 num) {
    ElementwiseLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data) {
        ERROR_CHECK_STATUS(releaseGraphHandle(node, data->handle));
        delete data;
    }
    return VX_SUCCESS;
}

vx_status publishElementwiseLayer(vx_context context) {

    // add kernel to the context with callbacks
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.elementwise_layer", VX_KERNEL_ELEMENTWISE_LAYER, processElementwiseLayer, 6, validateElementwiseLayer, initializeElementwiseLayer, uninitializeElementwiseLayer);
    ERROR_CHECK_OBJECT(kernel);

    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));

    // set kernel parameters
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));

    // finalize and release kernel object
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxElementwiseLayer(vx_graph graph, vx_tensor input1, vx_tensor input2, vx_float32 alpha1, vx_float32 alpha2, vx_float32 beta, vx_tensor output) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar s_alpha1 = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &alpha1, sizeof(alpha1));
        vx_scalar s_alpha2 = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &alpha2, sizeof(alpha2));
        vx_scalar s_beta = vxCreateScalarWithSize(context, VX_TYPE_FLOAT32, &beta, sizeof(beta));


        if (vxGetStatus((vx_reference)s_alpha1) == VX_SUCCESS &&
                vxGetStatus((vx_reference)s_alpha2) == VX_SUCCESS &&
                vxGetStatus((vx_reference)s_beta) == VX_SUCCESS) {

            vx_reference params[] = {
                (vx_reference)input1,
                (vx_reference)input2,
                (vx_reference)s_alpha1,
                (vx_reference)s_alpha2,
                (vx_reference)s_beta,
                (vx_reference)output
            };
            node = createNode(graph, VX_KERNEL_ELEMENTWISE_LAYER, params, sizeof(params) / sizeof(params[0]));
        }
    }
    return node;

}
