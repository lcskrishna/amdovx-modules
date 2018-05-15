# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os, sys
import numpy as np
import argparse
from google.protobuf import text_format
import collections
import struct
from nnir import *
import math

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import tensor_util

def tf_name_to_ir_name(name):
    return '_'.join(('_'.join(name.split('/')).split('-')))

def tf_tensor_to_ir_tensor(tensor_name, tensor_data_type, tensor_shape, layout):
    tensor = IrTensor()
    tensor.setName(tf_name_to_ir_name(tensor_name))
    tensor.setInfo(tensor_data_type, [int(x) for x in tensor_shape])
    tensor.setFormat(layout)
    return tensor

def extractInput(graph_def, inputs, verbose, graph, dims):
    
    input_info = collections.OrderedDict()
    layers = graph_def.node
    input_dims = []
    tensor_data_type = "F032"
    isInputFound = False
    if dims is not None:
        input_dims = dims.split(",")
        if (verbose):
            print ("Input dims added are : " + str(input_dims))
    for i in range(len(layers)):
        node = layers[i]
        if (node.name == inputs):
            attr_info = node.attr['value']
            tensor_proto = attr_info.tensor
            tensor_shape = tensor_proto.tensor_shape
            if (len(tensor_shape.dim) > 0 and (dims is None)):
                for j in range(len(tensor_shape.dim)):
                    input_dims.append(int(tensor_shape.dim[j].size))
            elif (input_dims != []):
                isInputFound = True
                break
            else:
                print ("ERROR: unable to access the input dims, please add the input dims manually using --input-dims parameter.")
                sys.exit(1)
            isInputFound = True
            break
    
    if not isInputFound:
        print ("ERROR: unable to find the input name given in the graph.")
        sys.exit(1)

    input_name = tf_name_to_ir_name(inputs)
    input_info[str(input_name)] = input_dims
    graph.addInput(tf_tensor_to_ir_tensor(input_name, tensor_data_type, input_dims, "NCHW"))

    if (verbose):
        print ("OK: extracted input information from graph")    
    return input_info

def extractBinary(graph_def, verbose, graph):
    layers = graph_def.node
    weight_map_info = collections.OrderedDict()
    for i in range(len(layers)):
        node = layers[i]
        if (node.op == "Const"):
            if "value" in node.attr:
                tensor_name = str(tf_name_to_ir_name(node.name))
                attr_info = node.attr["value"]
                graph.addBinary(tensor_name, attr_info.tensor.tensor_content)
                dims = attr_info.tensor.tensor_shape.dim
                weight_dims = []
                for j in range(len(dims)):
                    weight_dims.append(int(dims[j].size))
                weight_map_info[tensor_name] = weight_dims
                graph.addVariable(tf_tensor_to_ir_tensor(tensor_name, "F032", weight_dims, "HWCN"))

    if (verbose):
        #print (weight_map_info)
        print ("OK: done extracting weights")
    
    return weight_map_info                                
    
def extractTFAttrInfo(node, input_info_map, weight_info_map, identityMapAliasList):
    layer_type = str(node.op)
    attribute_map = {}
    if (layer_type == "Conv2D"):
        inputs = node.input
        input_weight_name = tf_name_to_ir_name(str(inputs[1]))
        weight_name = identityMapAliasList[input_weight_name] if (input_weight_name in identityMapAliasList) else input_weight_name
        kernel_h, kernel_w, _ , _  = weight_info_map[weight_name]
        strides_info = node.attr["strides"].list.i
        dilations = []
        
        #extract data format 
        data_format = "NCHW"
        if ("data_format" in node.attr):
            data_format = node.attr["data_format"].s
        
        if "dilations" in node.attr:
            dilations  = node.attr["dilations"].list.i
        else:
            dilations = [1,1]
        
        ## strides info.
        strides = []
        if (data_format == "NCHW"):
            strides = [int(strides_info[2]), int(strides_info[3])]
            attribute_map["strides"] = [int(strides_info[2]), int(strides_info[3])]
        elif (data_format == "NHWC"):
            strides = [int(strides_info[1]), int(strides_info[2])]
            attribute_map["strides"] = [int(strides_info[1]), int(strides_info[2])]
        else:
            print ("ERROR: unsupported data format " + data_format)
            sys.exit(1)

        attribute_map["kernel_shape"] = [int(kernel_w), int(kernel_h)]
        attribute_map["dilations"] = [int(dilations[0]), int(dilations[1])]
        
        pads = [0,0]
        inputs = input_info_map.keys()
        n,c,h,w = 0,0,0,0
        if (data_format == "NCHW"):
            n,c,h,w = input_info_map[inputs[0]]
        else:
            n,h,w,c = input_info_map[inputs[0]]

        if (h % (strides[1]) == 0):
            pads[1] = max(kernel_h - strides[1], 0)
        else:
            pads[1] = max(kernel_h - (h % strides[1]), 0)

        if (w % strides[0] == 0):
            pads[0] = max(kernel_w - strides[0], 0)
        else:
            pads[0] = max(kernel_w - (w % strides[0]), 0)
        
        pad_top = pads[1] // 2
        pad_bottom = pads[1] - pad_top
        pad_left = pads[0] // 2
        pad_right = pads[0] - pad_left

        attribute_map["pads"] = [pad_left, pad_top, pad_right, pad_bottom]
        
    elif (layer_type == "MaxPool"):
        kernel_size = node.attr["ksize"].list.i
        strides = node.attr["strides"].list.i

        data_format = "NCHW"
        if ("data_format" in node.attr):
            data_format = node.attr['data_format'].s

        if (data_format == "NCHW"):
            attribute_map["kernel_shape"] = [int(kernel_size[2]), int(kernel_size[3])]
            attribute_map["strides"] = [int(strides[2]), int(strides[3])]
        elif (data_format == "NHWC"):
            attribute_map["kernel_shape"] = [int(kernel_size[1]), int(kernel_size[2])]
            attribute_map["strides"] = [int(strides[1]), int(strides[2])]
        else:
            print ("ERROR: unsupported data format for pooling")
            sys.exit(1)

        pads = [0,0]
        inputs = input_info_map.keys()
        n,c,h,w = 0,0,0,0
        if (h % (strides[1]) == 0):
            pads[1] = max(kernel_size[1] - strides[1], 0)
        else:
            pads[1] = max(kernel_size[1] - (kernel_size[1] % strides[1]), 0)

        if (w % strides[0] == 0):
            pads[0] = max(kernel_size[0] - strides[0], 0)
        else:
            pads[0] = max(kernel_size[0] - (kernel_size[0] % strides[0]), 0)

        pad_top = pads[1] // 2
        pad_bottom = pads[1] - pad_top
        pad_left = pads[0] // 2
        pad_right = pads[0] - pad_left

        attribute_map["pads"] = [pad_left, pad_top, pad_right, pad_bottom]

    elif (layer_type == "LRN"):
        alpha = node.attr["alpha"].f
        beta = node.attr["beta"].f
        bias = node.attr["bias"].f
        local_size = node.attr["depth_radius"].i

        attribute_map["alpha"] = alpha
        attribute_map["beta"] = beta
        attribute_map["bias"] = bias
        attribute_map["size"] = int(local_size)
        
    return attribute_map

def calculateTensorDims(node, input_info_map, weights_info_map, identityMapAliasList , attribute_map):
    dimList = {}
    output_dims = [0, 0, 0, 0]
    inputs = input_info_map.keys()
    layer_type = str(node.op)
    
    if (layer_type == "Conv2D"):
        strides = attribute_map["strides"]
        dilations = attribute_map["dilations"]
        kernel_shape = attribute_map["kernel_shape"]
        data_format = "NCHW"
        if ("data_format" in node.attr):
            data_format = node.attr["data_format"].s
        n,c,h,w = 0,0,0,0
        if (data_format == "NCHW"):
            n,c,h,w = input_info_map[inputs[0]]
        else:
            n,h,w,c = input_info_map[inputs[0]]

        weights_name = tf_name_to_ir_name(str(node.input[1]))
        if (weights_name in identityMapAliasList):
            weights_name = identityMapAliasList[weights_name]

        weight_dims = weights_info_map[weights_name]

        output_height = h
        output_width = w
        if (node.attr["padding"].s == "SAME"):
            print ("INFO: output dimensions for same padding is calculated")
            output_height = math.ceil(float(h) / float(strides[1]))
            output_width = math.ceil(float(w) / float(strides[0]))
        elif (node.attr["padding"].s == "VALID"):
            print ("INFO: output dimensions for valid padding is calculated.")
            output_height = math.ceil(float(h - kernel_shape[1] + 1 )/float(strides[1]))
            output_width = math.ceil(float(w - kernel_shape[0] + 1)/ float(strides[0]))
        else:
            print ("ERROR: unsupported padding format for calculating conv output dimensions")
            sys.exit(1)
   
        if (data_format == "NCHW"):
            output_dims[0] = n
            output_dims[1] = weight_dims[3]
            output_dims[2] = int(output_height)
            output_dims[3] = int(output_width)
        elif (data_format == "NHWC"):
            output_dims[0] = n
            output_dims[1] = int(output_height)
            output_dims[2] = int(output_width)
            output_dims[3] = weight_dims[3]
        else:
            print ("ERROR: unsupported data format in convolution")
            sys.exit(1)
    
    elif (layer_type == "MaxPool"):
        data_format = "NCHW"
        strides = attribute_map["strides"]
        kernel_shape = attribute_map["kernel_shape"]        
        if ("data_format" in node.attr):
            data_format = node.attr["data_format"].s
        
        n,c,h,w = 0,0,0,0
        if (data_format == "NCHW"):
            n,c,h,w = input_info_map[inputs[0]]
        else:
            n,h,w,c = input_info_map[inputs[0]]

        output_height = h
        output_width = w
        if (node.attr["padding"].s == "SAME"):
            output_height = math.ceil(float(h)/ float(strides[1]))
            output_width = math.ceil(float(w)/float(strides[0]))
        elif (node.attr["padding"].s == "VALID"):
            output_height = math.ceil(float(h - kernel_shape[1] + 1) / float(strides[1]))
            output_width = math.ceil(float(w - kernel_shape[0] + 1)/ float(strides[0]))
        else:
            print ("ERROR: unsupported padding format for calculating the pooling output dimensions")
            sys.exit(1)

        if (data_format == "NCHW"):
            output_dims[0] = n
            output_dims[1] = c
            output_dims[2]= int(output_height)
            output_dims[3] = int(output_width)
        elif (data_format == "NHWC"):
            output_dims[0] = n
            output_dims[1] = int(output_height)
            output_dims[2] = int(output_width)
            output_dims[3] = c
        else:
            print ("ERROR: unsupported data format in pooling : " + data_format )
            sys.exit(1)
    else:
        print ("Layer type is :::::" + layer_type)
        output_dims = input_info_map[inputs[0]]

    dimList["outputs"] = output_dims
    return dimList


def extractTFNodeInfo(graph_def, input_names, output_names, graph_input_info, weights_info_map, verbose, graph):

    layers = graph_def.node
    inputs_tf = input_names.split(",")
    outputs_tf = output_names.split(",")

    count = 0
    identityMapAliasList = {}
    inputOutputMap = collections.OrderedDict()
    inputsMap = {}
    outputsMap = {}
    for i in range(len(layers)):
        node = layers[i]
        layer_info_map = {}
        
        layer_name = tf_name_to_ir_name(str(node.name))
        layer_type = str(node.op)
        
        if (node.op == "RandomShuffleQueueV2" or node.op == "QueueDequeueManyV2" or node.op == "Const"):
            continue

        if (node.op == "Identity"):
            output_name = str(tf_name_to_ir_name(node.name))
            input_name = str(tf_name_to_ir_name(node.input[0]))
            identityMapAliasList[output_name] = input_name  
            continue

        input_info_map = collections.OrderedDict()
        output_info_map = collections.OrderedDict()

        layer_info_map["layer_name"] = layer_name
        layer_info_map["layer_type"] = layer_type

        ## extract inputs to the layers.
        inputs = node.input
        if (len(inputs) > 0):
            if (count == 0):
                in_name = tf_name_to_ir_name(str(inputs[0]))
                if (in_name in identityMapAliasList):
                    in_name = identityMapAliasList[in_name]
                input_info_map[in_name] = graph_input_info[in_name]
            else:
                if (layer_type != "ConcatV2" or layer_type != "Add" or layer_type != "Sub"):
                    in_name = tf_name_to_ir_name(str(inputs[0]))
                    if (in_name in identityMapAliasList):
                        in_name = identityMapAliasList[in_name]
                    if (in_name in outputsMap):
                        input_info_map[in_name] = outputsMap[in_name]
                    elif (in_name in weights_info_map):
                        input_info_map[in_name] = weights_info_map[in_name]
                    else:
                        print ("ERROR: unable to extract input dims for : " + in_name)
                        sys.exit(1) 
                else:
                    for j in range(len(inputs)):
                        in_name = tf_name_to_ir_name(str(inputs[j]))
                        if (in_name in identityMapAliasList):
                            in_name = identityMapAliasList[in_name]
                        if (in_name in outputsMap):
                            input_info_map[in_name] = outputsMap[in_name]
                        elif (in_name in weights_info_map):
                            input_info_map[in_name] = weights_info_map[in_name]
                        else:
                            print ("ERROR: unable to extract input dims for : " + in_name)
                            sys.exit(1)
        if (input_info_map):            
            inputsMap.update(input_info_map)

        ## add inputs to the layer info.
        layer_info_map["inputs"] = input_info_map

        ## extract TF attributes information.
        attribute_map = extractTFAttrInfo(node, input_info_map, weights_info_map, identityMapAliasList)
        layer_info_map["attributes"] = attribute_map

        ## calculate output tensor dimensions.        
        dimList = calculateTensorDims(node, input_info_map, weights_info_map, identityMapAliasList, attribute_map)

        ## add output info to layer info.
        output_info_map[layer_name] = dimList["outputs"]
        outputsMap[layer_name] = dimList["outputs"]
        layer_info_map["outputs"] = output_info_map
        
        inputOutputMap[count] = layer_info_map

        if (verbose):
            print (" ============================= " + str(count) + " =========================")
            print (layer_info_map)

        count += 1

    print ("Total usable layers :  " + str(count))

def tf_graph_to_ir_graph(graph_def, inputs, outputs, verbose, dims):
    graph = IrGraph()
    input_info = extractInput(graph_def, inputs, verbose, graph, dims)
    weight_map_info = extractBinary(graph_def, verbose, graph)
    extractTFNodeInfo(graph_def, inputs, outputs, input_info, weight_map_info, verbose, graph)
    return graph

def tf2ir(graph_def, inputs, outputs, outputFolder, verbose, dims):
    graph = tf_graph_to_ir_graph(graph_def, inputs, outputs, verbose, dims)
    graph.updateLocals()
    graph.toFile(outputFolder)
    print ("OK: graph successfully formed.")

def main():

    verbose = args.verbose      
    outputFolder = args.nnir_folder
    if not os.path.isfile(args.pb_file):
        print ("ERROR: unable to open : " + args.pb_file)
        sys.exit(1)
    
    if ('.pb' not in args.pb_file):
        print ("ERROR: unsupported file format, currently this tool supports frozen pb file")
        sys.exit(1)  

    graph_def = graph_pb2.GraphDef()
    graph_def.ParseFromString(open(args.pb_file, "rb").read())
    print ("OK: tensorflow model read successful.")
    print ("converting to AMD NNIR format in %s folder ... " % (outputFolder))

    tf2ir(graph_def, args.inputs, args.outputs, outputFolder, verbose, args.input_dims)

if __name__ == '__main__':

    if len(sys.argv) < 4:
        print ("Usage: python tf2nnir.py --pb-file <tensorflow-pb-file> --nnir-folder <nnirOutputFolder> --inputs <input layer name> --outputs <output layer names> [--verbose 0|1]")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pb-file', type = str, required = True, help = 'Frozen TF pb file')
    parser.add_argument('--nnir-folder', type = str, required = True, help = 'NNIR output folder')
    parser.add_argument('--inputs', type = str, required = True, help = 'input name of the graph.')
    parser.add_argument('--outputs', type = str, required = True, help = 'comma seperated output names to the graph')
    parser.add_argument('--input-dims', type = str, required = False, help = 'input dims to the graph')
    parser.add_argument('--verbose', type = int, required = False, default = 0, help = 'verbose enable/disable')
    args = parser.parse_args()
    main()
