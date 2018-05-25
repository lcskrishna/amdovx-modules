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

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

'''
TF2IRUtil contains the utility classes that are required to convert from TF to NNIR format.
'''
class TF2IRUtil:
    # converts TF name to IR name.
    def tf_name_to_ir_name(self, name):
        return '_'.join(('_'.join(name.split("/")).split("-")))

    # TF tensor to IR tensor.
    def tf_tensor_to_ir_tensor(self, tensor_name, tensor_data_type, tensor_shape, layout):
        tensor = IrTensor()
        tensor.setName(self.tf_name_to_ir_name(tensor_name))
        tensor.setInfo(tensor_data_type, [int(x) for x in tensor_shape])
        tensor.setFormat(layout)
        return tensor

'''
This class is an object class that contains information of all the layers that are extracted from TF model.
'''
class LayerInfo:
    def __init__(self):
        self.layer_name = "Unknown"
        self.layer_type = ""
        self.inputs = collections.OrderedDict()
        self.outputs = collections.OrderedDict()
        self.weights = {}
        self.biases = {}
        self.attributes = {}

    def set_layer_name(self, name):
        self.layer_name = name

    def set_layer_type(self, layer_type):
        self.layer_type = layer_type

    def set_inputs(self, input_map):
        self.inputs.update(input_map)

    def set_outputs(self, output_map):
        self.outputs.update(output_map)

    def set_attributes(self, attribute_map):
        self.attributes.update(attribute_map)

    def set_weights(self, weights_map):
        self.weights.update(weights_map)

    def set_biases(self, bias_map):
        self.biases.update(bias_map)

    def print_info(self):
        return str(self.layer_name) + " | " + str(self.layer_type) + " | " + str(self.inputs) + " | " + str(self.outputs) + " | " + str(self.weights) + " | " + str(self.biases) + " | " + str(self.attributes)

class TFNodeInfoExtractor:

    skip_type = set([
     "L2Loss",
     "VariableV2",
     "Const",
     "Assign",
     "RandomUniform",
     "FIFOQueueV2",
     "Assert",
     "Unpack",
     "NextIteration",
     "TensorArrayV3",
     "Range",
     "TensorArrayScatterV3",
     "TensorArrayReadV3",
     "TensorArrayWriteV3",
     "Dequantize",
     "ExpandDims",
     "Placeholder",
     "Pack",
     "Squeeze"
    ])

    def __init__(self):
        self.tf2ir_util = TF2IRUtil()

    # extraction of input from TF graph.
    def extractInput(self, graph_def, inputs, verbose, graph, dims):
        input_info = collections.OrderedDict()
        layers = graph_def.node
        input_dims = []
        tensor_data_type = "F032"
        isInputFound = False
        if dims != []:
            input_dims = dims
            if (verbose):
                print ("OK: input dims added are : " + str(input_dims))
        
        for i in range(len(layers)):
            node = layers[i]
            if (node.name == inputs):
                if (node.op == "Placeholder"):
                    if ("shape" in node.attr):
                        input_shape_dim = node.attr["shape"].shape.dim
                        for j in range(len(input_shape_dim)):
                            input_dims.append(int(input_shape_dim[j].size))
                elif (input_dims != []):
                    isInputFound = True
                    break
                else:
                    if ("value" in node.attr):
                        attr_info = node.attr["value"]
                        tensor_proto = attr_info.tensor
                        tensor_shape = tensor_proto.tensor_shape
                        
                        if (len(tensor_shape.dim) > 0):
                            for j in range(len(tensor_shape.dim[j].size)):
                                input_dims.append(int(tensor_shape.dim[j].size))
                    else:
                        print ("ERROR: unable to access input dims, please set input dims manually using --input-dims param.")
                        sys.exit(1)
                isInputFound = True
                break

        if not isInputFound:
            print ("ERROR: unable to find the input names specified in the model.")
            sys.exit(1)

        ## add to nnir graph.
        input_name = self.tf2ir_util.tf_name_to_ir_name(inputs)
        input_info[str(input_name)] = input_dims
        data_format = "NHWC"
        if (input_dims[1] < 5):
            data_format = "NCHW"
        graph.addInput(self.tf2ir_util.tf_tensor_to_ir_tensor(input_name, tensor_data_type, input_dims, data_format))

        if (verbose):
            print ("OK: extract input information from the graph.")
        
        return input_info

    # extraction of binary data in the graph.
    def extractBinary(self, graph_def, verbose, graph):
        layers = graph_def.node
        weight_info_map = collections.OrderedDict()
        for i in range(len(layers)):
            node = layers[i]
            if (node.op == "Const"):
                if "value" in node.attr:
                    tensor_name = self.tf2ir_util.tf_name_to_ir_name(str(node.name))
                    attr_info = node.attr["value"]
                    graph.addBinary(tensor_name, attr_info.tensor.tensor_content)
                    dims = attr_info.tensor.tensor_shape.dim
                    weight_dims = []
                    for j in range(len(dims)):
                        weight_dims.append(int(dims[j].size))
                    weight_info_map[tensor_name] = weight_dims
                    graph.addVariable(self.tf2ir_util.tf_tensor_to_ir_tensor(tensor_name, "F032", weight_dims, "HWCN"))

        if (verbose):
            print ("OK: done extracting weights")

        return weight_info_map

    # extraction of node information from TF nodes.
    def extractTFNodeInformation(self, graph_def, inputs, graph_input_info, weights_info_map, verbose, graph):
        network_layer_info = collections.OrderedDict()
        inputsMap = {}
        outputsMap = {}
        count = 0
        identityAliasMap = {}
        multiple_input_layer_types = set(["ConcatV2", "Mul", "Add", "Sub"])
        
        layers = graph_def.node
        for i in range(len(layers)):
            node = layers[i]
            layer_info = LayerInfo()
            layer_name = self.tf2ir_util.tf_name_to_ir_name(str(node.name))
            layer_type = str(node.op)
            
            ## if layer type is any of the operators present in skip type ignore them.
            if (node.op in self.skip_type):
                continue

            ## if the layer type is identity, alias the input with with output.
            if (node.op == "Identity"):
                input_name = self.tf2ir_util.tf_name_to_ir_name(str(node.input))
                output_name = self.tf2ir_util.tf_name_to_ir_name(str(node.name))
                identityAliasMap[output_name] = input_name

            ## layer type information.
            layer_info.set_layer_type(str(node.op)) 
            network_layer_info[count] = layer_info
          
            ## extraction of input of each layer.
            input_info_map = collections.OrderedDict()
            if (count == 0):
                input_name = self.tf2ir_util.tf_name_to_ir_name(str(node.input[0]))
                if (input_name in graph_input_info):
                    input_info_map[input_name] = graph_input_info[input_name]
                else:
                    print ("ERROR: unable to extract input dims from the graph.")
                    sys.exit(1)
            else:
                if (node.op not in multiple_input_layer_types):
                    input_name = self.tf2ir_util.tf_name_to_ir_name(str(node.input[0]))
                    input_info_map[input_name] = []
                else:
                    input_names = node.input
                    for j in range (len(input_names)):
                        input_name = self.tf2ir_util.tf_name_to_ir_name(str(input_names[j]))
                        input_info_map[input_name] = []

            layer_info.set_inputs(input_info_map)

            ## extraction of output.
            output_info_map = collections.OrderedDict()
            output_name = layer_name
            output_info_map[output_name] = []
            layer_info.set_outputs(output_info_map)
            
            ## print information for debugging.
            if (verbose):
                print ("Count ------------------------------> " + str(count))
                print ("layer-type: " + str(layer_info.layer_type) + " |inputs: " + str(layer_info.inputs) + " |outputs: " + str(layer_info.outputs))

            count += 1

        return network_layer_info
        
'''
TFFrozenModelUtil is used to load the frozen TF model and removes unnecessary 
training parameters and unused libraries in the network.
'''
class TFFrozenModelUtil:
    def __init__(self):
        self.graph_def = graph_pb2.GraphDef()
        
    def loadModel(self, pb_file_path, inputs, outputs, outputFolder):
        self.graph_def.ParseFromString(open(pb_file_path, "rb").read())
        self.graph_def = tf.graph_util.remove_training_nodes(self.graph_def)
        input_node_names = inputs.split(",")
        output_node_names = outputs.split(",")
        
        #TODO: need decision on this one.
        gdef = strip_unused_lib.strip_unused(
                    input_graph_def = self.graph_def,
                    input_node_names = input_node_names,
                    output_node_names = output_node_names,
                    placeholder_type_enum = dtypes.float32.as_datatype_enum)

        if not os.path.isdir(outputFolder):
            os.mkdir(outputFolder)
        frozen_model_file = outputFolder + "/frozen.pb"
        with gfile.GFile(frozen_model_file, "wb") as f:
            f.write(gdef.SerializeToString())
        f.close()
        print ("OK: tensorflow model read successful.")

def tf_graph_to_ir_graph(graph_def, inputs, input_dims, outputs, verbose):
    graph = IrGraph()
    tf_node_extractor = TFNodeInfoExtractor()
    input_info = tf_node_extractor.extractInput(graph_def, inputs, verbose, graph, input_dims)
    weights_info_map = tf_node_extractor.extractBinary(graph_def, verbose, graph)
    network_layer_info = tf_node_extractor.extractTFNodeInformation(graph_def, inputs, input_info, weights_info_map, verbose, graph)
    return graph
    
def tf2ir(pb_file_path, inputs, input_dims,  outputs, outputFolder, verbose):
    model_util = TFFrozenModelUtil()
    model_util.loadModel(pb_file_path, inputs, outputs, outputFolder)
    tf_graph = model_util.graph_def
    graph = tf_graph_to_ir_graph(tf_graph, inputs, input_dims, outputs, verbose)
    graph.updateLocals()
    graph.toFile(outputFolder) 
    print ("OK: graph successfully formed.")

def main():
    verbose = True if args.verbose else False
    outputFolder = args.nnir_folder
    input_dims = []

    if not os.path.isfile(args.pb_file):
        print ("ERROR: unable to open : " + args.pb_file)
        sys.exit(1)

    if (".pb" not in args.pb_file):
        print ("ERROR: unsupported file format, currently this tool supports frozen pb files")
        sys.exit(1)

    if args.input_dims is not None:
        input_dims = args.input_dims.split(",")

    tf2ir(args.pb_file, args.inputs, input_dims, args.outputs, outputFolder, verbose)
    
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print ("Usage: python tf2nnir.py --pb-file <tensorflow-pb-file>"  + "\n" + \
               "                         --nnir-folder <nnirOutputFolder>" + "\n" + \
               "                         --inputs <input layer names>"  + "\n" + \
               "                         --outputs <output layer names>" + "\n" + \
               "                         [--verbose 0|1] [--input-dims n,c,h,w]")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pb-file', type = str, required = True, help = "Frozen TF pb file")
    parser.add_argument('--nnir-folder', type = str, required = True, help = "NNIR output folder")
    parser.add_argument('--inputs', type = str, required = True, help = "input name of the graph")
    parser.add_argument('--outputs', type = str, required = True, help = "output names of the graph")
    parser.add_argument('--input-dims', type = str, required = False, help = "optional input dims if not present")
    parser.add_argument('--verbose' , type = int, required = False, default = 0, help = "verbose enable/disable")

    args = parser.parse_args()
    main()
