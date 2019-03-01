import tensorflow as tf
import keras
from tensorflow.core.framework import graph_pb2
import numpy as np
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
from keras import backend as K
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes

class GraphDefParser():
    def __init__(self, graph):
        self.graph = graph
        self.nodes = self.graph.node
        self.nodes_dict = {}
        for i, node in enumerate(self.nodes):
            self.nodes_dict[node.name] = i

    def if_delete(self, current_node, nodes_names_for_deleting):
        delete = False
        for nodes_name_for_del in nodes_names_for_deleting:
            if -1 != current_node.find(nodes_name_for_del):
                delete = True
        return delete

    def delete_nodes(self, outputs, nodes_names_for_deleting):
        self._delete_nodes(outputs, nodes_names_for_deleting)
        i = 0
        while i < len(self.nodes):
            current_node = self.nodes[i].name
            if self.if_delete(current_node, nodes_names_for_deleting):
                del self.nodes[i]
            else:
                i += 1
        return self.nodes

    def _delete_nodes(self, outputs, nodes_names_for_deleting, perent=None, last_norm = None):
        for output in outputs:
            current_node = self.nodes[self.nodes_dict[output]].name
            current_node_s = self.nodes[self.nodes_dict[output]]
            for i, inp in enumerate(current_node_s.input):
                if inp not in self.nodes_dict.keys():
                    del current_node_s.input[i]
            if last_norm != None:
                last = self.nodes[self.nodes_dict[last_norm]]
                for i, inp in enumerate(last.input):
                    if inp == perent:
                        last.input[i] = current_node
                        print("replace ", perent, " on ", current_node)
            if self.if_delete(current_node, nodes_names_for_deleting):
                last_norm = last_norm
                self._delete_nodes(current_node_s.input, nodes_names_for_deleting, current_node, last_norm)
            else:
                last_norm = current_node
                self._delete_nodes(current_node_s.input, nodes_names_for_deleting, current_node, last_norm)

def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        [print(u'└─── %d ─ %s' % (i, n)) for i, n in enumerate(node.input)]

def delete_training_layers_from_keras(model):
    i = 0
    while i < len(model.layers):
        k = model.layers[i]
        if type(k) is keras.layers.Dropout:
            model.layers.remove(k)
        elif type(k) is keras.layers.BatchNormalization:
            model.layers.remove(k)
        elif  type(k) is keras.engine.training.Model:
            delete_training_layers_from_keras(k)
            i += 1
        else:
            i += 1
    return model

def convert_keras_to_freeze_pb(model, frozen_model_path, optimize = False, delete = []):
    model.trainable = False
    model = delete_training_layers_from_keras(model)
    INPUT_NODE = [layer.name.split(":")[0]  for layer in model.outputs]
    OUTPUT_NODES = [layer.name.split(":")[0]  for layer in model.inputs]
    out_names = ",".join(INPUT_NODE)
    inp_names = ",".join(OUTPUT_NODES)
    print("OUTPUT: {}".format(out_names))
    print("INPUT: {}".format(inp_names))
    model.summary()
    K.set_learning_phase(0)
    sess = K.get_session()
    saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
    checkpoint_path = saver.save(sess, './saved_ckpt', global_step=0, latest_filename='checkpoint_state')

    graph_io.write_graph(sess.graph, '.', './tmp.pb')
    freeze_graph.freeze_graph('./tmp.pb', '',
                            False, checkpoint_path, out_names,
                            "save/restore_all",
                            "save/Const:0",
                            frozen_model_path,
                            False, "")
    # load
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(frozen_model_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    print("FROZEN NODES")
    display_nodes(graph_def.node)

    optimized_graph_def = graph_def

    if bool(delete):
        # delete layers
        graphDefParser = GraphDefParser(optimized_graph_def)
        nodes = graphDefParser.delete_nodes(OUTPUT_NODES, delete)
        print("NODES AFTER DELETING")
        display_nodes(nodes)
        optimized_graph_def = graph_pb2.GraphDef()
        optimized_graph_def.node.extend(nodes)

    if optimize:
        # optimize
        optimized_graph_def = optimize_for_inference(graph_def,
                                   INPUT_NODE,
                                   OUTPUT_NODES,
                                   dtypes.float32.as_datatype_enum)
        print("OPTIMIZED NODES")
        display_nodes(optimized_graph_def.node)

    with tf.gfile.GFile(frozen_model_path, 'w') as f:
        f.write(optimized_graph_def.SerializeToString())