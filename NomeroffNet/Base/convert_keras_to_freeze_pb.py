import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
from keras import backend as K

def convert_keras_to_freeze_pb(model, frozen_model_path):
        out_names = ",".join([layer.name.split(":")[0]  for layer in model.outputs])
        inp_names = ",".join([layer.name.split(":")[0]  for layer in model.inputs])
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