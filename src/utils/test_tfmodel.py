import tensorflow as tf
from pytorch2onnx import AntiSpoofPredict
import cv2
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
import time

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


if __name__=="__main__":

    #load pytorch
    device_id = 0
    model_path = "./resources/anti_spoof_models/2020-09-19-04-15_Anti_Spoofing_1.2_112x112_model_iter-399.pth"
    anti_model = AntiSpoofPredict(device_id, model_path)

    dummy_img = cv2.imread("/home/dmp/Silent-Face-Anti-Spoofing/datasets/RGB_Images/1.2_112x112/test_caffee_model/0/1599816415801_18.png")
    # dummy_output = anti_model.predict(dummy_img)
    # print("dummy_output_pytorch", dummy_output)

    tf_graph = load_pb('/home/dmp/Silent-Face-Anti-Spoofing/resources/converted_models/tfmodel.pb')
    sess = tf.Session(graph=tf_graph, config=config)

    # Show tensor names in graph
    # for op in tf_graph.get_operations():
    #     print(op.values())

    output_tensor = tf_graph.get_tensor_by_name('test_output:0')
    input_tensor = tf_graph.get_tensor_by_name('test_input:0')
    test_speed = 0
    for _ in range(1000):
        start = time.time()
        dummy_input = anti_model.transform_input(dummy_img)
        output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input})
        print(time.time()-start)
        print("dummy_output_tf", output)
        test_speed += time.time()-start
    print("test_speed", test_speed/1000)