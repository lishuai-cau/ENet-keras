from nets.ENet import enet
model = enet(n_classes=2,input_height=320, input_width=320)
model.load_weights("./logs/ep064-loss0.006-val_loss0.007.h5")
model.save("./weight/ENet320.h5")

import keras2onnx
import onnx
onnx_model = keras2onnx.convert_keras(model, model.name)
temp_model_file = './onnx/Enet.onnx'
onnx.save_model(onnx_model, temp_model_file)