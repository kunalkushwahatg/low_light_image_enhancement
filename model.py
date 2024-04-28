import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/LOL_model.tflite"

interpreter = tf.lite.Interpreter(model_path = MODEL_PATH)
print("model loaded sucessfully")


def infer(interpreter,image):

    #it is the max lenght or width that image can have 
    threshold_size = 600 
    print(image.size)

    if image.size[0] >threshold_size or image.size[1] >threshold_size:
        #resize image by maintaining length to breadth ratio
        length , height = image.size
        if length >= height :
            new_height = int((height/length)*threshold_size)
            new_height = new_height-(new_height%100)
            new_length = threshold_size
        else:
            new_length = int((length/height)*threshold_size)
            new_length = new_length -(new_length%100)
            new_height = threshold_size

        image = image.resize((new_length,new_height))
    else:
        pass
        
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    interpreter.resize_tensor_input(0, list(image.shape))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'],image)

    interpreter.invoke()

    output_image = interpreter.get_tensor(output_details[0]['index'])
    output_image = output_image[0] * 255.0
    output_image = output_image.clip(0,255)
    print(output_image.shape)
    #output_image = np.squeeze(output_image,axis=0)
    output_image = Image.fromarray(np.uint8(output_image))
    return output_image

#read image 
input_image = Image.open("images/test6.jpg")

#get output 
output_image = infer(interpreter,input_image)
output_image.show()


