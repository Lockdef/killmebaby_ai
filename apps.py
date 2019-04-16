import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import model_from_json
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
import tensorflow as tf
import os
import io
import base64
import urllib

graph = tf.get_default_graph()

app = Flask(__name__)

classes = [u'あぎり',u'ぼつ',u'ソーニャ',u'やすな']
def predict(x):
    model = model_from_json(open('killmebabyai_cnn.json').read())
    model.load_weights('weights.14-0.34-0.88-0.35-0.91.hdf5')
    x = np.array(Image.open(x).convert('RGB').resize((128, 128)))
    x = x.astype('float') / 255
    x = np.array([x])
    r = model.predict(x)

    plot_data = """あぎり  : {:}%
                    ボツ    : {:}%
                    ソーニャ: {:}%
                    やすな  : {:}%""".format(int(r[0][0]*100), int(r[0][1]*100),
                    int(r[0][2]*100), int(r[0][3]*100)).split('\n')

    r = classes[r.tolist()[0].index(max(r[0]))]
    return r, plot_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['POST'])
def send():
    global graph
    if request.files['img_file']:
        with graph.as_default():
            img_file = request.files['img_file'].stream
            class_label, plot_data = predict(img_file)
            image = Image.open(img_file)
            buf = io.BytesIO()
            image.save(buf, 'png')
            qr_b64str = base64.b64encode(buf.getvalue()).decode("utf-8")
            qr_b64data = "data:image/png;base64,{}".format(qr_b64str)
            return render_template('result.html', result=class_label, img=qr_b64data, plot=plot_data)

if __name__ == '__main__':
    #port = int(os.environ.get('PORT'))
    app.run(host='0.0.0.0', port=8888)