import os
import cv2
import numpy as np

from multiprocessing.managers import BaseManager

from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from werkzeug.utils import secure_filename


UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

BaseManager.register('predict')
manager = BaseManager(address=('192.168.6.220', 1448), authkey=b'ksenia')
manager.connect()



class FilesData:
    def __init__(self, filename, label_id):
        self.filename = filename
        self.filepath = os.path.join('uploads', filename)
        self.label_name = self.get_label_name(label_id)

    def get_label_name(self, label):
        if label == 0:
            return 'Голые стены'
        elif label == 1:
            return 'Обои в цветочек'
        elif label == 2:
            return 'Евроремонт, все как у людей'
        elif label == 3:
            return 'Золотой унитаз edition'
        return '-1'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/kek_icon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static', 'images'),
        'kek_icon.ico',
        mimetype='image/vnd.microsoft.icon')


@app.route('/', methods=['GET', 'POST'])
def index():
    files_data = []

    if request.method == 'POST':
        if 'image' not in request.files:
            print('No file part')
            return redirect(request.url)

        files = request.files.getlist("image")

        for file in files:
            if file.filename == '':
                print('No selected file')
                return redirect(request.url)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                frame = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                label_id = manager.predict(frame)._getvalue()

                files_data.append(FilesData(file.filename, label_id))

    return render_template('index.html', files_data=files_data)


if __name__ == '__main__':
    app.run(debug=True)
