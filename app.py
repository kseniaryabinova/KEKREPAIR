import os
import cv2
import numpy as np

from multiprocessing.managers import BaseManager

from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from werkzeug.utils import secure_filename


UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# подключаемся к remote manager
BaseManager.register('predict')
manager = BaseManager(address=('192.168.6.220', 1448), authkey=b'ksenia')
manager.connect()


# класс для хранения данных о картинке и результатах работы нейросети для этой картинки
class FilesData:
    def __init__(self, filename, output_tensor):
        self.filename = filename
        self.filepath = os.path.join('uploads', filename)
        self.label_name = self.get_label_name(np.argmax(output_tensor))

        top_3_labels_ids = np.argsort(output_tensor)[-3:][::-1]
        self.top_3_labels_list = [(self.get_label_name_normal(label_id),
                                   np.round(output_tensor[label_id]*100, 2  )) for label_id in top_3_labels_ids]
        print(self.top_3_labels_list)

    def get_label_name_normal(self, label):
        if label == 0:
            return 'Без отделки'
        elif label == 1:
            return 'Требуется косметический ремонт'
        elif label == 3:
            return 'Стандартный ремонт'
        elif label == 2:
            return 'Люкс'
        elif label == 4:
            return 'На фото не квартира'
        return '-1'

    def get_label_name(self, label):
        if label == 0:
            return 'Голые стены'
        elif label == 1:
            return 'Бабулин ремонт'
        elif label == 3:
            return 'Евроремонт, все как у людей'
        elif label == 2:
            return 'Золотой унитаз edition'
        elif label == 4:
            return 'Здесь что угодно, но не ремонт'
        return '-1'


# фильтрация файлов
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# отрисовка иконки
@app.route('/kek_icon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static', 'images'),
        'kek_icon.ico',
        mimetype='image/vnd.microsoft.icon')


@app.route('/', methods=['GET', 'POST'])
def index():
    files_data = []

    # если нам пришли файлы
    if request.method == 'POST':
        if 'image' not in request.files:
            print('No file part')
            # return redirect(request.url)

        files = request.files.getlist("image")

        for file in files:
            if file.filename == '':
                print('No selected file')
                # return redirect(request.url)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                # читаем картинку из файла
                frame = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # запускам получаем результаты нейронки по картинке
                output_pred_list = manager.predict(frame)._getvalue()
                # запоминаем пусть до картинки и ее вероятности
                files_data.append(FilesData(file.filename, output_pred_list))

    return render_template('index.html', files_data=files_data)


if __name__ == '__main__':
    app.run(debug=True)
