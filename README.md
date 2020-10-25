# Вебсервис

Сначала запускается Remote manager, потом вебсервис.
Машины с Remote manager и вебсервисом должны пинговаться между собой 
(например, они могут быть связаны с помощью VPN)

## Запуск Remote manager

Remote manager запускается на машине с видеокартой

В 64 строке в файле remote_manager.py заменить порт и auth_key на свои

```python
manager = BaseManager(address=('0.0.0.0', <port>),
                      authkey='<authkey>'.encode('utf-8'))
```

Также необходимо указать путь до весов нейросети в строке 34. 
Веса нейронной сети расположены на [Яндекс диске](https://yadi.sk/d/W60C2mkM3u-1Kw)

```python
path = 'model_0.677.pth'
```

Запуск

```bash
pip3 install -r remote_manager_requirements.txt
python3 remote_manager.py
```


## Запуск вебсервиса:


В файле app.py заменить в 20 строчке ip, порт, auth_key на свои. Ip должен быть тот, на котором сидит Remote manager. 
Порт и auth_key должны совпадать с портом и auth_key в файле remote_manager.py:

```python
manager = BaseManager(address=(<ip to remote server>, <port>), authkey=b'<authkey>')
```

Запуск

```bash
pip3 install -r requirements.txt
python3 app.py
```

