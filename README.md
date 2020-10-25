# Как использовать процедуру обучения ИНС?

Для обучения модели необходимо указать параметры процедуры обучения в файле
`train_confgi.yaml`, после чего необходимо в терминале запустить команду

    ptyhon train.py --c train_config.yaml
    
После запуска этой команды процедура обучения начнется автоматически. Для
отображения графиков обучения можно воспользоваться утилитой `tensorboard`.
Для этого также в терминале нужно набрать команду

    tensorboard --logdir %путь до директории с результатми эксперимента%

после чего перейти в бразуер по адресу `localhost:6006/`. По данному адресу
будет осуществляться логирование метрик и функции потерь.

# Результаты экспериментов.

| Name  | LR    | OPTIMIZER     | SCHEDULER  | BS  | IMAGE\_MODE | AUG                                                                                                            | NORMALIZE | Heuristics                         | VAL\_F1 |
| ----- | ----- | ------------- | ---------- | --- | ----------- | -------------------------------------------------------------------------------------------------------------- | --------- | ---------------------------------- | ------- |
| exp1  | 0.01  | Adam          | None       | 768 | RGB         | \-Resize((224, 224));                                                                                          | None      | None                               | 0.305   |
| exp2  | 0.01  | Adam          | None       | 768 | RGB         | \-Resize((224, 224));                                                                                          | None      | \- Cost-sensitive loss;            | 0.447   |
| exp3  | 0.01  | Adam          | None       | 768 | RGB         | \-Resize((224, 224));                                                                                          | None      | \- Cost-sensitive loss;<br>\- EMA. | 0.484   |
| exp4  | 0.01  | Adam          | None       | 768 | RGB         | \- Resize((224, 224));<br>\- RandomHorizontalFlip(p=0.5);                                                      | None      | \- Cost-sensitive loss;<br>\- EMA. | 0.447   |
| exp5  | 0.01  | Adam          | None       | 768 | RGB         | \- Resize((224, 224));<br>\- RandomAffine(degrees=5, trainslate=(0.1, 0.1);                                    | None      | \- Cost-sensitive loss;<br>\- EMA. | 0.454   |
| exp6  | 0.01  | Adam          | OneCycleLR | 768 | RGB         | \- Resize((224, 224)).                                                                                         | None      | \- Cost-sensitive loss;<br>\- EMA. | 0.428   |
| exp7  | 0.001 | Adam          | OneCycleLR | 768 | RGB         | \- Resize((224, 224)).                                                                                         | None      | \- Cost-sensitive loss;<br>\- EMA. | 0.54    |
| exp8  | 0.001 | Adam          | OneCycleLR | 768 | RGB         | \- Resize((224, 224));<br>\- RandomHorizontalFlip(p=0.5);                                                      | None      | \- Cost-sensitive loss;<br>\- EMA. | 0.547   |
| exp9  | 0.001 | Adam          | OneCycleLR | 768 | RGB         | \- Resize((224, 224));<br>\- RandomHorizontalFlip(p=0.5);<br>\- RandomAffine(degrees=5, translate=(0.1, 0.1)). | None      | \- Cost-sensitive loss;<br>\- EMA. | 0.524   |
| exp10 | 0.001 | Adam, wd=5e-4 | OneCycleLR | 768 | RGB         | \- Resize((224, 224));<br>\- RandomHorizontalFlip(p=0.5).                                                      | None      | \- Cost-sensitive loss;<br>\- EMA. | 0.545   |
| exp11 | 0.001 | Adam          | OneCycleLR | 768 | RGB         | \- Resize((224, 224));<br>\- RandomHorizontalFlip(p=0.5).                                                      | None      | \- Cost-sensitive loss;<br>\- EMA. | 0.61    |
| exp12 | 0.001 | RAdam          | OneCycleLR | 768 | RGB         | \- Resize((224, 224));<br>\- RandomHorizontalFlip(p=0.5).                                                      | None      | \- Cost-sensitive loss;<br>\- EMA. | 0.619    |
| exp13 | 0.001 | Ranger          | OneCycleLR | 768 | RGB         | \- Resize((224, 224));<br>\- RandomHorizontalFlip(p=0.5).                                                      | None      | \- Cost-sensitive loss;<br>\- EMA. | 0.627    |
| exp14 | 0.001 | Ranger, Clean dataset| OneCycleLR | 768 | RGB         | \- Resize((224, 224));<br>\- RandomHorizontalFlip(p=0.5).                                                      | None      | \- Cost-sensitive loss;<br>\- EMA. | 0.657|
| exp15 | 0.001 | Ranger, Clean dataset| OneCycleLR | 768 | RGB         | \- Resize((224, 224));<br>\- RandomHorizontalFlip(p=0.5);<br>\- ColorJitter(0.1, 0.1, 0.1, 0.1);                                                      | None      | \- Cost-sensitive loss;<br>\- EMA. | 0.677|

