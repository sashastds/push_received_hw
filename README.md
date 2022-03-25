# MVP тестового задания

## Сборка докер-образа:

```shell
docker build -t best_push_time:dev .
```

## Исходные данные

Заархивированный csv-файл, разделитель “;”
[gdrive](https://drive.google.com/file/d/1vDub1LgsFKT7qp8EW3xz9JueqTLBjmm5/view?usp=sharing)

* user_id, content_id - идентификаторы пользователя и контента  
* push_opened - был ли совершён переход по пушу в приложение
* push_time - время когда пуш был отправлен
* сreate_at - время, когда контент был загружен в систему
* content_type - тип контента

## Факторы модели: 

`push_hour` и агрегаты по взаимодействию пользователя с пушами за предыдущий календарный день.

## Тетрадка с исследованием: 

`Nalitkin-AS Push Task EDA.ipynb`

## Контейнер

* обучает модель на данных из `/data` и сохраняет её в ту же примонтированную к контейнеру директорию 
* выполняет предикт лучшего времени пуша на следующий день после максимально доступного для всех известных по данным `user_id` как целое число из диапазона [0, 23]
* сохраняет файл с предиктом по пути `PATH_TO_OUTPUT_FILE`, по умолчанию в примонтированную `/data`

## Вид результирующего файла:

| user_id | best_push_hour |


## Запуск контейнера
Директорию с данными необходимо монтировать в контейнер с помощью ключа -v.
Обучение из корня проекта с созданной директорией `/data/` внутри него запускается, например, следующей командой

на `windows`

```shell
docker run -v %cd%\data:/data -e PATH_TO_DATA_FILE="data/data.gz" -e PATH_TO_OUTPUT_FILE="data/predictions.csv" -e MODEL_DIRECTORY="data/" best_push_time:dev python src/train.py 
```

или на `linux`

```shell
docker run -v {pwd}/data:/data -e PATH_TO_DATA_FILE="data/data.gz" -e PATH_TO_OUTPUT_FILE="data/predictions.csv" -e MODEL_DIRECTORY="data/" best_push_time:dev python src/train.py 
```
