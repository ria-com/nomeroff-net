Nomeroff Net annotation tool
==========================

Набор утилит для извличения номеров, которые оформлены в стиле сайта [Avto-nomer.ru](Avto-nomer.ru), 
а иммено, в имени файла (до расширения) содержится текст распознаного номерного знака автомобиля, который на фотграфии 
расположен на переднем плане. 

## Установка
Установите последнюю версию nodejs, загрузите модули для работы:
```bash
yum install nodejs
cd tools/js
npm install
```



## Запуск утилит

#### Python-скрипт извлечения номеров
Это черновой набросок кода, который будет извлекать области похожие на номерные знаки для нальнейшего создания к ним 
аннотаций и разметке в [админке](https://github.com/ria-com/nomeroff-net/tree/master/moderation)  


#### Создание OCR-датасета Nomeroff Net
Эта команда позволяет создать аннотации к номерам
```bash
./console.js --section=default --action=createAnnotations  --opt.baseDir=../../datasets/ocr/kz/kz2
```

#### Перенести в одельную папку из OCR-датасета промодеированные данные
```bash
./console.js --section=default --action=moveChecked  --opt.srcDir=../../datasets/ocr/kz/draft --opt.targetDir=../../datasets/ocr/kz/checked  
```

#### Перенести в одельную папку из OCR-датасета дубликаты (определенные по фото)
```bash
./console.js --section=default --action=moveDupes --opt.srcDir=../../datasets/ocr/train --opt.targetDir=../../datasets/ocr/dupes  
```

#### Поделить датасет на 2 части в заданой пропорции
Перед разделением данные будут рамдомно перемешаны
```bash
./console.js --section=default --action=dataSplit --opt.rate=0.2  --opt.srcDir=../../datasets/ocr/draft --opt.targetDir=../../datasets/ocr/test  
```

#### Склеить 2 датасета в один
Перед разделением данные будут рамдомно перемешаны
```bash
./console.js --section=default --action=dataJoin --opt.srcDir=/var/www/html2/js/nomeroff-net_2/datasets/ocr/ge2/ge  --opt.srcDir=/var/www/html2/js/nomeroff-net_2/datasets/ocr/ge2/ge.ok     --opt.targetDir=--opt.srcDir=/var/www/html2/js/nomeroff-net_2/datasets/ocr/ge2/target  
```




#### Поделить датасет для Mask RCNN на 2 части в заданой пропорции
```bash
./console.js --section=via --action=split --opr.rate=0.2 --opt.srcDir=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-07/draft --opt.targetDir=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-07 --opt.viaFile=via_region_data.json```
```

#### Объеденить 2 датасета в один для Mask RCNN
```bash
./console.js --section=via --action=joinVia --opt.srcJson=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-11/src1/via_data_ria_1_full.json --opt.srcJson=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-11/src2/via_data_ria2.json --opt.targetDir=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-11/target --opt.viaFile=via_region_data.json
```


