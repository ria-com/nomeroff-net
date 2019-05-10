Avto-nomer annotation tool
==========================

Набор утилит для извличения номеров, которые оформлены в стиле сайта [Avto-nomer.ru](Avto-nomer.ru), 
а иммено, в имени файла (до расширения) содержится текст распознаного номерного знака автомобиля, который на фотграфии 
расположен на переднем плане. 

## Установка
Установите последнюю версию nodejs, загрузите модули для работы:
```bash
yum install nodejs
cd tools/avto-nomer-tool
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


#### Поделить датасет на 2 части в заданой пропорции
Перед разделением двнные будут рамдомно перемешаны
```bash
./console.js --section=default --action=dataSplit --opt.splitRate=0.2  --opt.srcDir=../../datasets/ocr/draft --opt.targetDir=../../datasets/ocr/test  
```
