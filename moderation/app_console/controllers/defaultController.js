const config = require('config'),
      fs = require('fs'),
      path = require('path'),
      jsonStore = require('../../app/helpers/jsonStore'),
      checkDir = require('../../app/helpers/checkDir'),
      checkDirStructure = require('../../app/helpers/checkDirStructure'),
      moveDatasetFiles = require('../../app/helpers/moveDatasetFiles'),
      joinAnnotationOcrDataset = require('../../app/helpers/joinAnnotationOcrDataset'),
      arrayShuffle = require('../../app/helpers/arrayShuffle'),
      sizeOf = require('image-size'),
      md5File = require('md5-file')


;


/**
 * EN: Console script example
 * UA: Приклад консольного скрипту
 * RU: Пример консольного скрипта
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=index
 * @module controllers/defaultController
 */
async function index (options) {
    console.log('Hello world defaultController & index action with options: ' + JSON.stringify(options));
}


/**
 * EN: Creating an OCR dataset Nomeroff Net
 * UA: Створення OCR-датасету Nomeroff Net
 * RU: Создание OCR-датасета Nomeroff Net
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=createAnnotations  \
 *                                               --opt.baseDir=../data/dataset/TextDetector/ocr_example2/train/ \
 *                                               --opt.imageDir=../data/examples/numberplate_zone_images/
 */
async function createAnnotations (options) {
        let baseDir = options.baseDir || config.dataset.baseDir;
        let imageDir = options.imageDir || Error("You must setup imageDir param");
        checkDir(baseDir, true);

        const imgPath = path.join(baseDir, config.dataset.img.dir),
              annPath = path.join(baseDir, config.dataset.ann.dir),
              imgExt = '.'+config.dataset.img.ext,
              tplPath = path.join(config.template.path, config.template.annDefault),
              annTrmplate = require(tplPath);
        checkDir(imgPath, true);
        checkDir(annPath,true);

        console.log("readdir", imageDir);
        fs.readdir(imageDir, async function(err, items) {
                for (let filename of items) {
                        const fileObj = path.parse(filename);
                        if (fileObj.ext === imgExt) {
                                const annFile = path.join(annPath, `${fileObj.name}.${config.dataset.ann.ext}`),
                                      imgFile = path.join(imageDir, filename),
                                      resImgFile = path.join(imgPath, filename),
                                      imgSize = sizeOf(imgFile);
                                fs.copyFileSync(imgFile, resImgFile)
                                let data = Object.assign(annTrmplate,{
                                        description: fileObj.name,
                                        name: fileObj.name,
                                        size: {
                                                width: imgSize.width,
                                                height: imgSize.height
                                        }
                                });
                                console.log(`Store ${annFile}`);
                                await jsonStore(annFile, data);
                        }
                }
        });
}


/**
 * EN: Move the moderated data to a separate folder from the OCR dataset
 * UA: Перенести в окрему папку з OCR-датасету промодейовані дані
 * RU: Перенести в отдельную папку из OCR-датасета промодеированные данные
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=moveChecked \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_example2/train/ \
 *                                               --opt.targetDir=../data/dataset/TextDetector/ocr_example2/val/
 */
async function moveChecked (options) {
        const srcDir = options.srcDir || './draft',
              targetDir = options.targetDir || './checked',
              annExt = '.'+config.dataset.ann.ext,
              src = { annPath: path.join(srcDir, config.dataset.ann.dir) }
        ;
        let checkedAnn = [],
            checkedImg = []
        ;
        checkDirStructure(srcDir,[config.dataset.img.dir,config.dataset.ann.dir], true);
        checkDirStructure(targetDir, [config.dataset.img.dir,config.dataset.ann.dir], true);

        fs.readdir(src.annPath, async function(err, items) {
                for (let filename of items) {
                        const fileObj = path.parse(filename);
                        if (fileObj.ext === annExt) {
                                const annName = `${fileObj.name}.${config.dataset.ann.ext}`,
                                      annFileName = path.join( src.annPath, annName);
                                const data = require(path.isAbsolute(annFileName)?annFileName:path.join(process.cwd(),
                                                                                                        annFileName)),
                                      imgName = `${data.name}.${config.dataset.img.ext}`
                                ;
                                if (data.moderation !== undefined
                                    && data.moderation.isModerated !== undefined
                                    && data.moderation.isModerated) {
                                        checkedAnn.push(annName);
                                        checkedImg.push(imgName);
                                }
                        }
                }
                console.log(`Checked: ${checkedAnn.length}`);
                moveDatasetFiles({srcDir, targetDir, Anns: checkedAnn, Imgs: checkedImg, annDir:config.dataset.ann.dir, imgDir:config.dataset.img.dir, test:false});
        });
}


/**
 * EN: Divide the dataset into 2 parts in a given proportion (move from a given folder to a specified one with a given proportion)
 * UA: Поділити датасет на 2 частини у заданій пропорції (перенести із заданої папки у зазначену із заданою пропорцією)
 * RU: Поделить датасет на 2 части в заданой пропорции (перенести из заданной папки в указанную с заданной пропорцией)
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=dataSplit \
 *                                               --opt.rate=0.5 \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_example2/train/ \
 *                                               --opt.targetDir=../data/dataset/TextDetector/ocr_example2/val/ \
 *                                               --opt.test=1 #use opt.test=1 if you want emulate split process
 */
async function dataSplit (options) {
        const srcDir = options.srcDir || './train',
            targetDir = options.targetDir || './val',
            splitRate = options.rate || 0.2,
            testMode = options.test || false,
            annExt = '.'+config.dataset.ann.ext,
            src = { annPath: path.join(srcDir, config.dataset.ann.dir) }
        ;
        let checkedAnn = [],
            checkedImg = []
        ;

        checkDirStructure(srcDir,[config.dataset.img.dir,config.dataset.ann.dir], true);
        checkDirStructure(targetDir, [config.dataset.img.dir,config.dataset.ann.dir], true);

        fs.readdir(src.annPath, async function(err, items) {
                let sItems = arrayShuffle(items),
                    cnt = Math.round(sItems.length * splitRate),
                    itemsTest = sItems.slice(0,cnt);

                for (let i=0; i<itemsTest.length; i++) {
                        const  filename = items[i],
                               fileObj = path.parse(filename);
                        if (fileObj.ext === annExt) {
                                const annName = `${fileObj.name}.${config.dataset.ann.ext}`,
                                    annFileName = path.join( src.annPath, annName);
                                const data = require(path.isAbsolute(annFileName)?annFileName:path.join(process.cwd(),
                                                                                                        annFileName)),
                                    imgName = `${data.name}.${config.dataset.img.ext}`
                                ;
                                checkedAnn.push(annName);
                                checkedImg.push(imgName);
                        }
                }
                moveDatasetFiles({srcDir, targetDir, Anns: checkedAnn, Imgs: checkedImg, annDir:config.dataset.ann.dir, imgDir:config.dataset.img.dir, test:testMode});
                console.log(`All records: ${items.length}`);
                console.log(`Moved records: ${itemsTest.length}`);
        });
}


/**
 * EN: Move to a separate folder "garbage" ("region_id": 0) from the OCR dataset
 * UA: Перенести в папку "сміття" ("region_id": 0) з OCR-датасету
 * RU: Перенести в одельную папку "мусор" ("region_id": 0) из OCR-датасета
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=moveGarbage  \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_example2/train/ \
 *                                               --opt.targetDir=../data/dataset/TextDetector/ocr_example2/garbage/
 */
async function moveGarbage (options) {
    const srcDir = options.srcDir || './draft',
        targetDir = options.targetDir || './checked',
        annExt = '.'+config.dataset.ann.ext,
        src = { annPath: path.join(srcDir, config.dataset.ann.dir) }
    ;
    let checkedAnn = [],
        checkedImg = []
    ;
    checkDirStructure(srcDir,[config.dataset.img.dir,config.dataset.ann.dir], true);
    checkDirStructure(targetDir, [config.dataset.img.dir,config.dataset.ann.dir], true);

    fs.readdir(src.annPath, async function(err, items) {
        for (let filename of items) {
            const fileObj = path.parse(filename);
            if (fileObj.ext === annExt) {
                const annName = `${fileObj.name}.${config.dataset.ann.ext}`,
                    annFileName = path.join( src.annPath, annName);
                const data = require(path.isAbsolute(annFileName)?annFileName:path.join(process.cwd(),
                                                                                        annFileName)),
                    imgName = `${data.name}.${config.dataset.img.ext}`
                ;
                if (data["region_id"] !== undefined && data["region_id"] === 0) {
                    checkedAnn.push(annName);
                    checkedImg.push(imgName);
                }
            }
        }
        console.log(`Garbage: ${checkedAnn.length}`);
        moveDatasetFiles({srcDir, targetDir, Anns: checkedAnn, Imgs: checkedImg, annDir:config.dataset.ann.dir, imgDir:config.dataset.img.dir, test:false});
    });
}


/**
 * EN: Move something to a separate folder by condition in the code text from the OCR dataset
 * UA: Перенести в окрему папку щось за умовою в тексті коду з OCR-датасету
 * RU: Перенести в отдельную папку что-либо по условию в тексте кода из OCR-датасета
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=moveSomething \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_example2/train/ \
 *                                               --opt.targetDir=../data/dataset/TextDetector/ocr_example2/train.true/
 */
async function moveSomething (options) {
    const srcDir = options.srcDir || './draft',
        targetDir = options.targetDir || './checked',
        annExt = '.'+config.dataset.ann.ext,
        src = { annPath: path.join(srcDir, config.dataset.ann.dir) }
    ;
    let checkedAnn = [],
        checkedImg = []
    ;
    checkDirStructure(srcDir,[config.dataset.img.dir,config.dataset.ann.dir], true);
    checkDirStructure(targetDir, [config.dataset.img.dir,config.dataset.ann.dir], true);

    fs.readdir(src.annPath, async function(err, items) {
        for (let filename of items) {
            const fileObj = path.parse(filename);
            if (fileObj.ext === annExt) {
                const annName = `${fileObj.name}.${config.dataset.ann.ext}`,
                    annFileName = path.join( src.annPath, annName);
                const data = require(path.isAbsolute(annFileName)?annFileName:path.join(process.cwd(),
                                                                                        annFileName)),
                    imgName = `${data.name}.${config.dataset.img.ext}`
                ;

                if (data.description.length == 7) {
                //if (data.size.height >= 32) {
                    checkedAnn.push(annName);
                    checkedImg.push(imgName);
                }
            }
        }
        console.log(`Garbage: ${checkedAnn.length}`);
        moveDatasetFiles({srcDir, targetDir, Anns: checkedAnn, Imgs: checkedImg, annDir:config.dataset.ann.dir, imgDir:config.dataset.img.dir, test:false});
    });
}


/**
 * EN: Move duplicate records (by photo) to a separate folder
 * UA: Перенести дублікати записай (по фото) в окрему папку
 * RU: Перенести дубликаты записай (по фото) в отдельную папку
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=moveDupes \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_example2/train/ \
 *                                               --opt.targetDir=../data/dataset/TextDetector/ocr_example2/train.dupes/
 */
async function moveDupes (options) {
    const srcDir = options.srcDir || './draft',
        targetDir = options.targetDir || './checked',
        annExt = '.'+config.dataset.ann.ext,
        src = { annPath: path.join(srcDir, config.dataset.ann.dir) },
        imgPath = path.join(srcDir, config.dataset.img.dir)
    ;
    let checkedAnn = [],
        checkedImg = []
    ;
    let checkSum = {}, checklogs = {};
    checkDirStructure(srcDir,[config.dataset.img.dir,config.dataset.ann.dir], true);
    checkDirStructure(targetDir, [config.dataset.img.dir,config.dataset.ann.dir], true);

    fs.readdir(src.annPath, async function(err, items) {
        for (let filename of items) {
            const  fileObj = path.parse(filename);
            if (fileObj.ext === annExt) {
                const annName = `${fileObj.name}.${config.dataset.ann.ext}`,
                    annFileName = path.join( src.annPath, annName);
                const data = require(path.isAbsolute(annFileName)?annFileName:path.join(process.cwd(),
                                                                                        annFileName)),
                    imgName = `${data.name}.${config.dataset.img.ext}`
                ;
                let imgFullFile = path.join(imgPath, imgName),
                    imgSize = sizeOf(imgFullFile),
                    imgMd5 = await md5File(imgFullFile),
                    imgSizeHash = `${imgMd5}-${imgSize.width}x${imgSize.height}`;

                if (checkSum[imgSizeHash] !== undefined) {
                    checkedAnn.push(annName);
                    checkedImg.push(imgName);
                    if (checklogs[checkSum[imgSizeHash]] === undefined) {
                        checklogs[checkSum[imgSizeHash]] = []
                    }
                    checklogs[checkSum[imgSizeHash]].push(imgName)
                } else {
                    checkSum[imgSizeHash] = imgName;
                }
            }
        }
        console.log(`Garbage: ${checkedAnn.length}`);
        moveDatasetFiles({srcDir, targetDir, Anns: checkedAnn, Imgs: checkedImg, annDir:config.dataset.ann.dir, imgDir:config.dataset.img.dir, test:false});
        fs.writeFileSync('./logs.json', JSON.stringify(checklogs,null,2));
    });
}


/**
 * EN: Glue several folders into one only for unpainted numbers
 * UA: Склеїти кілька папок в одну лише для незафарбованих номерів
 * RU: Склеить несколько папок в одну только для незакрашеных номеров
 *
 * @param options
 * @example NODE_ENV=consoleExample ./console.js --section=default --action=dataJoin \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_example2/train/ \
 *                                               --opt.srcDir=../data/dataset/TextDetector/ocr_example2/val/ \
 *                                               --opt.targetDir=../data/dataset/TextDetector/ocr_example2/train/
 */
async function dataJoin (options) {
    if (!Array.isArray(options.srcDir) || options.srcDir.length < 2) {
        throw new Error('"opt.srcDir" must be array for 2 (min) elements!');
    }
    console.log("", options.targetDir);
    const srcDir = options.srcDir,
          targetDir = options.targetDir || new Error('"opt.targetDir" is not defined!'),
          annExt = '.'+config.dataset.ann.ext
    ;
    for (let dir of srcDir) {
        checkDirStructure(dir,[config.dataset.img.dir,config.dataset.ann.dir]);
    }
    checkDirStructure(targetDir, [config.dataset.img.dir,config.dataset.ann.dir], true);
    joinAnnotationOcrDataset(srcDir, annExt, targetDir)
}


module.exports = {
    index,
    createAnnotations,
    moveChecked,
    dataSplit,
    moveGarbage,
    dataJoin,
    moveSomething,
    moveDupes,
};
