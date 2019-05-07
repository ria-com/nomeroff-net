const config = require('config'),
      fs = require('fs'),
      path = require('path'),
      jsonStore = require('../../app/helpers/jsonStore'),
      checkDir = require('../../app/helpers/checkDir'),
      checkDirStructure = require('../../app/helpers/checkDirStructure'),
      moveDatasetFiles = require('../../app/helpers/moveDatasetFiles')
      sizeOf = require('image-size')
;

/**
 * @module controllers/defaultController
 */
async function index (options) {
        console.log('Hello world defaultController & index action with options: ' +JSON.stringify(options));
};


/**
 * @param options
 * @example ./console.js --action=createAnnotations  --baseDir=../../dataset/ocr/kz/kz2
 */
async function createAnnotations (options) {
        let baseDir = options.baseDir || config.dataset.baseDir;
        checkDir(baseDir);

        const imgPath = path.join(baseDir, config.dataset.img.dir),
              annPath = path.join(baseDir, config.dataset.ann.dir),
              imgExt = '.'+config.dataset.img.ext,
              tplPath = path.join(config.template.path, config.template.annDefault),
              annTrmplate = require(tplPath);
        checkDir(imgPath);
        checkDir(annPath);

        console.log(imgPath);
        fs.readdir(imgPath, async function(err, items) {
                for (var i=0; i<items.length; i++) {
                        const  filename = items[i],
                               fileObj = path.parse(filename);
                        if (fileObj.ext == imgExt) {
                                const annFile = path.join(annPath, `${fileObj.name}.${config.dataset.ann.ext}`),
                                      imgFile = path.join(imgPath, filename),
                                      imgSize = sizeOf(imgFile);
                                let data = Object.assign(annTrmplate,{
                                        description: fileObj.name,
                                        name: fileObj.name,
                                        size: {
                                                width: imgSize.width,
                                                height: imgSize.height
                                        }
                                });
                                await jsonStore(annFile, data);
                                // if (data.description.length > 8) {
                                //         console.log(`File: ${filename} [${data.description}]`);
                                // }
                        }
                }
        });
};

/**
 * @param options
 * @example ./console.js --section=default --action=moveChecked  --opt.srcDir=../../datasets/ocr/kz/draft --opt.targetDir=../../datasets/ocr/kz/checked
 */
async function moveChecked (options) {
        const srcDir = options.srcDir || './draft',
              targetDir = options.targetDir || './checked',
              annExt = '.'+config.dataset.ann.ext,
              src = { annPath: path.join(srcDir, config.dataset.ann.dir) },
              target = { annPath: path.join(targetDir, config.dataset.ann.dir) }
        ;
        let checkedAnn = [],
            checkedImg = []
        ;
        checkDirStructure(srcDir,[config.dataset.img.dir,config.dataset.ann.dir]);
        checkDirStructure(targetDir, [config.dataset.img.dir,config.dataset.ann.dir]);

        fs.readdir(src.annPath, async function(err, items) {
                for (var i=0; i<items.length; i++) {
                        const  filename = items[i],
                            fileObj = path.parse(filename);
                        //console.log(fileObj)
                        if (fileObj.ext == annExt) {
                                const annName = `${fileObj.name}.${config.dataset.ann.ext}`,
                                      annFilename = path.join( src.annPath, annName);
                                const data = require(path.isAbsolute(annFilename)?annFilename:path.join(process.cwd(), annFilename)),
                                      imgName = `${data.name}.${config.dataset.img.ext}`
                                ;
                                if (data.moderation != undefined && data.moderation.isModerated != undefined && data.moderation.isModerated) {
                                        checkedAnn.push(annName);
                                        checkedImg.push(imgName);
                                }
                        }
                }
                console.log(`Checked: ${checkedAnn.length}`);
                moveDatasetFiles({srcDir, targetDir, Anns: checkedAnn, Imgs: checkedImg, annDir:config.dataset.ann.dir, imgDir:config.dataset.img.dir, test:false});
        });
}


module.exports = {index, createAnnotations, moveChecked};


