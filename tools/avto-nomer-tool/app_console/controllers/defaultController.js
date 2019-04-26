const config = require('config'),
      fs = require('fs'),
      path = require('path'),
      jsonStore = require('../../app/helpers/jsonStore'),
      checkDir = require('../../app/helpers/checkDir'),
      sizeOf = require('image-size')
;

/**
 * @module controllers/defaultController
 */
async function index (options) {
        console.log('Hello world defaultController & index action with options: ' +JSON.stringify(options));
};


/**
 * ./console.js --action=createAnnotations  --baseDir=../../dataset/ocr/kz/kz2
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


module.exports = {index, createAnnotations};


