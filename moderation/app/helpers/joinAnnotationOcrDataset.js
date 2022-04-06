const fs = require('fs'),
      path = require('path'),
      config = require('config'),
      moveDatasetFiles = require('./moveDatasetFiles')
;

module.exports = function joinAnnotationOcrDataset(srcDir, annExt, targetDir) {
    for (let dir of srcDir) {
        let annDir = path.join(dir, config.dataset.ann.dir),
            Anns = [],
            Imgs = []
        ;

        fs.readdir(annDir, async function(err, items) {
            for (let filename of items) {
                let fileObj = path.parse(filename);
                if (fileObj.ext === annExt) {
                    let annName = `${fileObj.name}.${config.dataset.ann.ext}`,
                        annFileName = path.join( annDir, annName);
                    let data = require(path.isAbsolute(annFileName)?annFileName:path.join(process.cwd(), annFileName)),
                        imgName = `${data.name}.${config.dataset.img.ext}`
                    ;
                    if (data["state_id"] !== undefined && data["state_id"] === 2) {
                        Anns.push(annName);
                        Imgs.push(imgName);
                    }
                }
            }
            console.log(`Anns: ${Anns.length}`);
            moveDatasetFiles({srcDir:dir, targetDir, Anns, Imgs, annDir:config.dataset.ann.dir, imgDir:config.dataset.img.dir, test:false});
        })
    }
};
