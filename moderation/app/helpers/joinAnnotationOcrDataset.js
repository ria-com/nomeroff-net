const fs = require('fs'),
      path = require('path'),
      config = require('config'),
      moveDatasetFiles = require('./moveDatasetFiles')
;

module.exports = function joinAnnotationOcrDataset(srcDir, annExt) {
    for (let dir of srcDir) {
        let annDir = path.join(dir, config.dataset.ann.dir),
            Anns = [],
            Imgs = []
        ;

        fs.readdir(annDir, async function(err, items) {
            for (let file_name of items) {
                let fileObj = path.parse(file_name);
                if (fileObj.ext === annExt) {
                    let annName = `${fileObj.name}.${config.dataset.ann.ext}`,
                        annfile_name = path.join( annDir, annName);
                    let data = require(path.isAbsolute(annfile_name)?annfile_name:path.join(process.cwd(), annfile_name)),
                        imgName = `${data.name}.${config.dataset.img.ext}`
                    ;
                    if (data.state_id !== undefined && data.state_id === 2) {
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