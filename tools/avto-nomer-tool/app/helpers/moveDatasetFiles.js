const fs = require('fs'),
    path = require('path')
;

module.exports = function moveFiles({srcDir, targetDir, Anns, Imgs, annDir, imgDir, test = false}) {
    for(let i in Anns) {
        let imgFilename = path.join(imgDir,Imgs[i]);
        let annFilename = path.join(annDir,Anns[i]);

        if (test) {
            console.log(`Rename ${path.join(srcDir,annFilename)} - ${path.join(targetDir, annFilename)}`)
            console.log(`Rename ${path.join(srcDir,imgFilename)} - ${path.join(targetDir, imgFilename)}`);
        } else {
            fs.renameSync(path.join(srcDir,annFilename), path.join(targetDir, annFilename));
            fs.renameSync(path.join(srcDir,imgFilename), path.join(targetDir, imgFilename));
        }
    }
}
