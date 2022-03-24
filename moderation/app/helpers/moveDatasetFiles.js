const fs = require('fs'),
      path = require('path')
;

module.exports = function moveFiles({srcDir, targetDir, Anns, Imgs, annDir, imgDir, test = false}) {
    for(let i in Anns) {
        let imgfile_name = path.join(imgDir, Imgs[i]);
        let annfile_name = path.join(annDir, Anns[i]);

        if (test) {
            console.log(`Rename ${path.join(srcDir,annfile_name)} - ${path.join(targetDir, annfile_name)}`)
            console.log(`Rename ${path.join(srcDir,imgfile_name)} - ${path.join(targetDir, imgfile_name)}`);
        } else {
            fs.renameSync(path.join(srcDir,annfile_name), path.join(targetDir, annfile_name));
            fs.renameSync(path.join(srcDir,imgfile_name), path.join(targetDir, imgfile_name));
        }
    }
}
