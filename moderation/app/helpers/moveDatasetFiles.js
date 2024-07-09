const fs = require('fs'),
      path = require('path')
;

function moveFilesPack({datasetName, sourceDir, targetDir, Files, filesDir, test = false}) {
    if (test) { console.log(`Move dataset ${datasetName}: ${Object.keys(Files).length} items was started!`) }
    for(let i in Files) {
        let file_name = path.join(filesDir, Files[i]);
        console.log(`Move ${file_name}!`)

        if (test) {
            console.log(`Rename ${path.join(sourceDir, file_name)} - ${path.join(targetDir, file_name)}`)
        } else {
            fs.renameSync(path.join(sourceDir, file_name), path.join(targetDir, file_name));
        }
    }
}

module.exports = function moveFiles({sourceDir, targetDir,
                                        Anbs, anbDir, Anns, annDir, Imgs, imgDir,
                                        Boxs, boxDir, Srcs, srcDir,
                                        test = false}) {
    moveFilesPack({datasetName: 'Anns', sourceDir, targetDir, Files:Anns, filesDir:annDir, test});
    moveFilesPack({datasetName: 'Imgs', sourceDir, targetDir, Files:Imgs, filesDir:imgDir, test});

    Anbs = Array.from(new Set(Anbs));
    moveFilesPack({datasetName: 'Anbs', sourceDir, targetDir, Files:Anbs, filesDir:anbDir, test});

    Boxs = Array.from(new Set(Boxs));
    moveFilesPack({datasetName: 'Boxs', sourceDir, targetDir, Files:Boxs, filesDir:boxDir, test});

    Srcs = Array.from(new Set(Srcs));
    moveFilesPack({datasetName: 'Srcs', sourceDir, targetDir, Files:Srcs, filesDir:srcDir, test});
}
