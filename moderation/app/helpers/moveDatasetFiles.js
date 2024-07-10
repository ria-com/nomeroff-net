const fs = require('fs'),
      path = require('path')
;

function moveFilesPack({datasetName, sourceDir, targetDir, Files, filesDir, stayFiles = [], test = false}) {
    Files = Array.from(new Set(Files));
    stayFiles = Array.from(new Set(stayFiles));
    if (test) { console.log(`Move dataset ${datasetName}: ${Object.keys(Files).length} items was started!`) }
    for(let i in Files) {
        let file_name = path.join(filesDir, Files[i]);
        //console.log(`Move ${file_name}!`)

        if (test) {
            console.log(`Rename ${path.join(sourceDir, file_name)} - ${path.join(targetDir, file_name)}`)
        } else {
            if (stayFiles.includes(Files[i])) {
                console.log(`Copy ${path.join(sourceDir, file_name)} to ${path.join(targetDir, file_name)}`)
                fs.copyFileSync(path.join(sourceDir, file_name), path.join(targetDir, file_name));
            } else {
                fs.renameSync(path.join(sourceDir, file_name), path.join(targetDir, file_name));
            }
        }
    }
}

module.exports = function moveFiles({sourceDir, targetDir,
                                        Anbs, anbDir, stayAnbs,
                                        Anns, annDir, Imgs, imgDir,
                                        Boxs, boxDir, stayBoxs,
                                        Srcs, srcDir, staySrcs,
                                        test = false}) {

    moveFilesPack({datasetName: 'Anns', sourceDir, targetDir, Files:Anns, filesDir:annDir, test});
    moveFilesPack({datasetName: 'Imgs', sourceDir, targetDir, Files:Imgs, filesDir:imgDir, test});

    moveFilesPack({datasetName: 'Anbs', sourceDir, targetDir, Files:Anbs, filesDir:anbDir, stayFiles:stayAnbs, test});
    moveFilesPack({datasetName: 'Boxs', sourceDir, targetDir, Files:Boxs, filesDir:boxDir, stayFiles:stayBoxs, test});
    moveFilesPack({datasetName: 'Srcs', sourceDir, targetDir, Files:Srcs, filesDir:srcDir, stayFiles:staySrcs, test});
}
