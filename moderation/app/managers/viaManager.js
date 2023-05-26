const fs = require('fs'),
      path = require('path')
;

module.exports = {
    prepareViaPart (data, keys, srcDir){
        const dataPart = Object.assign({},data);
        dataPart._via_img_metadata = {};
        for (let item of keys) {
            let file =  path.join(srcDir, data._via_img_metadata[item].filename);
            if (fs.existsSync(file)) {
                dataPart._via_img_metadata[item] = data._via_img_metadata[item]
            } else {
                throw new Error(`File "${item}" is not exists!`)
            }
        }
        return dataPart
    },

    moveViaPart(dataPart, srcDir, targetDir, subDir) {
        for (let key in dataPart._via_img_metadata) {
            let fileIn = path.join(srcDir, dataPart._via_img_metadata[key].filename),
                fileOut = path.join(targetDir, subDir, dataPart._via_img_metadata[key].filename)
            ;
            try {
                if (!fs.existsSync(fileOut)) {
                    fs.renameSync(fileIn, fileOut);
                }
            } catch (e) {
                console.log("Rename error: ", fileIn, fileOut);
                console.log(e)
            }
        }
        return dataPart
    },

    writeViaPart(dataPart, targetDir, subDir, viaFile) {
        let fullViaFile = path.join(targetDir, subDir, viaFile);
        module.exports.writeViaPartFull(dataPart, fullViaFile);
    },

    writeViaPartFull(dataPart, fullViaFile) {
        fs.writeFileSync(fullViaFile, JSON.stringify(dataPart, null, 2), 'utf-8');
        // let wstream = fs.createWriteStream(fullViaFile);
        // wstream.write(JSON.stringify(dataPart,null,2));
        // wstream.end();
    }
}