const fs = require('fs'),
      path = require('path')
;

module.exports = {
    prepareViaPart (data, keys, srcDir){
        const dataPart = Object.assign({},data);
        dataPart._via_img_metadata = {};
        // console.log('keys-----------------------------');
        // console.log(keys);
        for (let item of keys) {
            //console.log(`item: ${item}`);
            let file =  path.join(srcDir, data._via_img_metadata[item].filename);
            //console.log(file);
            if (fs.existsSync(file)) {
                dataPart._via_img_metadata[item] = data._via_img_metadata[item]
            } else {
                new Error(`File "${item}" is not exists!`)
            }
        }
        //console.log(dataPart);
        return dataPart
    },

    moveViaPart(dataPart, srcDir, targetDir, subDir) {
        //console.log(`Start moving ${targetDir}`);
        for (let key in dataPart._via_img_metadata) {
            let fileIn = path.join(srcDir, dataPart._via_img_metadata[key].filename),
                fileOut = path.join(targetDir, subDir, dataPart._via_img_metadata[key].filename)
            ;
            //console.log(`fileIn: ${fileIn}\nfileOut: ${fileOut}`);
            try {
                if (!fs.existsSync(fileOut)) {
                    fs.renameSync(fileIn, fileOut);
                }
            } catch (e) {
                console.log("Not exists rename", fileIn, fileOut);
            }
        }
        return dataPart
    },

    writeViaPart(dataPart, targetDir, subDir, viaFile) {
        let fullViaFile = path.join(targetDir, subDir, viaFile);
        module.exports.writeViaPartFull(dataPart, fullViaFile);
    },

    writeViaPartFull(dataPart, fullViaFile) {
        let wstream = fs.createWriteStream(fullViaFile);
        wstream.write(JSON.stringify(dataPart,null,2));
        wstream.end();
    }
}