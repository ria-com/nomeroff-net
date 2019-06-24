const config = require('config'),
    fs = require('fs'),
    path = require('path'),
    checkDir = require('../../app/helpers/checkDir'),
    checkDirStructure = require('../../app/helpers/checkDirStructure'),
    arrayShuffle = require('../../app/helpers/arrayShuffle'),
    sleep = require('sleep-promise'),
    {prepareViaPart,moveViaPart,writeViaPart} = require('../../app/managers/viaManager')
;

/**
 * @module controllers/defaultController
 */
async function index (options) {
    console.log('Hello world defaultController & index action with options: ' +JSON.stringify(options));
};


/**
 * Создание val i train для Mask RCNN
 *
 * @param options
 * @example ./console.js --section=via --action=split --opr.rate=0.2 --opt.srcDir=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-07/draft --opt.targetDir=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-07 --opt.viaFile=via_region_data.json
 */
async function split (options) {
    const srcDir = options.srcDir || new Error('"opt.srcDir" is not defined!'), // /mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-07/draft,
          targetDir = options.targetDir || new Error('"opt.targetDir" is not defined!'), // /mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-07,
          viaFile = options.viaFile || new Error('"opt.viaFile" is not defined!'), // via_region_data.json,
          splitRate = options.rate || 0.2,
          srcViaPath = path.join(srcDir, viaFile),
          data = require(srcViaPath),
          keys = arrayShuffle(Object.keys(data._via_img_metadata))
          cnt = Math.round(keys.length * splitRate),
          partKeys = {
              'train': keys.slice(cnt),
              'val': keys.slice(0,cnt)
          }
    ;
    checkDir(srcDir);
    checkDirStructure(targetDir,config.via.partDirs,true);

    for (let key of config.via.partDirs) {
        //checkDir(path.join(targetDir, key), true);
        let dataPart = prepareViaPart(data, partKeys[key], srcDir);
        //console.log(`making dataPart for ${key} is done...`)
        moveViaPart(dataPart,srcDir,targetDir,key);
        //console.log(`moveViaPart for ${key} is done...`)
        writeViaPart(dataPart,targetDir,key,viaFile);
        console.log(`Macking ${key} done...`)
    }
    await sleep(1000)
};


/**
 * Объеденить 2 датасета в один для Mask RCNN
 *
 * @param options
 * @example ./console.js --section=via --action=joinVia --opt.srcJson=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-11/src1/via_data_ria_1_full.json --opt.srcJson=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-11/src2/via_data_ria2.json --opt.targetDir=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-11/target --opt.viaFile=via_region_data.json
 */
async function joinVia (options) {
    //console.log('Hello world defaultController & index action with options: ' +JSON.stringify(options));
    if (options.srcJson!=undefined && Array.isArray(options.srcJson)) { // --opt.srcJson=/mnt/data/home/nn/datasets/mrcnn2/via_data_ria_1_full.json --opt.srcJson=/mnt/data/home/nn/datasets/mrcnn3/via_data_ria2.json
        new Error('"opt.srcJson" must be array for 2 (min) elements!')
    }
    const srcJson = options.srcJson ,
          targetDir = options.targetDir || new Error('"opt.targetDir" is not defined!'), // --opt.targetDir=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-11/target
          viaFile = options.viaFile || new Error('"opt.viaFile" is not defined!'), // via_region_data.json,
          srcJsonData = []
    ;
    let data;

    for (let item of srcJson) {
        let srcDir = path.dirname(item);
        checkDir(srcDir);
        let dataPart = require(item);
        if (data == undefined) { data = dataPart } else {
            data._via_img_metadata = {...data._via_img_metadata, ...dataPart._via_img_metadata};
        }
        srcJsonData.push(dataPart);
        moveViaPart(dataPart,srcDir,path.dirname(targetDir),path.basename(targetDir));
    }
    writeViaPart(data,path.dirname(targetDir),path.basename(targetDir),viaFile);
    await sleep(1000)
};


module.exports = {index, split, joinVia};