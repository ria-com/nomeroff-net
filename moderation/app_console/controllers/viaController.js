const config = require('config'),
      path = require('path'),
      checkDir = require('../../app/helpers/checkDir'),
      checkDirStructure = require('../../app/helpers/checkDirStructure'),
      arrayShuffle = require('../../app/helpers/arrayShuffle'),
      arrayGroupFix = require('../../app/helpers/arrayGroupFix'),
      sleep = require('sleep-promise'),
      {prepareViaPart,moveViaPart,writeViaPart,writeViaPartFull} = require('../../app/managers/viaManager')
;

/**
 * @module controllers/defaultController
 */
async function index (options) {
    console.log('Hello world defaultController & index action with options: ' +JSON.stringify(options));
}


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
          keys = arrayShuffle(Object.keys(data._via_img_metadata)),
          cnt = Math.round(keys.length * splitRate),
          partKeys = {
              'train': keys.slice(cnt),
              'val': keys.slice(0,cnt)
          }
    ;
    checkDir(srcDir);
    checkDirStructure(targetDir,config.via.partDirs,true);

    for (let key of config.via.partDirs) {
        let dataPart = prepareViaPart(data, partKeys[key], srcDir);
        moveViaPart(dataPart,srcDir,targetDir,key);
        console.log(`moveViaPart for ${key} is done...`)
        writeViaPart(dataPart,targetDir,key,viaFile);
        console.log(`Macking ${key} done...`)
    }
    await sleep(1000)
}


/**
 * Объеденить 2 датасета в один для Mask RCNN
 *
 * @param options
 * @example ./console.js --section=via --action=joinVia \
 * --opt.srcJson=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-11/src1/via_data_ria_1_full.json
 * --opt.srcJson=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-11/src2/via_data_ria2.json \
 * --opt.targetDir=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-11/target
 * --opt.viaFile=via_region_data.json
 */
async function joinVia (options) {
    if (options.srcJson !== undefined && Array.isArray(options.srcJson)) {
        throw new Error('"opt.srcJson" must be array for 2 (min) elements!')
    }
    const srcJson = options.srcJson,
          targetDir = options.targetDir || new Error('"opt.targetDir" is not defined!'),
          viaFile = options.viaFile || new Error('"opt.viaFile" is not defined!')
    ;
    let data = {};

    for (let item of srcJson) {
        let srcDir = path.dir_name(item);
        checkDir(srcDir);
        let dataPart = require(item);
        if (data === undefined) {
            data = dataPart
        } else {
            data._via_img_metadata = {...data._via_img_metadata, ...dataPart._via_img_metadata};
        }
        moveViaPart(dataPart,srcDir,path.dir_name(targetDir),path.basename(targetDir));
    }
    writeViaPart(
        data,
        path.dir_name(targetDir),
        path.basename(targetDir),
        viaFile);
    await sleep(1000)
}

/**
 * Объеденить 2 датасета в один для Mask RCNN
 *
 * @param options
 * @example ./console.js --section=via --action=addAttribute --opt.srcJson=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-09-17/emptyPlate/via_region_data_emptyPlate.json --opt.attrName=class --opt.attrValue=emptyPlate
 */
async function addAttribute (options) {
    const srcJson = options.srcJson || new Error('"opt.srcJson" is not defined!') ,
        attrName = options.attrName || new Error('"opt.attrName" is not defined!'),
        attrValue = options.attrValue || new Error('"opt.attrName" is not defined!')
    ;
    let srcDir = path.dir_name(srcJson);
    checkDir(srcDir);
    let dataPart = require(srcJson), data = dataPart._via_img_metadata;
    for (let key in data) {
        let item = data[key];
        for (let region of item.regions) {
            region.region_attributes[attrName] = attrValue
        }
    }
    writeViaPartFull(dataPart,srcJson);
    await sleep(1000)
}

function convertToViaIdx(viaIdx,arr) {
    let newArr = [];
    for(let idx of arr) {
        newArr.push(viaIdx[idx])
    }
    return newArr;
}

/**
 * Создание val i train для Mask RCNN
 *
 * @param options
 * @example ./console.js --section=via --action=groupSplit --opr.rate=0.2 --opt.srcDir=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-07/draft --opt.targetDir=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-07 --opt.viaFile=via_region_data.json
 */
async function groupSplit (options) {
    const srcDir = options.srcDir || new Error('"opt.srcDir" is not defined!'),
        targetDir = options.targetDir || new Error('"opt.targetDir" is not defined!'),
        viaFile = options.viaFile || new Error('"opt.viaFile" is not defined!'),
        splitRate = options.rate || 0.2,
        srcViaPath = path.join(srcDir, viaFile),
        data = require(srcViaPath),
        viaIdx = {}
    ;
    for (let via_idx in data._via_img_metadata) {
        viaIdx[data._via_img_metadata[via_idx].file_name] = via_idx;
    }

    const keys = arrayShuffle(Object.keys(viaIdx)),
          cnt = Math.round(keys.length * (1-splitRate)),
          partKeys = arrayGroupFix((1-splitRate), {
            'train': keys.slice(cnt),
            'val': keys.slice(0,cnt)
          })
    ;

    partKeys.train =  convertToViaIdx(viaIdx,partKeys.train);
    partKeys.val =  convertToViaIdx(viaIdx,partKeys.val);

    checkDir(srcDir);
    checkDirStructure(targetDir,config.via.partDirs,true);

    for (let key of config.via.partDirs) {
        let dataPart = prepareViaPart(data, partKeys[key], srcDir);
        moveViaPart(dataPart,srcDir,targetDir,key);
        console.log(`moveViaPart for ${key} is done...`)
        writeViaPart(dataPart,targetDir,key,viaFile);
        console.log(`Macking ${key} done...`)
    }
    console.log(`Final rate: ${partKeys.val.length/(partKeys.train.length+partKeys.val.length)}`);
    await sleep(1000)
}


module.exports = {
    index,
    split,
    joinVia,
    addAttribute,
    groupSplit,
};