const config = require('config'),
      path = require('path'),
      fs =require('fs'),
      checkDir = require('../../app/helpers/checkDir'),
      checkDirStructure = require('../../app/helpers/checkDirStructure'),
      arrayShuffle = require('../../app/helpers/arrayShuffle'),
      arrayGroupFix = require('../../app/helpers/arrayGroupFix'),
      sleep = require('sleep-promise'),
      {prepareViaPart,moveViaPart,writeViaPart,writeViaPartFull} = require('../../app/managers/viaManager')
;
const {match} = require("yarn/lib/cli");

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
          keys = arrayShuffle(Object.keys(data["_via_img_metadata"])),
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
 *
 * or
 *
 * @example ./console.js --section=via --action=joinVia \
 * --opt.srcJsonList=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-11/via_list.json
 * --opt.targetDir=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-06-11/target
 * --opt.viaFile=via_region_data.json
 *
 */
async function joinVia (options) {
    console.log(JSON.stringify(options));
    //process.exit(0)
    if (options.srcJson != undefined && !Array.isArray(options.srcJson)) {
        options.srcJson = [options.srcJson]
    }
    if (options.srcJsonList != undefined) {
        options.srcJson = require(options.srcJsonList)
    }
    const srcJson = options.srcJson || new Error('"opt.srcJson" is not defined!'),
          targetDir = options.targetDir || new Error('"opt.targetDir" is not defined!'),
          viaFile = options.viaFile || new Error('"opt.viaFile" is not defined!')
    ;
    let data = {};

    for (let item of srcJson) {
        let srcDir = path.dirname(item);
        checkDir(srcDir);
        let dataPart = require(item);
        if (Object.keys(data).length == 0) {
            data = dataPart
        } else {
            data["_via_img_metadata"] = {...data["_via_img_metadata"], ...dataPart["_via_img_metadata"]};
        }
        moveViaPart(dataPart, srcDir, path.dirname(targetDir),path.basename(targetDir));
    }
    writeViaPart(
        data,
        path.dirname(targetDir),
        path.basename(targetDir),
        viaFile);
    await sleep(1000)
}

/**
 * Добавить атрибут в region
 *
 * @param options
 * @example ./console.js --section=via --action=addAttribute --opt.srcJson=/mnt/data/home/nn/datasets/autoriaNumberplateDataset-2019-09-17/emptyPlate/via_region_data_emptyPlate.json --opt.attrName=class --opt.rewrite=1 --opt.attrValue=emptyPlate
 */
async function addAttribute (options) {
    const srcJson = options.srcJson || new Error('"opt.srcJson" is not defined!') ,
        attrName = options.attrName || new Error('"opt.attrName" is not defined!'),
        attrValue = options.attrValue
    ;
    let rewriteSettedAttribute = 1;
    let rewriteEmptyValue = 1;
    if (options.rewrite != undefined) { rewriteSettedAttribute = options.rewrite }
    if (options.rewriteEmpty != undefined) { rewriteEmptyValue = options.rewriteEmpty }
    rewriteSettedAttribute = Boolean(Number(rewriteSettedAttribute));
    rewriteEmptyValue = Boolean(Number(rewriteEmptyValue));
    // console.log(JSON.stringify(options))
    console.log(`attrName="${attrName}"`)
    console.log(`attrValue="${attrValue}"`)
    let srcDir = path.dirname(srcJson);
    checkDir(srcDir);
    let dataPart = require(srcJson), data = dataPart["_via_img_metadata"];
    for (let key in data) {
        let item = data[key];
        for (let region of item.regions) {
            if (region["region_attributes"][attrName] != undefined && (!rewriteSettedAttribute)) {
                // pass
                if (region["region_attributes"][attrName].trim().length == 0 && rewriteEmptyValue) {
                    region["region_attributes"][attrName] = attrValue
                }
            } else {
                region["region_attributes"][attrName] = attrValue
            }
        }
    }
    writeViaPartFull(dataPart,srcJson);
    //await sleep(1000)
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
    for (let viaId in data["_via_img_metadata"]) {
        viaIdx[data["_via_img_metadata"][viaId].filename] = viaId;
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

/**
 * Очистить неразмеченные записи
 *
 * @param options
 * @example ./console.js --section=via --action=purgeEmpty --opt.srcJson=/tmp/30_29_Hyundai_bb100.json --opt.targetJson=/tmp/30_29_Hyundai_bb100_.json
 */
async function purgeEmpty (options) {
    const
        srcJson = options.srcJson || new Error('"opt.srcJson" is not defined!') ,
        targetJson = options.targetJson || new Error('"opt.targetJson" is not defined!')
    ;
    // console.log(JSON.stringify(options))
    if (fs.existsSync(srcJson)) {
        let dataPart = require(srcJson), data = dataPart["_via_img_metadata"];
        for (let key in data) {
            let item = data[key];
            if (item.regions == undefined || (item.regions.length == 0)) {
                delete data[key];
            }
        }
        writeViaPartFull(dataPart, targetJson);
        await sleep(1000)
    } else {
        throw new Error(`"${options.srcJson}" is not found!`)
    }
}


/**
 * Вырезать первых {opt.max} записей
 *
 * @param options
 * @example ./console.js --section=via --action=cropFirst --opt.max=100 --opt.srcJson=/tmp/30_29_Hyundai_bb100.json --opt.targetJson=/tmp/30_29_Hyundai_bb100_.json
 */
async function cropFirst (options) {
    const
        srcJson = options.srcJson || new Error('"opt.srcJson" is not defined!') ,
        targetJson = options.targetJson || new Error('"opt.targetJson" is not defined!'),
        max = options.max || new Error('"options.records" is not defined!')
    ;
    // console.log(JSON.stringify(options))
    if (fs.existsSync(srcJson)) {
        let dataPart = require(srcJson), data = dataPart["_via_img_metadata"], new_via_img_metadata = {}, cnt=0;
        for (let key in data) {
            new_via_img_metadata[key] = data[key];
            cnt++;
            if (cnt==max) break;
        }
        dataPart["_via_img_metadata"]=new_via_img_metadata
        writeViaPartFull(dataPart, targetJson);
        await sleep(1000)
    } else {
        throw new Error(`"${options.srcJson}" is not found!`)
    }
}


/**
 * Вырезать первых {opt.max} записей
 *
 * @param options
 * @example ./console.js --section=via --action=getUpdated --opt.nnDatasetDir=/mnt/sdd1/datasets/2-3lines/train_orig \
 *                       --opt.targetJson=/mnt/sdd1/datasets/2-3lines/train_orig/viaCorrected.json
 */
async function getUpdated (options) {
    const
        base_dir = options.nnDatasetDir,
        img = path.join(base_dir, config.dataset.img.dir),
        ann = path.join(base_dir, config.dataset.ann.dir),
        anb = path.join(base_dir, config.dataset.anb.dir),
        box = path.join(base_dir, config.dataset.box.dir),
        src = path.join(base_dir, config.dataset.src.dir),
        anb_files = fs.readdirSync(anb),
        via_template = require(path.join(config.template.path, config.template.via.dir, config.template.via.tpl_name)),
        targetJson = options.targetJson,
        targetDir = path.dirname(targetJson),
        targetName = path.basename(targetJson, '.json')
    ;
    via_template._via_img_metadata = {};
    via_template._via_settings.project.name = targetName;
    for (const anb_file of anb_files) {
        let toViaItems = false;
        const
            anb_data = require(path.join(anb, anb_file))
            src_path = path.join(src, anb_data.src),
            target_path = path.join(targetDir, anb_data.src),
            stats = fs.statSync(src_path),
            itemNew = {
                filename: anb_data.src,
                size: stats.size,
                regions: [],
                file_attributes: {}
            }
        ;
        for(const region in anb_data.regions) {
            const item = anb_data.regions[region];
            if (item.updated != undefined && item.updated) {
                toViaItems = true;
            }
            if (item.keypoints.length = 4) {
                itemNew.regions.push(
                    {
                        shape_attributes: {
                            name: "polygon",
                            all_points_x: [
                                Math.round(item.keypoints[0][0]),
                                Math.round(item.keypoints[1][0]),
                                Math.round(item.keypoints[2][0]),
                                Math.round(item.keypoints[3][0])
                            ],
                            all_points_y: [
                                Math.round(item.keypoints[0][1]),
                                Math.round(item.keypoints[1][1]),
                                Math.round(item.keypoints[2][1]),
                                Math.round(item.keypoints[3][1])
                            ]
                        },
                        region_attributes: {
                            class: "numberplate",
                            label: "numberplate"
                        }
                    }
                )
            } else {
                console.warn(`Incorect keypoins ${item.keypoints.length} in region ${region} for image ${anb_data.src}!`)
            }
        }
        if (toViaItems) {
            via_template._via_img_metadata[anb_data.src] = itemNew;
            fs.copyFileSync(src_path,target_path);
        }
    }
    if (Object.keys(via_template._via_img_metadata)) {
        fs.writeFileSync(targetJson, JSON.stringify(via_template, null, 2), 'utf-8');
        console.log(`Writing via JSON to "${targetJson}"!`)
    }
    console.log(`Done`)
}



module.exports = {
    index,
    split,
    joinVia,
    addAttribute,
    groupSplit,
    purgeEmpty,
    cropFirst,
    getUpdated
};