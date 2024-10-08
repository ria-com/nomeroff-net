const config  = require('config'),
    path = require("path"),
    fs = require("fs"),
    sizeOf = require('image-size')
;


function deleteImgAndAnn({img, ann, anb, box, src, number, f}) {
    //console.log(f.deleted);
    fs.unlinkSync(path.join(img, `${number}.png`));
    //console.log(`Unlink ${path.join(img, number+".png")}`);
    if (fs.existsSync(path.join(ann, `${number}.json`))) {
        fs.unlinkSync(path.join(ann, `${number}.json`));
        //console.log(`Unlink ${path.join(ann, number+".json")}`);
    }
    //fs.unlinkSync(path.join(img, `${number}.png`));

    //console.log(`Check other lines`);
    // Check other lines
    const
        box_basename = number.replace(/-line-\d/,""),
        photo_id = box_basename.split('-')[0],
        anb_name = `${photo_id}.${config.dataset.anb.ext}`,
        anb_path = path.join(anb, anb_name),
        data_anb =  JSON.parse(fs.readFileSync(anb_path)),
        src_name = data_anb.src,
        box_name = `${box_basename}.${config.dataset.box.ext}`,
        box_path = path.join(box, box_name),
        src_path = path.join(src, src_name)
    ;
    //console.log(`data_anb`);
    //console.log(data_anb);
    let isHaveAnotherLine = false;
    for (const regionKey in data_anb.regions) {
        //console.log(`regionKey: ${regionKey}`);
        for (const i in data_anb.regions[regionKey].lines) {
            //console.log(`i: ${i}`);
            const checked_ann_number = `${regionKey}-line-${i}.${config.dataset.ann.ext}`
            //console.log(`checked_ann_number: ${checked_ann_number}`);
            if (fs.existsSync(path.join(ann, checked_ann_number))) {
                isHaveAnotherLine = true;
            }
        }
    }
    //console.log(`isHaveAnotherLine ${isHaveAnotherLine}`);
    if (!isHaveAnotherLine) {
        //console.log(`unlink "${anb_path}"`);
        fs.unlinkSync(anb_path);
        //console.log(`unlink "${box_path}"`);
        fs.unlinkSync(box_path);
        //console.log(`unlink "${src_path}"`);
        fs.unlinkSync(src_path);
    }
}

function changeAnn({img, ann, number, anb, template, newNumber, f}) {
    const
        box_basename = number.replace(/-line-\d/,""),
        lineId = number.split('-line-')[1],
        photo_id = box_basename.split('-')[0],
        anb_name = `${photo_id}.${config.dataset.anb.ext}`,
        anb_path = path.join(anb, anb_name),
        data_anb = JSON.parse(fs.readFileSync(anb_path))
    ;
    console.log(`data_anb`);
    console.log(data_anb);
    const dimensions = sizeOf(path.join(img, `${number}.png`));
    let data;
    if(fs.existsSync(path.join(ann, `${number}.json`))) {
        data = Object.assign({}, template, JSON.parse(fs.readFileSync(path.join(ann, `${number}.json`))));
    } else {
        data = Object.assign({}, template);
    }
    data.size = {
        width: dimensions.width,
        height: dimensions.height
    };
    const options = config.moderation.regionOCRModeration.options;
    for (let key in options) {
        data[key] = f[key];
    }
    data.description = newNumber;
    data.name = number;
    data.moderation = Object.assign({},data.moderation || {}, {isModerated: 1});
    fs.writeFileSync(path.join(ann, `${number}.json`), JSON.stringify(data));

    data_anb.regions[box_basename].lines[lineId] = newNumber
    fs.writeFileSync(anb_path, JSON.stringify(data_anb,null,2));

}

function change_annotation ({img, ann, anb, box, src, changed_numbers, template}) {
    if (!changed_numbers) {
        return;
    }
    for (let i in changed_numbers) {
        const f = changed_numbers[i];
        const number = f.number;
        const newNumber = f.newNumber;

        if (fs.existsSync(path.join(img, `${number}.png`))) {
            console.log(`Processing numberplate "${number}"`)
            if (Boolean(Number(f.deleted))) {
                console.log(`deleteImgAndAnn "${number}"`);
                deleteImgAndAnn({img, ann, anb, box, src, number, f});
            } else {
                console.log(`changeAnn "${number}"`);
                changeAnn({img, ann, anb, number, template, newNumber, f});
            }
        }
    }
}

function packRes(files, ann, template) {
    const res = [];
    let count = 0;
    for (let f of files) {
        const number = f.substring(0, f.length - 4);

        const jsonPath = path.join(ann, `${number}.json`);

        let data = {};
        if (!fs.existsSync(jsonPath)) {
            data = template;
        } else {
            data = Object.assign({}, template, JSON.parse(fs.readFileSync(jsonPath)));
        }
        if (!data.moderation || !data.moderation.isModerated) {
            const data_item = {
                img_path: `img/${f}`,
                name: number,
                predicted: data.moderation === undefined ? "" : data.moderation.predicted || "",
                description: data.description,
            };
            console.log(data_item);
            const options = config.moderation.regionOCRModeration.options;
            for (let key in options) {
                data_item[key] = data[key];
            }
            res.push(data_item)
        } else {
            count++;
        }
    }
    return {res, count};
}

function loadAnbDesc(data) {
    const
        base_dir = config.moderation.regionOCRModeration.base_dir,
        anb = "anb",
        anb_dir = path.join(base_dir, anb),
        anb_data = {}
    ;
    const anb_ids = data.map(item=>item.name.split('-')[0]);
    // console.log('anb_ids');
    // console.log(anb_ids);
    for (const anb_id of anb_ids) {
        const
            anb_path = path.join(anb_dir, `${anb_id}.json`)
        ;
        if (fs.existsSync(anb_path)) {
            console.log(`anb_path ${anb_path}`);
            //anb_data[anb_id] = require(anb_path);
            anb_data[anb_id] = JSON.parse(fs.readFileSync(anb_path))
        }
    }
    return anb_data
}

function reshape(arr, idx=0) {
    if (idx == 0) {
        return arr;
    }
    let arr_right = arr.slice(0,idx),
    arr_left = arr.slice(idx);
    return arr_left.concat(arr_right);
}

function update_region_by_xyArr(region, rotate, rotate_new) {
    const
        src_value = rotate || "0",
        value = rotate_new
    ;

    let xArr = region.shape_attributes.all_points_x,
        yArr = region.shape_attributes.all_points_y,
        x = xArr,
        y = yArr
    ;
    if (src_value != "0") {
        if (src_value == "-90") {
            x = reshape(xArr, 1)
            y = reshape(yArr, 1)
        } else if (src_value == "90") {
            x = reshape(xArr, 3)
            y = reshape(yArr, 3)
        } else if (src_value == "180") {
            x = reshape(xArr, 2)
            y = reshape(yArr, 2)
        }
    }

    if (value == "-90") {
        x = reshape(xArr, 3)
        y = reshape(yArr, 3)
    } else if (value == "90") {
        x = reshape(xArr, 1)
        y = reshape(yArr, 1)
    } else if (value == "180") {
        x = reshape(xArr, 2)
        y = reshape(yArr, 2)
    }

    region.shape_attributes.all_points_x = x;
    region.shape_attributes.all_points_y = y;
    region.shape_attributes.rotate = value
}




const compareArrays = (a, b) => {
    console.log(`Ok change_via_items get_region_by_xyArr compareArrays: ${JSON.stringify(a)} and ${JSON.stringify(a)}`);
    return JSON.stringify(a) === JSON.stringify(b);
}

function get_region_by_xyArr(regions, xArr, yArr) {
    console.log(`Ok change_via_items get_region_by_xyArr by xArr: ${xArr}, yArr: ${yArr}`);
    for (const region of regions) {
        console.log(`Ok change_via_items get_region_by_xyArr region.shape_attributes ${region.shape_attributes}`);
        const all_points_x = region.shape_attributes.all_points_x,
              all_points_y =  region.shape_attributes.all_points_y
        ;
        if (compareArrays(xArr, all_points_x) && compareArrays(yArr, all_points_y)) {
            return region;
        }
    }
    return null;
}


//.split(",").map(x => parseInt(x))
function prepare_via_items (changed_via_items) {
    for (let item of changed_via_items) {
        item.all_points_x = item.all_points_x.split(",").map(x => parseInt(x))
        item.all_points_y = item.all_points_y.split(",").map(x => parseInt(x))
    }
    return changed_via_items;
}


function change_via_items ({changed_via_items, via}) {
    if (!changed_via_items) {
        return;
    }
    console.log("Ok change_via_items 1");
    changed_via_items = prepare_via_items(changed_via_items)
    console.log("Ok change_via_items 2");
    for (let i in changed_via_items) {
        console.log(`Ok change_via_items 3 ${i}`);
        const
            item = changed_via_items[i],
            via_item = via['_via_img_metadata'][item.key],
            region = get_region_by_xyArr(via_item.regions, item.all_points_x, item.all_points_y)
        ;
        console.log(`Ok change_via_items 4 ${region}`);
        region.shape_attributes.checked = true;
        if (region !== null) {
            const
                rotate = region.rotate || "0",
                new_rotate = item.rotate || "0"
            ;
            if (rotate != new_rotate) {
                console.log(`Ok change_via_items update rotate with ${new_rotate} (from ${rotate})`);
                update_region_by_xyArr(region, rotate, new_rotate)
            }
            if (Number(item.deleted)) {
                console.log(`Ok change_via_items delete region ${region}`);
                let via_item_regions = [];
                for (const new_region of via_item.regions) {
                    if (new_region != region) {
                        via_item_regions.push(new_region)
                    }
                }
                via_item.regions = via_item_regions;
            }
            console.log(`Ok change_via_items done`);
        }
    }
}



module.exports = function(ctx, next) {
    const
        via_json = config.moderation.VIABoxPointsModeration.base_json,
        via_dir = path.dirname(config.moderation.VIABoxPointsModeration.base_json),
        boxes_preview_dir = config.moderation.VIABoxPointsModeration.boxes_preview_dir
    ;
    const max_files_count = ctx.request.body.max_count || 10;
    const changed_via_items = ctx.request.body.changed_via_items;

    // checkers
    if (!fs.existsSync(via_dir)) {
        ctx.body = {
            message: `Path to '${via_dir}' not exists!`
        };
    }

    if (!fs.existsSync(boxes_preview_dir)) {
        ctx.body = {
            message: `Path to '${boxes_preview_dir}' not exists!`
        };
    }

    if (!fs.existsSync(via_json)) {
        ctx.body = {
            message: `Path to json file '${via_json}' not exists!`
        };
    }

    if (!ctx.body) {
        const
            via = JSON.parse(fs.readFileSync(via_json)),
            items = []
        ;
        if (changed_via_items) {
            console.log('Change via items start')
            console.log('changed_via_items')
            //console.log(JSON.stringify(changed_via_items, null,2))
            console.log(changed_via_items)
            change_via_items({changed_via_items, via});
            console.log('Write via changes!')
            fs.writeFileSync(via_json, JSON.stringify(via,null,4))
        }

        console.log(`Preparing data for moderation`)
        for (const key in via['_via_img_metadata']) {
            console.log(`Preparing key ${key}`)
            let item = via['_via_img_metadata'][key];
            //let regions = []
            for (const region of item.regions) {
                if (!(region["shape_attributes"]["checked"] !== undefined && region["shape_attributes"]["checked"])) {
                    region.key = key;
                    region.filename = item.filename;
                    console.log(`Make item for ${region.filename} and bbox with xArr: ${region.shape_attributes.all_points_x} & ${region.shape_attributes.all_points_y}`)
                    items.push({
                        key,
                        filename: item.filename,
                        all_points_x: region.shape_attributes.all_points_x,
                        all_points_y: region.shape_attributes.all_points_y,
                        label: region.region_attributes.label?region.region_attributes.label:"numberplate"
                    })
                }
            }
        }

        console.log(`Slicing data for moderation`)
        let data = items.slice(0, max_files_count)
        console.log(`Load template`)
        let template = Object.assign({}, config.moderation.template);
        console.log(`return body`)
        ctx.body = {
            expectModeration: items.length,
            data,
            user: template.moderation.moderatedBy || "defaultUser"
        };
        console.log("___________________");
    }

    next();
};