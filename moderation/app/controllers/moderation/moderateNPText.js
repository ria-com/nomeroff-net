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
        data_anb = require(anb_path),
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
        data_anb = require(anb_path)
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
            anb_data[anb_id] = require(anb_path)
        }
    }
    return anb_data
}

module.exports = function(ctx, next) {
    const base_dir = config.moderation.regionOCRModeration.base_dir;
    const max_files_count = ctx.request.body.max_count || 100;
    const changed_numbers = ctx.request.body.changed_numbers;

    const img = path.join(base_dir, config.dataset.img.dir),
          ann = path.join(base_dir, config.dataset.ann.dir),
          anb = path.join(base_dir, config.dataset.anb.dir),
          box = path.join(base_dir, config.dataset.box.dir),
          src = path.join(base_dir, config.dataset.src.dir)
    ;


    let template = Object.assign({}, config.moderation.template);

    // checkers
    if (!fs.existsSync(base_dir)) {
        ctx.body = {
            message: `Path to '${base_dir}' not exists`
        };
    }
    if (!fs.existsSync(img)) {
        fs.mkdirSync(img);
        ctx.body = {
            message: `Image dir '${img}' empty`
        };
    }

    if (!ctx.body) {
        if (!fs.existsSync(ann)) {
            fs.mkdirSync(ann);
        }
        console.log('Change annotations start')
        change_annotation({img, ann, anb, box, src, changed_numbers, template});

        console.log(template);
        const files = fs.readdirSync(img);
        const {res, count} = packRes(files, ann, template)
        let data = res.slice(0, max_files_count)
        ctx.body = {
            expectModeration: files.length - count,
            data,
            anbIdx: loadAnbDesc(data),
            options: config.moderation.regionOCRModeration.options,
            user: template.moderation.moderatedBy || "defaultUser"
        };
        console.log("___________________");
    }

    next();
};