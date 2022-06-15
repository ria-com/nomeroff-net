const config  = require('config'),
    path = require("path"),
    fs = require("fs"),
    sizeOf = require('image-size')

;


function deleteImgAndAnn(img, ann, number, f) {
    console.log(f.deleted);
    fs.unlinkSync(path.join(img, `${number}.png`));
    if (fs.existsSync(path.join(ann, `${number}.json`))) {
        fs.unlinkSync(path.join(ann, `${number}.json`));
    }
}

function changeAnn(img, ann, number, template, newNumber, f) {
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
}

function chenche_annotation (img, ann, chended_numbers, template) {
    if (!chended_numbers) {
        return;
    }
    for (let i in chended_numbers) {
        const f = chended_numbers[i];
        const number = f.number;
        const newNumber = f.newNumber;

        if (fs.existsSync(path.join(img, `${number}.png`))) {
            if (Boolean(Number(f.deleted))) {
                deleteImgAndAnn(img, ann, number, f);
            } else {
                changeAnn(img, ann, number, template, newNumber, f);
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

module.exports = function(ctx, next) {
    const base_dir = config.moderation.regionOCRModeration.base_dir;
    const max_files_count = ctx.request.body.max_count || 100;
    const chended_numbers = ctx.request.body.chended_numbers;

    const img = path.join(base_dir, "/img/");
    const ann = path.join(base_dir, "/ann/");
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
        chenche_annotation(img, ann, chended_numbers, template);

        console.log(template);
        const files = fs.readdirSync(img);
        const {res, count} = packRes(files, ann, template)
        ctx.body = {
            expectModeration: files.length - count,
            data: res.slice(0, max_files_count),
            options: config.moderation.regionOCRModeration.options,
            user: template.moderation.moderatedBy || "defaultUser"
        };
        console.log("___________________");
    }

    next();
};