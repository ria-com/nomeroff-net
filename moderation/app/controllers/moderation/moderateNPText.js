const config  = require('config'),
      path = require("path"),
      fs = require("fs")
;
const base_dir = config.moderation.regionOCRModeration.base_dir;
const img = path.join(base_dir, "/img/");
const ann = path.join(base_dir, "/ann/");

function chencheAnnotation (img, ann, chended_numbers, who) {
    if (chended_numbers === undefined) {
        return;
    }

    chended_numbers.forEach(function (numberData) {
        let annotationPath = path.join(
            ann,
            numberData.name,
        ) + '.json';
        if (numberData.deleted) {
            if (fs.existsSync(annotationPath)) {
                fs.unlinkSync(annotationPath);
            }
            fs.unlinkSync(path.join(img, numberData.name));
            return;
        }
        Object.assign(numberData, {
            moderation: {isModerated: 1, moderatedBy:who || "unknownUser"}
        });
        fs.writeFileSync(annotationPath, JSON.stringify(numberData));
    });
}

module.exports = function(ctx, next) {
    const max_files_count = ctx.request.body.max_count || 100;
    const chended_numbers = ctx.request.body.chended_numbers;
    const who_changed = ctx.request.body.who_changed;

    if (!fs.existsSync(base_dir)) {
        ctx.body = {
            message: `Path to ${base_dir} not exists`
        };
    } else if (!fs.existsSync(img)) {
        fs.mkdirSync(img);
    } else if (!fs.existsSync(ann)) {
        fs.mkdirSync(ann);
    } else {
        chencheAnnotation(img, ann, chended_numbers, who_changed);

        const files = fs.readdirSync(img);

        const err_data = [];
        const res = [];
        let count = 0;
        let iter = 0;
        for (let i in files) {

            const f = files[i];
            // more reliable way
            const number = path.basename(f, path.extname(f));

            const jsonPath = path.join(ann, `${number}.json`);
            const imgPath = path.join(img, `${number}.png`);

            let data = {};
            if (!fs.existsSync(jsonPath)) {
                data = {
                    description: ""
                }
            } else {
                data = JSON.parse(fs.readFileSync(jsonPath));
            }


            if (data.moderation == undefined || data.moderation.isModerated != 1) {
                err_data.push(data.moderation);
                count++;
                console.log(data.moderation === undefined ? "" : data.moderation.predicted || "");
                const data_item = {
                    img_path: `img/${f}`,
                    name: number,
                    predicted: data.moderation === undefined ? "" : data.moderation.predicted || "",
                    description: data.description,
                };
                const options = config.moderation.regionOCRModeration.options;
                for (let key in options) {
                    data_item[key] = data[key];
                }
                res.push(data_item)
            } else {
                iter++;
            }
        }

        //console.log(iter);
        ctx.body = {
            expectModeration: files.length - iter,
            data:res.slice(0, max_files_count),
            options: config.moderation.regionOCRModeration.options,
            user: config.user.name
        };
    }

    next();
};