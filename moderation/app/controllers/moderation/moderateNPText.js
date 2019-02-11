const config  = require('config'),
      path = require("path"),
      fs = require("fs")

;

function chencheAnnotation (img, ann, chended_numbers, who) {

    if (chended_numbers != undefined) {

        for (let k in chended_numbers) {
            const f = chended_numbers[k];
            const number = f.number;
            const newNumber = f.newNumber;

            if (fs.existsSync(path.join(img, `${number}.png`)) && fs.existsSync(path.join(ann, `${number}.json`))) {
                if (Boolean(Number(f.deleted))) {
                    console.log(f.deleted);
                    fs.unlinkSync(path.join(img, `${number}.png`));
                    fs.unlinkSync(path.join(ann, `${number}.json`));
                } else {
                    const data = JSON.parse(fs.readFileSync(path.join(ann, `${number}.json`)));
                    const options = config.moderation.regionOCRModeration.options;
                    for (let key in options) {
                        data[key] = f[key];
                    }
                    data.description = newNumber;
                    data.name = number;
                    data.moderation = {isModerated: 1, moderatedBy:who || "unkdownUser"};
                    //console.log(JSON.stringify(data));
                    fs.writeFileSync(path.join(ann, `${number}.json`), JSON.stringify(data));
                }
            }
        }
    }
}

module.exports = function(ctx, next) {
    const base_dir = config.moderation.regionOCRModeration.base_dir;
    const max_files_count = ctx.request.body.max_count || 100;
    const chended_numbers = ctx.request.body.chended_numbers;
    const who_changed = ctx.request.body.who_changed;

    const img = path.join(base_dir, "/img/");
    const ann = path.join(base_dir, "/ann/");

    chencheAnnotation(img, ann, chended_numbers, who_changed);

    const files = fs.readdirSync(ann);

    const res = [];
    let count = 0;
    let iter = 0;
    for (let i in files) {
        iter++;
        if (count > max_files_count) {
            break;
        }

        const f = files[i];
        const number = f.substring(0, f.length - 5);

        const jsonPath = path.join(ann, `${number}.json`);
        const imgPath = path.join(img, `${number}.png`);

        const data = JSON.parse(fs.readFileSync(jsonPath));

        if (data.moderation == undefined || data.moderation.isModerated != 1) {
            count++;
            const data_item = {
                img_path: `img/${number}.png`,
                name: number,
                description: data.description,
            };
            const options = config.moderation.regionOCRModeration.options;
            for (let key in options) {
                data_item[key] = data[key];
            }
            res.push(data_item)
        }
    }

    ctx.body = {
        expectModeration: files.length - iter,
        data:res,
        options: config.moderation.regionOCRModeration.options,
        user: config.user.name
    };
    next();
};