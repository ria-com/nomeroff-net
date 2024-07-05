const Jimp = require('jimp');
const config = require("config");
const path = require("path");
const sizeOf = require("image-size");
const fs = require("fs");


module.exports = async function(ctx, next) {
    const
        base_dir = config.moderation.regionOCRModeration.base_dir,
        anb_subdir = 'anb',
        src_subdir = 'src',
        tmp_subdir = 'tmp',
        anb_dir = path.join(base_dir, anb_subdir),
        src_dir = path.join(base_dir, src_subdir),
        tmp_dir = path.join(base_dir, tmp_subdir),
        key = ctx.params.key,
        key_arr = key.split('-'),
        anb_basename = key_arr[0],
        zoneId = key_arr[1]+'-'+key_arr[2],
        anb_json = path.join(anb_dir, `${anb_basename}.json`),
        zone = require(anb_json),
        src_img =  path.join(src_dir, zone.src),
        tmp_img =  path.join(tmp_dir, `${anb_basename}-${zoneId}.png`),
        left_top = key_arr[1].split('x'),
        width_height = key_arr[2].split('x'),
        imgInfo = sizeOf(src_img),
        redirectUrl = path.join('/',tmp_subdir, `${anb_basename}-${zoneId}.png`)
    ;
    if (!fs.existsSync(tmp_dir)) {
        fs.mkdirSync(tmp_dir)
    }
    if (!fs.existsSync(tmp_img)) {
        const image = await Jimp.read(src_img);
        image.crop(Number(left_top[0]), Number(left_top[1]), Number(width_height[0]), Number(width_height[1]))
            .write(tmp_img);
    }
    //console.log(`Write ${tmp_img}`)
    ctx.status = 301;
    ctx.redirect(redirectUrl);
    next();
}

