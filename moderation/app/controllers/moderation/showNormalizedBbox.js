const
    Jimp = require('jimp'),
    config = require("config"),
    path = require("path"),
    util = require('util'),
    exec = util.promisify(require('child_process').exec),
    via_dir = path.dirname(config.moderation.VIABoxPointsModeration.base_json),
    boxes_preview_dir = path.dirname(config.moderation.VIABoxPointsModeration.boxes_preview_dir)
;
//const fs = require("fs");
function get_bbox_filename(filename,xArr,yArr) {
    const
        basename = filename.split('.')[0],
        minX = Math.round(Math.min(...xArr)),
        minY = Math.round(Math.min(...yArr)),
        maxX = Math.round(Math.max(...xArr)),
        maxY = Math.round(Math.max(...yArr))
    ;
    return `${basename}-${minX}x${minY}-${maxX}x${maxY}.png`
}

module.exports = async function(ctx, next) {
    const
        filename = ctx.params.filename,
        xArr = ctx.request.query.x.split(',').map(i=>Math.round(Number(i))),
        yArr = ctx.request.query.y.split(',').map(i=>Math.round(Number(i))),
        image_path = path.join(via_dir, filename),
        cmd = `cd bin; ./makeNormalizedBbox.py -src_image ${image_path} -dest_dir ${boxes_preview_dir} -x ${ctx.request.query.x} -y ${ctx.request.query.y}`
    ;
    console.log(`${cmd}`)
    const { stdout, stderr } = await exec(cmd)
    console.log("stdout", stdout)
    console.log("stderr", stderr)
    const
        bbox_path = path.join(boxes_preview_dir, get_bbox_filename(filename,xArr,yArr))
        image = await Jimp.read(bbox_path)
        buffer = await image.getBufferAsync(Jimp.MIME_PNG)
    ;
    ctx.set('Content-Type', Jimp.MIME_PNG);
    ctx.body = buffer;
    next();
}

