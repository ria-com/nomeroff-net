const
    config = require("config"),
    path = require("path"),
    mathjs = require('mathjs')
;
const fs = require("fs");

function buildBbox(keypoints){
    const
        x1 = mathjs.min(keypoints.map(k=>k[0])),
        y1 = mathjs.min(keypoints.map(k=>k[1])),
        x2 = mathjs.max(keypoints.map(k=>k[0])),
        y2 = mathjs.max(keypoints.map(k=>k[1]))
    return [x1,y1,x2,y2]
}

module.exports = async function(ctx, next) {
    const
        base_dir = config.moderation.regionOCRModeration.base_dir,
        anb_subdir = 'anb',
        src_subdir = 'src',
        anb_dir = path.join(base_dir, anb_subdir),
        src_dir = path.join(base_dir, src_subdir),
        anb_json = path.join(anb_dir, `${ctx.request.body.basename}.json`),
        src_img = path.join(src_dir, ctx.request.body.src),
        zone = require(anb_json)
    ;
    console.log("Request body:");
    console.log(JSON.stringify(ctx.request.body));
    //console.log(JSON.stringify(zone, null, 4))
    zone.regions[ctx.request.body.key].keypoints = ctx.request.body.keypoints;
    //zone.regions[ctx.request.body.key].bbox = buildBbox(ctx.request.body.keypoints);
    zone.regions[ctx.request.body.key].updated = true;
    //console.log(JSON.stringify(zone, null, 4))
    fs.writeFileSync(anb_json, JSON.stringify(zone, null, 2));

    ctx.body = {
        err_code: 0,
        message: "New keypoints was stored!"
    }
    next();
}