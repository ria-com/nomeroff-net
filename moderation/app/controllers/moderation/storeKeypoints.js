const
    config = require("config"),
    fs = require("fs"),
    path = require("path"),
    mathjs = require('mathjs'),
    util = require('util'),
    exec = util.promisify(require('child_process').exec)
;

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
        rotate = ctx.request.body.rotate,
        //zone = require(anb_json)
        zone = JSON.parse(fs.readFileSync(anb_json))
    ;
    let add_params = ''

    console.log("Request body:");
    console.log(JSON.stringify(ctx.request.body));
    //console.log(JSON.stringify(zone, null, 4))
    zone.regions[ctx.request.body.key].keypoints = ctx.request.body.keypoints;
    if (rotate) {
        zone.regions[ctx.request.body.key].rotate = rotate
    }
    console.log("Checkpoint 1 done");
    //zone.regions[ctx.request.body.key].bbox = buildBbox(ctx.request.body.keypoints);
    zone.regions[ctx.request.body.key].updated = true;
    console.log("Checkpoint 2 done");
    //console.log(JSON.stringify(zone, null, 4))

    // Update lines
    if (ctx.request.body.lines != undefined) {
        let lines = {}, cnt =0
        for (const line of ctx.request.body.lines.split(/\n/)) {
            lines[cnt] = line.trim();
            cnt++;
            if (cnt == 3) break;
        }
        add_params = ' -update_lines'
        console.log(`Update lines from ${JSON.stringify(lines)}`);
        zone.regions[ctx.request.body.key].lines = lines
    }

    fs.writeFileSync(anb_json, JSON.stringify(zone, null, 2));
    console.log("Checkpoint 3 done");

    // Rebuild lines
    let cmd = `cd bin; ./rebuild_nn_dataset_image.py -anb_key ${ctx.request.body.basename} -dataset_dir ${base_dir} -rotate ${rotate} -debug${add_params}`
    console.log(`${cmd}`)
    const { stdout, stderr } = await exec(cmd)
    // rebuild_nn_dataset_image.py -anb_key p14955810 -dataset_dir /mnt/sdd1/datasets/2-3lines-test

    console.log("Checkpoint 4 done");

    ctx.body = {
        err_code: 0,
        message: "New keypoints was stored!",
        cmd,
        stdout,
        stderr
    }
    next();
}