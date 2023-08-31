const
    config = require("config"),
    path = require("path"),
    fs = require("fs"),
    util = require('util'),
    exec = util.promisify(require('child_process').exec)
;

module.exports = async function(ctx, next) {
    const
        base_dir = config.moderation.regionOCRModeration.base_dir,
        src_name = ctx.request.body.src_name,
        target_img = path.join(base_dir, "/img/", src_name)
    ;
    console.log(`Need to rotate "${target_img}"`)

    try {
        const cmd = `convert -rotate 180 ${target_img} ${target_img}`;
        console.log(`Executing: "${cmd}"`);
        const { stdout, stderr } = await exec(cmd);
        console.log('stdout:', stdout);
        console.log('stderr:', stderr);
        ctx.body = {
            message: `Image "${src_name}" has been successfully rotated 180 degrees`,
            error: 0
        }
    } catch (e) {
        console.error(e); // should contain code (exit code) and signal (that caused the termination).
        ctx.body = {
            error_message: e,
            error: 1
        }
    }
    next();
}