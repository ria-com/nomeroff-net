const
    config  = require('config'),
    path = require("path"),
    fs = require("fs"),
    sizeOf = require("image-size")
;

function centrify(zone, imgInfo, container) {
    console.log('zone')
    console.log(zone)
    const
        x1 = zone.bbox[0], y1 = zone.bbox[1],
        x2 = zone.bbox[2], y2 = zone.bbox[3],
        w = x2 - x1, h = y2 - y1
    ;
    console.log(`w: ${w} h: ${h}`)

    if (w>container.width) {
        container.width = w
    }
    if (h>container.height) {
        container.width = h
    }
    container.w = w;
    container.h = h;


    const
        dW = Math.round((imgInfo.width-container.width)/2), dH = Math.round((imgInfo.height-container.height)/2),
        dX = Math.round((container.width-w)/2), dY = Math.round((container.height-h)/2)
    ;
    console.log(`dX: ${dX} dY: ${dY}`)
    console.log(`imgInfo.width: ${imgInfo.width} imgInfo.height: ${imgInfo.height}`)

    let zoom = 1, zoomW = 1, zoomH = 1;
    if (imgInfo.width<container.width) {
        zoomW = container.width/imgInfo.width
    }
    if (imgInfo.height<container.height) {
        zoomH = container.height/imgInfo.height
    }
    if (zoomW>zoomH) {
        zoom = zoomW
    } else {
        zoom = zoomH
    }
    container.zoom = zoom

    if (x1 <= dX) {
        container.x1 = x1
        container.xOffset = 0
        console.log("dX Stage 1")
    } else if (imgInfo.width <= (x2+dX)) {
        container.x1 = x1-(imgInfo.width-1-container.width)
        container.xOffset = imgInfo.width-1-container.width
        console.log("dX Stage 2")
    } else {
        container.x1 = dX
        container.xOffset = Math.round(x1-(container.width-w)/2)
        console.log("dX Stage 3")
    }

    if (y1 <= dY) {
        container.y1 = y1
        container.yOffset = 0
        console.log("dY Stage 1")
    } else if (imgInfo.height <= (y2+dY)) {
        container.y1 = y1-(imgInfo.height-1-container.height)
        container.yOffset = imgInfo.height-1-container.height
        console.log("dY Stage 2")
    } else {
        container.y1 = dY
        container.yOffset = Math.round(y1-(container.height-h)/2)
        console.log("dY Stage 3")
    }

    return container;
}

function fixContainer(container, imgInfo) {
    if (container.width > imgInfo.width) {
        container.width = imgInfo.width-1
    }
    if (container.height > imgInfo.height) {
        container.height = imgInfo.height-1
    }
    return container;
}

module.exports = async function(ctx, next) {
    const
        base_dir = config.moderation.regionOCRModeration.base_dir,
        anb_subdir = 'anb',
        src_subdir = 'src',
        anb_dir = path.join(base_dir, anb_subdir),
        src_dir = path.join(base_dir, src_subdir),
        key = ctx.params.key,
        key_arr = key.split('-'),
        anb_basename = key_arr[0],
        zoneId = key_arr[1]+'-'+key_arr[2],
        anb_json = path.join(anb_dir, `${anb_basename}.json`),
        zone = JSON.parse(fs.readFileSync(anb_json)),
        src_img =  path.join(src_dir, zone.src),
        imgInfo = sizeOf(src_img),
        lines = Object.values(zone.regions[key].lines).join('\n')
    ;
    let
        container = Object.assign({},config.get('pages.editKeypoints.container'))
    ;
    container = fixContainer(container, imgInfo);
    console.log('container')
    console.log(container)

    console.log('zone')
    console.log(zone)

    console.log('zone.regions')
    console.log(zone.regions)

    console.log('key')
    console.log(key)

    console.log('lines')
    console.log(lines)

    container = centrify(zone.regions[key], imgInfo, container);

    console.log('container')
    console.log(container)
    ctx.body = await ctx.render(config.get('koa_view.template.name'), {
        zone, anb_basename, zoneId, imgInfo, container, key, lines
    });

    ctx.type = 'text/html; charset=utf-8';
    next();
}