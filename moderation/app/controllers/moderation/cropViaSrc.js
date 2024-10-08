const
    Jimp = require('jimp'),
    config = require("config"),
    path = require("path"),
    via_dir = path.dirname(config.moderation.VIABoxPointsModeration.base_json)
;
//const fs = require("fs");


function centrify(bbox, imgInfo, container) {
    const
        x1 = bbox.x1, y1 = bbox.y1,
        x2 = bbox.x2, y2 = bbox.y2,
        w = x2 - x1, h = y2 - y1
    ;
    console.log(`w: ${w} h: ${h}`)

    if (w>container.width) {
        container.width = w
    }
    if (h>container.height) {
        container.height = h
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

// Функція для малювання лінії за алгоритмом Брезенхема
function drawLine(image, x1, y1, x2, y2, color) {
  const dx = Math.abs(x2 - x1);
  const dy = Math.abs(y2 - y1);
  const sx = (x1 < x2) ? 1 : -1;
  const sy = (y1 < y2) ? 1 : -1;
  let err = dx - dy;

  while (true) {
    // Встановлюємо колір пікселя
    image.setPixelColor(color, x1, y1);

    // Перевіряємо, чи досягли кінця лінії
    if (x1 === x2 && y1 === y2) break;

    const e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x1 += sx;
    }
    if (e2 < dx) {
      err += dx;
      y1 += sy;
    }
  }
}


function write_box(image, xArr, yArr) {
    // Визначаємо колір лінії (синій, у форматі ARGB)
    const blue = Jimp.rgbaToInt(0, 0, 255, 255); // (R, G, B, Alpha)

    // Малюємо лінію між координатами (x1, y1) і (x2, y2)
    drawLine(image, xArr[0], yArr[0], xArr[1], yArr[1], blue);
    drawLine(image, xArr[1], yArr[1], xArr[2], yArr[2], blue);
    drawLine(image, xArr[2], yArr[2], xArr[3], yArr[3], blue);
}

function write_start_point(image, xArr, yArr) {
    // Визначаємо колір лінії (червоний, у форматі ARGB)
    const red = Jimp.rgbaToInt(255, 0, 0, 255); // (R, G, B, Alpha)

    // Малюємо лінію між координатами (x1, y1) і (x2, y2)
    image.setPixelColor(red, xArr[0], yArr[0]);
    if (xArr[0]-1 >= 0) { image.setPixelColor(red, xArr[0]-1, yArr[0]) }
    if (xArr[0]+1 < image.bitmap.width) { image.setPixelColor(red, xArr[0]+1, yArr[0]) }
    if (yArr[0]-1 >= 0) { image.setPixelColor(red, xArr[0], yArr[0]-1) }
    if (yArr[0]+1 < image.bitmap.height) { image.setPixelColor(red, xArr[0], yArr[0]+1) }
}


module.exports = async function(ctx, next) {
    const
        filename = ctx.params.filename,
        xArr = ctx.request.query.x.split(',').map(i=>Math.round(Number(i))),
        yArr = ctx.request.query.y.split(',').map(i=>Math.round(Number(i))),
        bbox = {
            x1: Math.min(...xArr),
            y1: Math.min(...yArr),
            x2: Math.max(...xArr),
            y2: Math.max(...yArr)
        },
        container = {
            width: Number(ctx.request.width || 300),
            height: Number(ctx.request.height || 100),
        }
        //border = ctx.request.query.border || 0,
        image_path = path.join(via_dir, filename),
        image = await Jimp.read(image_path)
    ;
    console.log(`bbox`);
    console.log(bbox);
    console.log(`ctx.request.query`);
    console.log(ctx.request.query);
    //console.log(`Write ${tmp_img}`)
    centrify(bbox, image.bitmap, container)
    write_box(image, xArr, yArr)
    write_start_point(image, xArr, yArr)

    const
        croppedImage = image.crop(container.xOffset, container.yOffset, container.width, container.height),
        buffer = await croppedImage.getBufferAsync(Jimp.MIME_PNG)
    ;
    ctx.set('Content-Type', Jimp.MIME_PNG);
    ctx.body = buffer;
    next();
}

