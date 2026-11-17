const fs = require('fs'),
    { imageSize } = require('image-size');

/**
 * Get image dimensions by file path.
 * image-size >= 2.x accepts only a buffer, so read the file first.
 * @param {string} filePath
 * @returns {{width: number, height: number, type: string}}
 */
module.exports = function sizeOf(filePath) {
    return imageSize(fs.readFileSync(filePath));
};
