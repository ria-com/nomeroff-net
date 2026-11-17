const config = require('config'),
    path = require('path');

/**
 * Settings of the optional VIA box points moderation section
 * (config.moderation.VIABoxPointsModeration).
 * Read lazily, so the OCR moderation works without this section.
 */
module.exports = function viaConfig() {
    const via = config.moderation.VIABoxPointsModeration;
    if (via === undefined || !via.base_json) {
        throw new Error('config.moderation.VIABoxPointsModeration.base_json is not set (see config/example.js)');
    }
    return {
        via_json: via.base_json,
        via_dir: path.dirname(via.base_json),
        boxes_preview_dir: via.boxes_preview_dir
    };
};
