const p = require('../package');
module.exports = {
    app: {
        name: p.name,
        description: p.description,
        version: p.version
    },
    template: {
        path: `${__dirname}/../templates`,
        annDefault: 'kz.json'
    },
    dataset: {
        baseDir: `${__dirname}/../../data/dataset/TextDetector/ocr_example_res`,
        ann: { dir: 'ann', ext: 'json'},
        img: { dir: 'img', ext: 'png'}
    },
    via: {
        partDirs: ['train', 'val']
    }

};
