const p = require('../package');
module.exports = {
    app: {
        name: p.name,
        description: p.description,
        version: p.version
    },
    template: {
        path: `${__dirname}/../templates`,
        annDefault: 'kz.json',
        via: {
            dir: 'via',
            tpl_name: 'via_region_data_template.json'
        }
    },
    dataset: {
        baseDir: `${__dirname}/../../data/dataset/TextDetector/ocr_example_res`,
        anb: { dir: 'anb', ext: 'json'},
        ann: { dir: 'ann', ext: 'json'},
        img: { dir: 'img', ext: 'png'},
        box: { dir: 'box', ext: 'png'},
        src: { dir: 'src', ext: 'jpeg'}
    },
    via: {
        partDirs: ['train', 'val']
    },
    koa_view: {
        template: {
            options: {map: {njk: 'nunjucks'}, extension: 'njk'},
            name: 'konvaBox',
            dir: __dirname + '/../nunjucks'
        }
    },
    pages: {
        editKeypoints: {
            container: {
                width: 700,
                height: 500
            }
        }
    }
};
