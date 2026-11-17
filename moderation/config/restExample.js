const p = require('../package');
module.exports = {
    app: {
        name: p.name,
        description: p.description,
        version: p.version
    },
    server: {
        port: process.env.NODE_APP_INSTANCE || 5005
    },
    dataset: {
        baseDir: `${__dirname}/../../data/dataset/TextDetector/ocr_example/train`,
        ann: { dir: 'ann', ext: 'json'},
        img: { dir: 'img', ext: 'png'},
        // optional, used only by the multi-line dataset format
        // (see nomeroff_net/tools/nn_numberplate_dataset.py)
        anb: { dir: 'anb', ext: 'json'},
        box: { dir: 'box', ext: 'png'},
        src: { dir: 'src', ext: 'jpeg'}
    },
    koa_view: {
        template: {
            options: {map: {njk: 'nunjucks'}, extension: 'njk'},
            name: 'konvaBox',
            dir: `${__dirname}/../nunjucks`
        }
    },
    pages: {
        editKeypoints: {
            container: {
                width: 700,
                height: 500
            }
        }
    },
    via: {
        partDirs: ['train', 'val']
    },
    moderation: {
        regionOCRModeration: {
            base_dir: `${__dirname}/../../data/dataset/TextDetector/ocr_example/train`,
            options: {
                region_id: [
                    "xx-unknown",
                    "eu-ua-2015",
                    "eu-ua-2004",
                    "eu-ua-1995",
                    "eu",
                    "xx-transit",
                    "ru",
                    "kz",
                    "eu-ua-ordlo-dpr",
                    "eu-ua-ordlo-lpr",
                    "ge",
                    "by",
                    "su",
                    "kg",
                    "am",
                    "ua-military",
                    "ru-military",
                    "md",
                    "eu-ua-custom"
                ],
                state_id: ["garbage", "filled", "not filled", "empty"],
                count_lines: ["0", "1", "2", "3"]
            }
        },
        template: {
            tags:[],
            objects:[],
            //"name":"",
            //"description":"",
            state_id:  0,
            region_id: 0,
            size:{
                width:  0,
                height: 0
            },
            moderation:{
                isModerated:0,
                moderatedBy:"dimabendera",
                predicted: ""
            }
        }
    },
};