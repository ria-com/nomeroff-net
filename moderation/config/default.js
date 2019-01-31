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
    template: {
        path: 'app/views',
        options: {
            extension: 'html',
            cache: false
        }
    },
    moderation: {
        regionOCRModeration: {
            base_dir: "/var/www/nomeroff-net/moderation/public/ocr/example/"
        }
    }
};