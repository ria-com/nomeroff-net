const Router = require('koa-router'),
    KoaBody = require('koa-body')
;

const
    moderateNPText = require('../controllers/moderation/moderateNPText'),
    rotation = require('../controllers/moderation/rotation')
;

const router = new Router(),
    koaBody = KoaBody();

router
    .post('/regionOCRModeration', koaBody, moderateNPText)
    .post('/rotate180degrees', koaBody, rotation)
;

module.exports = {
    routes: function routes () { return router.routes() },
    allowedMethods: function allowedMethods () { return router.allowedMethods() }
};
