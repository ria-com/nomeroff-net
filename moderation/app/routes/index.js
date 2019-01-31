const Router = require('koa-router'),
    convert = require('koa-convert'),
    KoaBody = require('koa-body'),
    views   = require('koa-views'),
    config  = require('config')
;

const
    moderateNPText = require('../controllers/moderation/moderateNPText')

;

const router = new Router(),
    koaBody = KoaBody(),
    render = views(config.template.path, config.template.options);

router
    .post('/regionOCRModeration', koaBody, moderateNPText)
;

module.exports = {
    routes: function routes () { return router.routes() },
    allowedMethods: function allowedMethods () { return router.allowedMethods() }
};