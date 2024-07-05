const
    config  = require('config'),
    Router = require('koa-router'),
    KoaBody = require('koa-body'),
    views = require('@ladjs/koa-views'),
    render = views(config.get('koa_view.template.dir'), config.get('koa_view.template.options')),


    moderateNPText = require('../controllers/moderation/moderateNPText'),
    editKeypoints = require('../controllers/moderation/editKeypoints'),
    cropSrc = require('../controllers/moderation/cropSrc'),
    storeKeypoints = require('../controllers/moderation/storeKeypoints')
;

const router = new Router(),
    koaBody = KoaBody();

router
    .post('/regionOCRModeration', koaBody, moderateNPText)
    .get('/editKeypoints/:key', render, editKeypoints)
    .get('/cropSrc/:key', koaBody, cropSrc)
    .post('/storeKeypoints', koaBody, storeKeypoints)
;

module.exports = {
    routes: function routes () { return router.routes() },
    allowedMethods: function allowedMethods () { return router.allowedMethods() }
};
