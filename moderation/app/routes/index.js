const
    config  = require('config'),
    Router = require('koa-router'),
    KoaBody = require('koa-body'),
    views = require('@ladjs/koa-views'),
    render = views(config.get('koa_view.template.dir'), config.get('koa_view.template.options')),


    moderateNPText = require('../controllers/moderation/moderateNPText'),
    moderateViaBoxPoints = require('../controllers/moderation/moderateViaBoxPoints'),
    editKeypoints = require('../controllers/moderation/editKeypoints'),
    cropSrc = require('../controllers/moderation/cropSrc'),
    cropViaSrc = require('../controllers/moderation/cropViaSrc'),
    storeKeypoints = require('../controllers/moderation/storeKeypoints'),
    showNormalizedBbox = require('../controllers/moderation/showNormalizedBbox')
;

const router = new Router(),
    koaBody = KoaBody();

router
    .post('/regionOCRModeration', koaBody, moderateNPText)
    .get('/editKeypoints/:key', render, editKeypoints)
    .get('/cropSrc/:key', koaBody, cropSrc)
    .get('/cropViaSrc/:filename', koaBody, cropViaSrc)
    .get('/showNormalizedBbox/:filename', koaBody, showNormalizedBbox)
    .post('/storeKeypoints', koaBody, storeKeypoints)

    .post('/VIABoxPointsModeration', koaBody, moderateViaBoxPoints)
;

module.exports = {
    routes: function routes () { return router.routes() },
    allowedMethods: function allowedMethods () { return router.allowedMethods() }
};
