const Router = require('koa-router'),
    KoaBody = require('koa-body')
;

const
    moderateNPText = require('../controllers/moderation/moderateNPText')

;

const router = new Router(),
    koaBody = KoaBody();

router
    .post('/regionOCRModeration', koaBody, moderateNPText)
;

module.exports = {
    routes: function routes () { return router.routes() },
    allowedMethods: function allowedMethods () { return router.allowedMethods() }
};
