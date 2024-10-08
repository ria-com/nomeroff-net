const Koa = require('koa'),
      config = require('config'),
      path = require('path'),
      logger = require('koa-logger'),
      err = require('./helpers/error'),
      serve = require('koa-static'),
      mount = require('koa-mount')
;


const {routes, allowedMethods} = require('./routes/index');

const app = new Koa();

app.use(err);
app.use(logger());
app.use(serve(config.moderation.regionOCRModeration.base_dir));
if (config.moderation.VIABoxPointsModeration !== undefined) {
    app.use(mount('/preview',serve(config.moderation.VIABoxPointsModeration.boxes_preview_dir)));
    app.use(mount('/via',serve(path.dirname(config.moderation.VIABoxPointsModeration.base_json))));
}
app.use(serve('./public'));
app.use(routes());
app.use(allowedMethods());


app.listen(config.server.port, function () {
    console.log('%s listening at port %d', config.app.name, config.server.port);
});