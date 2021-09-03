const argv = require('optimist')
        .usage('Usage: $0 --section [string] [--action [string]] [--opt [object]]')
        .demand(['section'])
        .options('action', {
            'default' : 'index'
        })
        .options('opt', {
            alias : 'options',
            'default' : {},
            description : 'example --opt.app=mobile --opt.s=1'
        })
        .argv;

const controllers  = require(`./controllers/${argv.section}Controller`);

(async () => {
    await controllers[argv.action](argv.opt);
})();
