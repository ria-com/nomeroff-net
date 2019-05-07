const fs = require('fs');

module.exports = dirname => { if (!fs.existsSync(dirname)) throw new Error(`Incorrect path ${dirname}`) }