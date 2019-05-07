const checkDir = require('./checkDir'),
      path = require('path')
;

module.exports = (dirname, subDirArr = []) => { checkDir(dirname); for (let subDir of subDirArr) { checkDir(path.join(dirname, subDir)) } }