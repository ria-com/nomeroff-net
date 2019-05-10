const checkDir = require('./checkDir'),
      path = require('path')
;

module.exports = (dirname, subDirArr = [], isCreate = false) => {
    checkDir(dirname, isCreate);
    for (let subDir of subDirArr) {
        checkDir(path.join(dirname, subDir), isCreate);
    }
};