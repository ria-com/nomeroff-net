const checkDir = require('./checkDir'),
      path = require('path')
;

module.exports = (dir_name, subDirArr = [], isCreate = false) => {
    checkDir(dir_name, isCreate);
    for (let subDir of subDirArr) {
        checkDir(path.join(dir_name, subDir), isCreate);
    }
};