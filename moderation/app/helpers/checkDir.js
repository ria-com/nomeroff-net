const fs = require('fs');

module.exports = (dir_name, isCreate = false) => {
    if (!fs.existsSync(dir_name)) {
        if (isCreate) {
            try {
                fs.mkdirSync(dir_name)
            } catch (e) {
                throw new Error(`Cannot create path "${dir_name}"`)
            }
        } else {
            throw new Error(`Incorrect path ${dir_name}`)
        }
    }
}