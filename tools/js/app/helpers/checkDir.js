const fs = require('fs');

module.exports = (dirname, isCreate = false) => {
    if (!fs.existsSync(dirname)) {
        if (isCreate) {
            try {
                fs.mkdirSync(dirname)
            } catch (e) {
                throw new Error(`Cannot create path "${dirname}"`)
            }
        } else {
            throw new Error(`Incorrect path ${dirname}`)
        }
    }
}