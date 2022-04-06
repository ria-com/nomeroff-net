const fs = require('fs');

module.exports = (dirname, isCreate = false) => {
    console.log(dirname, isCreate);
    if (!fs.existsSync(dirname)) {
        if (isCreate) {
            try {
                fs.mkdirSync(dirname, { recursive: true })
            } catch (e) {
                throw new Error(`Cannot create path "${dirname}"`)
            }
        } else {
            throw new Error(`Incorrect path ${dirname}`)
        }
    }
}