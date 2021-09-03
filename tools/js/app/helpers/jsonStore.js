const fs = require('fs');

module.exports = (file, data, space = 2) => {
    return new Promise((resolve, reject) => {
        fs.writeFile(file, JSON.stringify(data,null,space), error => {
            if (error) reject(error);
            resolve(`file "${file}" created successfully!`);
        });
    });
};
