const groupMask = /^(\d+)\-(.+)$/;

function makeGroups(arr) {
    let groupHash = {};
    for(let nameIdx in arr) {
        let name = arr[nameIdx];
        let result = name.match(groupMask);
        if (result !== undefined && result.length) {
            let group = result[1];
            if (groupHash[group] === undefined) {
                groupHash[group] = []
            }
            groupHash[group].push(nameIdx);
        }
    }
    console.log(`groupHash: `);
    console.log(groupHash);
    return groupHash;
}

function clearGroups(targetKeys, targetGroups, groups) {
    let newArr = [],
        toArr = [],
        removedIdsIdx = {}
    ;
    for (let group of groups) {
        for (let i of targetGroups[group]) {
            removedIdsIdx[i] = true;
        }
    }

    for (let i in targetKeys) {
        if (removedIdsIdx[i] !== undefined) {
            toArr.push(targetKeys[i]);
        } else {
            newArr.push(targetKeys[i]);
        }
    }

    return [newArr, toArr];
}


module.exports = function arrayGroupFix(splitRate,partKeys = { 'train': [], 'val': [] }){
    console.log('arrayGroupFix start');
    let   toTrainGroups = [],
          toValGroups = [],
          trainGroups = makeGroups(partKeys.train),
          valGroups = makeGroups(partKeys.val)
    ;

    for (let group in valGroups) {
        if (trainGroups[group] !== undefined) {
            if (Math.random() > splitRate) {
                toTrainGroups.push(group)
            } else {
                toValGroups.push(group)
            }
        }
    }

    console.log(`trainGroups`);
    console.log(trainGroups);

    console.log(`valGroups`);
    console.log(valGroups);

    let newTrain, newVal, toTrain, toVal;
    [newTrain,toVal] = clearGroups(partKeys.val, valGroups, toTrainGroups);
    [newVal,toTrain] = clearGroups(partKeys.train, trainGroups, toValGroups);

    console.log(`----------------toVal [${toVal.length}]`);
    console.log(toVal);

    console.log(`'----------------toTrain [${toTrain.length}]`);
    console.log(toTrain);

    newTrain = newTrain.concat(toTrain);
    newVal = newVal.concat(toVal);

    return {
        'train': newTrain,
        'val': newVal
    };
};