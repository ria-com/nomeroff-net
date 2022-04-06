const assert = require('assert');
const {
    index,
    createAnnotations,
    moveChecked,
    dataSplit,
    moveGarbage,
    dataJoin,
    moveSomething,
    moveDupes,
} = require("./app_console/controllers/defaultController")

describe("defaultController", function() {

  it("test index", async function() {
    await index()
  });

  it("test createAnnotations", async function() {
    const options = {
      "baseDir": "../data/dataset/TextDetector/ocr_example2/train/",
      "imageDir": "../data/examples/numberplate_zone_images/",
    }
    await createAnnotations(options)
  });

  it("test moveChecked", async function() {
    const options = {
      "baseDir": "../data/dataset/TextDetector/ocr_example2/train/",
      "targetDir": "../data/dataset/TextDetector/ocr_example2/val/",
    }
    await moveChecked(options)
  });

  it("test dataSplit", async function() {
    const options = {
      "rate": 0.5,
      "srcDir": "../data/dataset/TextDetector/ocr_example2/train/",
      "targetDir": "../data/dataset/TextDetector/ocr_example2/val/",
      "test": 1
    }
    await dataSplit(options)
  });

  it("test moveGarbage", async function() {
    const options = {
      "srcDir": "../data/dataset/TextDetector/ocr_example2/train/",
      "targetDir": "../data/dataset/TextDetector/ocr_example2/garbage/",
    }
    await moveGarbage(options)
  });

  it("test dataJoin", async function() {
    const options = {
      "srcDir": [
          "../data/dataset/TextDetector/ocr_example2/train/",
          "../data/dataset/TextDetector/ocr_example2/val/",
      ],
      "targetDir": "../data/dataset/TextDetector/ocr_example2/train/",
    }
    await dataJoin(options)
  });

  it("test moveSomething", async function() {
    const options = {
      "srcDir": "../data/dataset/TextDetector/ocr_example2/train/",
      "targetDir": "../data/dataset/TextDetector/ocr_example2/train.true/",
    }
    await moveSomething(options)
  });

  it("test moveDupes", async function() {
    const options = {
      "srcDir": "../data/dataset/TextDetector/ocr_example2/train/",
      "targetDir": "../data/dataset/TextDetector/ocr_example2/train.dupes/",
    }
    await moveDupes(options)
  });

});