latest_models = {
  "Detector": {
    "mrcnn": {
      "h5": {
        "gpu": "https://nomeroff.net.ua/models/mrcnn/mask_rcnn_numberplate_1000_2019_10_07.h5",
        "cpu": "https://nomeroff.net.ua/models/mrcnn/mask_rcnn_numberplate_1000_2019_10_07.h5"
      }
    }
  },
  "OptionsDetector": {
    "simple": {
      "h5": {
        "class_region": [
          "xx_unknown",
          "eu_ua_2015",
          "eu_ua_2004",
          "eu_ua_1995",
          "eu",
          "xx_transit",
          "ru",
          "kz",
          "eu-ua-fake-dnr",
          "eu-ua-fake-lnr",
          "ge"
        ],
        "gpu": "https://nomeroff.net.ua/models/options/0.4.x/numberplate_options_2020_08_20_tensorflow_v2.h5",
        "cpu": "https://nomeroff.net.ua/models/options/0.4.x/numberplate_options_2020_08_20_tensorflow_v2.h5"
      }
    }
  },
  "TextDetector": {
    "eu_ua_2004_2015": {
      "h5": {
        "gpu": "https://nomeroff.net.ua/models/ocr/ua/tf2/anpr_ocr_ua_2020_08_21_tensorflow_v2.h5",
        "cpu": "https://nomeroff.net.ua/models/ocr/ua/tf2/anpr_ocr_ua_2020_08_21_tensorflow_v2.h5"
      }
    },
    "eu_ua_1995": {
      "h5": {
        "gpu": "https://nomeroff.net.ua/models/ocr/ua-1995/tf2/anpr_ocr_ua-1995_2020_08_26_tensorflow_v2.h5",
        "cpu": "https://nomeroff.net.ua/models/ocr/ua-1995/tf2/anpr_ocr_ua-1995_2020_08_26_tensorflow_v2.h5"
      }
    },
    "eu": {
      "h5": {
        "cpu": "https://nomeroff.net.ua/models/ocr/eu/tf2/anpr_ocr_eu_2020_08_26_tensorflow_v2.h5",
        "gpu": "https://nomeroff.net.ua/models/ocr/eu/tf2/anpr_ocr_eu_2020_08_26_tensorflow_v2.h5"
      }
    },
    "ru": {
      "h5": {
        "cpu": "https://nomeroff.net.ua/models/ocr/ru/tf2/anpr_ocr_ru_2020_08_26_tensorflow_v2.h5",
        "gpu": "https://nomeroff.net.ua/models/ocr/ru/tf2/anpr_ocr_ru_2020_08_26_tensorflow_v2.h5"
      }
    },
    "kz": {
      "h5": {
        "cpu": "https://nomeroff.net.ua/models/ocr/kz/tf2/anpr_ocr_kz_2020_08_26_tensorflow_v2.h5",
        "gpu": "https://nomeroff.net.ua/models/ocr/kz/tf2/anpr_ocr_kz_2020_08_26_tensorflow_v2.h5"
      }
    },
    "ge": {
      "h5": {
        "cpu": "https://nomeroff.net.ua/models/ocr/ge/tf2/anpr_ocr_ge_2020_08_21_tensorflow_v2.h5",
        "gpu": "https://nomeroff.net.ua/models/ocr/ge/tf2/anpr_ocr_ge_2020_08_21_tensorflow_v2.h5"
      }
    }
  }
}
