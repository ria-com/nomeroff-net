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
          "eu-ua-fake-dpr",
          "eu-ua-fake-lpr",
          "ge",
          "by",
          "su",
          "kg"
        ],
        "gpu": "https://nomeroff.net.ua/models/options/tf2.3/numberplate_options_2020_12_17_tensorflow_v2_3.h5",
        "cpu": "https://nomeroff.net.ua/models/options/tf2.3/numberplate_options_2020_12_17_tensorflow_v2_3.h5"
      }
    }
  },
  "TextDetector": {
    "eu_ua_2004_2015": {
      "h5": {
        "gpu": "https://nomeroff.net.ua/models/ocr/ua/tf2/anpr_ocr_ua_2020_12_24_tensorflow_v2.h5",
        "cpu": "https://nomeroff.net.ua/models/ocr/ua/tf2/anpr_ocr_ua_2020_12_24_tensorflow_v2.h5"
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
        "cpu": "https://nomeroff.net.ua/models/ocr/eu/tf2/anpr_ocr_eu_2020_10_08_tensorflow_v2.3.h5",
        "gpu": "https://nomeroff.net.ua/models/ocr/eu/tf2/anpr_ocr_eu_2020_10_08_tensorflow_v2.3.h5"
      }
    },
    "ru": {
      "h5": {
        "cpu": "https://nomeroff.net.ua/models/ocr/ru/tf2/anpr_ocr_ru_2020_10_12_tensorflow_v2.3.h5",
        "gpu": "https://nomeroff.net.ua/models/ocr/ru/tf2/anpr_ocr_ru_2020_10_12_tensorflow_v2.3.h5"
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
    },
    "by": {
      "h5": {
        "cpu": "https://nomeroff.net.ua/models/ocr/by/tf2/anpr_ocr_by_2020_10_09_tensorflow_v2.3.h5",
        "gpu": "https://nomeroff.net.ua/models/ocr/by/tf2/anpr_ocr_by_2020_10_09_tensorflow_v2.3.h5"
      }
    },
    "su": {
      "h5":{
        "cpu": "https://nomeroff.net.ua/models/ocr/su/anpr_ocr_su_2020_11_27_tensorflow_v2.3.h5",
        "gpu": "https://nomeroff.net.ua/models/ocr/su/anpr_ocr_su_2020_11_27_tensorflow_v2.3.h5"
      }
    },
    "kg": {
      "h5": {
        "cpu": "https://nomeroff.net.ua/models/ocr/kg/tf2/anpr_ocr_kg_2020_12_31_tensorflow_v2.3.h5",
        "gpu": "https://nomeroff.net.ua/models/ocr/kg/tf2/anpr_ocr_kg_2020_12_31_tensorflow_v2.3.h5"
      }
    }

  }
}
