latest_models = {
  "Detector": {
    "mrcnn": {
      "h5": {
        "gpu": "https://nomeroff.net.ua/models/mrcnn/mask_rcnn_numberplate_0640_2019_06_24.h5",
        "cpu": "https://nomeroff.net.ua/models/mrcnn/mask_rcnn_numberplate_0640_2019_06_24.h5"
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
        "gpu": "https://nomeroff.net.ua/models/options/0.3.x/numberplate_options_2019_06_27.h5",
        "cpu": "https://nomeroff.net.ua/models/options/0.3.x/numberplate_options_2019_06_27.h5"
      }
    }
  },
  "TextDetector": {
    "eu_ua_2004_2015": {
      "h5": {
        "gpu": "https://nomeroff.net.ua/models/ocr/ua/anpr_ocr_ua_17-gpu.h5",
        "cpu": "https://nomeroff.net.ua/models/ocr/ua/anpr_ocr_ua_17-cpu.h5"
      }
    },
    "eu_ua_1995": {
      "h5": {
        "gpu": "https://nomeroff.net.ua/models/ocr/ua-1995/anpr_ocr_ua-1995_2-gpu.h5",
        "cpu": "https://nomeroff.net.ua/models/ocr/ua-1995/anpr_ocr_ua-1995_2-cpu.h5"
      }
    },
    "eu": {
      "h5": {
        "cpu": "https://nomeroff.net.ua/models/ocr/eu/anpr_ocr_eu_2-cpu.h5",
        "gpu": "https://nomeroff.net.ua/models/ocr/eu/anpr_ocr_eu_2-gpu.h5"
      }
    },
    "ru": {
      "h5": {
        "cpu": "https://nomeroff.net.ua/models/ocr/ru/anpr_ocr_ru_3-cpu.h5",
        "gpu": "https://nomeroff.net.ua/models/ocr/ru/anpr_ocr_ru_3-gpu.h5"
      }
    },
    "kz": {
      "h5": {
        "cpu": "https://nomeroff.net.ua/models/ocr/kz/anpr_ocr_kz_4-cpu.h5",
        "gpu": "https://nomeroff.net.ua/models/ocr/kz/anpr_ocr_kz_4-gpu.h5"
      }
    },
    "ge": {
      "h5": {
        "cpu": "https://nomeroff.net.ua/models/ocr/ge/anpr_ocr_ge_3-cpu.h5",
        "gpu": "https://nomeroff.net.ua/models/ocr/ge/anpr_ocr_ge_3-gpu.h5"
      }
    }
  }
}