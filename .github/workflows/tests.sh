#!/usr/bin/bash
# RUN: . .github/workflows/tests.sh

# test python inference examples
python examples/py/inference/get-started-demo.py
python examples/py/inference/get-started-tiny-demo.py
python examples/py/inference/number-plate-filling-demo.py
python examples/py/inference/number-plate-recognition-multiline-demo.py

# test jupyter inference examples
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/inference/custom-options-model-demo.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/inference/get-started-demo.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/inference/get-started-tiny-demo.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/inference/number-plate-bbox-filling.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/inference/number-plate-keypoints-filling.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/inference/number-plate-recognition-multiline-demo.ipynb

# test python benchmarks examples
python examples/py/benchmark/accuracy-test.py
python examples/py/benchmark/runtime-test.py

# test jupyter benchmarks examples
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/benchmark/accuracy-test.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/benchmark/accuracy-test-custom.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/benchmark/accuracy-test-multiline.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/benchmark/runtime-test.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/benchmark/runtime-test-multiline.ipynb

# test jupyter dataset tools examples
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/dataset_tools/analyze_via_dataset.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/dataset_tools/auto_number_grab.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/dataset_tools/check_ocr_model.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/dataset_tools/option_checker.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=6000 --execute --to html examples/ju/dataset_tools/ocr_dataset_format_checker.ipynb

# test image loaders
python3 nomeroff_net/image_loaders/base.py
python3 -m nomeroff_net.image_loaders.dumpy_loader -f nomeroff_net/image_loaders/dumpy_loader.py
python3 -m nomeroff_net.image_loaders.opencv_loader -f nomeroff_net/image_loaders/opencv_loader.py
python3 -m nomeroff_net.image_loaders.pillow_loader -f nomeroff_net/image_loaders/pillow_loader.py
python3 -m nomeroff_net.image_loaders.turbo_loader -f nomeroff_net/image_loaders/turbo_loader.py

# test nnmodels
python3 nomeroff_net/nnmodels/numberplate_classification_model.py
python3 -m nomeroff_net.nnmodels.numberplate_options_model -f nomeroff_net/nnmodels/numberplate_options_model.py
python3 -m nomeroff_net.nnmodels.fraud_numberpate_options -f nomeroff_net/nnmodels/fraud_numberpate_options.py
python3 -m nomeroff_net.nnmodels.numberplate_inverse_model -f nomeroff_net/nnmodels/numberplate_inverse_model.py
python3 -m nomeroff_net.nnmodels.numberplate_orientation_model -f nomeroff_net/nnmodels/numberplate_orientation_model.py
python3 -m nomeroff_net.nnmodels.ocr_model -f nomeroff_net/nnmodels/ocr_model.py

# test tools
python3 nomeroff_net/tools/test_tools.py

# test ocrs
# TODO: make runnable for GPU
python3 -m nomeroff_net.text_detectors.am -f nomeroff_net/text_detectors/am.py
python3 -m nomeroff_net.text_detectors.by -f nomeroff_net/text_detectors/by.py
python3 -m nomeroff_net.text_detectors.eu -f nomeroff_net/text_detectors/eu.py
python3 -m nomeroff_net.text_detectors.eu_ua_1995 -f nomeroff_net/text_detectors/eu_ua_1995.py
python3 -m nomeroff_net.text_detectors.eu_ua_2004_2015 -f nomeroff_net/text_detectors/eu_ua_2004_2015.py
python3 -m nomeroff_net.text_detectors.ge -f nomeroff_net/text_detectors/ge.py
python3 -m nomeroff_net.text_detectors.kg -f nomeroff_net/text_detectors/kg.py
python3 -m nomeroff_net.text_detectors.kz -f nomeroff_net/text_detectors/kz.py
python3 -m nomeroff_net.text_detectors.ru -f nomeroff_net/text_detectors/ru.py
python3 -m nomeroff_net.text_detectors.su -f nomeroff_net/text_detectors/su.py

# test text postprocessing
python3 -m nomeroff_net.text_postprocessings.eu_ua_2015_squire -f nomeroff_net/text_postprocessings/eu_ua_2015_squire.py
python3 -m nomeroff_net.text_postprocessings.eu_ua_2004_squire -f nomeroff_net/text_postprocessings/eu_ua_2004_squire.py
python3 -m nomeroff_net.text_postprocessings.eu_ua_2004 -f nomeroff_net/text_postprocessings/eu_ua_2004.py
python3 -m nomeroff_net.text_postprocessings.eu_ua_2015 -f nomeroff_net/text_postprocessings/eu_ua_2015.py
python3 -m nomeroff_net.text_postprocessings.eu_ua_1995 -f nomeroff_net/text_postprocessings/eu_ua_1995.py
python3 -m nomeroff_net.text_postprocessings.xx_xx -f nomeroff_net/text_postprocessings/xx_xx.py
python3 -m nomeroff_net.text_postprocessings.ge -f nomeroff_net/text_postprocessings/ge.py