#!/usr/bin/bash

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

python3 nomeroff_net/image_loaders/base.py
python3 -m nomeroff_net.image_loaders.dumpy_loader
python3 -m nomeroff_net.image_loaders.opencv_loader
python3 -m nomeroff_net.image_loaders.pillow_loader
python3 -m nomeroff_net.image_loaders.turbo_loader
