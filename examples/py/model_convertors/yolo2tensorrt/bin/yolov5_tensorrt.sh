: '
cd ./examples/py/model_convertors/
./yolo2tensorrt/bin/yolov5_tensorrt.sh
'

yolov5s_name="yolov5s-2021-12-14"
pt_ext="pt"
wts_ext="wts"
engine_ext="engine"
yolov5s_model=${yolov5s_name}.${pt_ext}
yolov5s_wts=${yolov5s_name}.${wts_ext}
yolov5s_engine=${yolov5s_name}.${engine_ext}
model_dir="../../../data/models/Detector/yolov5"
CLASS_NUM_VALUE=1
class_num_file="yololayer.h"

echo "Check yolov5s model: ${yolov5s_model}"
if [ -f ${model_dir}/${yolov5s_model} ]; then
  echo "Found..."
else
  echo "download ${yolov5s_model} to ${model_dir}"
  wget https://nomeroff.net.ua/models/object_detection/${yolov5s_model} -O ${model_dir}/${yolov5s_model}
fi

echo "Prepare repositories:"
git clone https://github.com/ultralytics/yolov5.git
git clone https://github.com/wang-xinyu/tensorrtx.git


echo "Prepare gen_wts.py:"
if [ -f "./yolov5/gen_wts.py" ]; then
  echo "gen_wts.py detected"
else
  cp ./tensorrtx/yolov5/gen_wts.py ./yolov5
fi

if [ -f ./yolov5/${yolov5s_model} ]; then
  echo "${yolov5s_model} detected"
else
  cp ${model_dir}/${yolov5s_model} ./yolov5
fi

cd ./yolov5

if [ -f ${yolov5s_wts} ]; then
  echo "${yolov5s_wts} detected"
else
  echo "python3 gen_wts.py -w ${yolov5s_model} -o ${yolov5s_wts}"
  python3 gen_wts.py -w ${yolov5s_model} -o ${yolov5s_wts}
fi

cd ../tensorrtx/yolov5/
echo "Apply patch for ${class_num_file}"
perl -e 's/CLASS_NUM = 80/CLASS_NUM = 1/g' -pi ${class_num_file}

mkdir build
cd build
cp ../../../yolov5/${yolov5s_wts} ./

cmake ..
make

if [ -f ${yolov5s_engine} ]; then
  echo "${yolov5s_engine} detected"
else
  echo "./yolov5 -s ${yolov5s_wts} ${yolov5s_engine} s"
  ./yolov5 -s ${yolov5s_wts} ${yolov5s_engine} s
fi

cp ${yolov5s_engine} libmyplugins.so ../../..
