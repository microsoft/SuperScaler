# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR
wget -c https://github.com/tensorflow/models/archive/v2.3.0.zip -O models.zip
unzip -n models.zip
rm models.zip
patch -p0 < tf_models.patch
python3 models-2.3.0/research/slim/setup.py install
models_dir=`pwd`"/models-2.3.0/research/slim"
wrapper_dir="$models_dir/nets/superscaler_wrapper.py"
echo "Add $models_dir in your PYTHONPATH"
echo "  use command: export PYTHONPATH=\$PYTHONPATH:$models_dir"
echo "  see: $wrapper_dir for getting more models"