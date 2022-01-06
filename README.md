# petfinder

### setup develop enviroment

```sh
make develop
```

### install cuml on colab

copy and paste

[1]

```
%sh
git clone https://github.com/rapidsai/rapidsai-csp-utils.git
bash rapidsai-csp-utils/colab/rapids-colab.sh stable

python rapidsai-csp-utils/colab/env-check.py
```

[2]
```
!bash rapidsai-csp-utils/colab/update_gcc.sh
import os
os._exit(00)
```

[3]
```
import condacolab
condacolab.install()
```

[4]
```
!python rapidsai-csp-utils/colab/install_rapids.py nightly
import os
os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'
os.environ['CONDA_PREFIX'] = '/usr/local'

import cuml

print(cuml.__version__)
```