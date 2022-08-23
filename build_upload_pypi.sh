#!/bin/bash

set -e

if [ -z "$1" ]
    then
        echo "please give the version number"
        exit 1
fi

python setup.py sdist bdist_wheel
python -m twine check dist/finch-clust-$1.tar.gz
python -m twine upload dist/finch-clust-$1.tar.gz




