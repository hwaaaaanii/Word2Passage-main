# https://github.com/facebookresearch/FiD/blob/main/get-data.sh

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e

ROOT="datasets/data"
DOWNLOAD="$ROOT/download"

mkdir -p "${ROOT}"
mkdir -p "${DOWNLOAD}"

echo "Downloading Wikipedia passages (psgs_w100.tsv.gz)..."
wget -c https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz -P "${DOWNLOAD}"

echo "Decompressing Wikipedia passages..."
gzip -d "${DOWNLOAD}/psgs_w100.tsv.gz"
mv "${DOWNLOAD}/psgs_w100.tsv" "${ROOT}/psgs_w100.tsv"

rm -r "${DOWNLOAD}"

echo "psgs_w100.tsv is ready at $(pwd)/${ROOT}/psgs_w100.tsv"
