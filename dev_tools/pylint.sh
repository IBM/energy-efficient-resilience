#!/usr/bin/env sh
# Copyright 2022 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# EERAI CI support scripts
#

set -e # Finish right after a non-zero return command

if [ "$(which pylint 2> /dev/null)" != "" ]; then
    # shellcheck disable=SC2010,SC2035,SC2046
    pylint --generated-members numpy.*,torch.* --disable C --disable W,R,similarities $(ls ./*.py */*.py */*/*.py | grep -v ^venv)
fi
