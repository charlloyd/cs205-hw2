#
#  Copyright 2017 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

include make.common
BIN=x.Q3
NVCC=gcc
NVOPTS=-O3 -m32 -DDEBUG

$(BIN): testCase.o
	$(NVCC) $(NVOPTS) -o $(BIN) testCase.o -lcublas

testCase.o: testCase.cu
	$(NVCC) $(NVOPTS) -c testCase.cu 

clean:
	rm -rf testCase.o $(BIN)
