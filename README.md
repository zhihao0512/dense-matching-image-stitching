# dense-matching-image-stitching
### Installation
Download the codes of LoFTR, FlowFormer and this paper.
```
git clone https://github.com/zhihao0512/dense-matching-image-stitching.git
cd dense-matching-image-stitching
git submodule add https://github.com/zju3dv/LoFTR.git LoFTR_module
git submodule add https://github.com/drinkingcoder/FlowFormer-Official.git FlowFormer_module
```
Create the environment
```
pip install -r requirements.txt
```
maxflow_v303.dll and line_python.dll are compiled files of c++ source codes maxflow_v303 and line_python.cpp, respectively. We compiled them on Windows using VS2017 and OpenCV3.4.0. Please download and add opencv_world340.dll in the main folder. You can also compile them by yourself.
