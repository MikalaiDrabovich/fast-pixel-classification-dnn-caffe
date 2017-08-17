# About
Fast object recognition of road objects with a deep neural neural network.
You can preview results in real time in GUI, generate video with results or save all frames as images.
Coloring scheme should be compatible with [CityScapes dataset](https://www.cityscapes-dataset.com/).

You just need to copy your own pretrained model and its description to __./models__

To run with GUI you need to [install OpenCV with GUI support](https://stackoverflow.com/questions/36833661/installing-opencv-with-gui-on-ubuntu).

If GUI is not necessary, `pip install opencv-python` may be sufficient.

# How to use
Set paths and parameters in __main__ function, then
```bash
python run_demo.py
```
