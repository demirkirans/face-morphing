# Face Morphing 

A tool for face morphing. 

**Algorithm:**
1. It finds the facial landmark points in given images using dlib library
2. Then we do Delaunay Triangulation using points found in previous step.
3. Morph the face using these triangles.

![alt text][example]
[example]: https://github.com/demirkirans/face-morphing/blob/main/results/erdkilic.gif



## Requirements

* You need to download [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and extract
* pip install -r requirements.txt
* Set environment variable `PREDICTOR_DATA_DIR` to the folder where `shape_predictor_68_face_landmarks.dat` is located.

## Usage

```
$ git clone https://github.com/demirkirans/face-morphing
```

Specify the paths of source and destination images

```python
python src/morpher.py --src=<source_image_path> --dest=<destination_image_path>
```

You can also enter fps for output video. 

```python
python src/morpher.py --src=<source_image_path> --dest=<destination_image_path> --fps=20
```