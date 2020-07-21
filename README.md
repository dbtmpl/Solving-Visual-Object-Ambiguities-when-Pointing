# Solving visual object ambiguities when pointing: an unsupervised learning approach
[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB.svg?logo=python)](https://www.python.org/) [![NumPy 1.18.5](https://img.shields.io/badge/NumPy-1.18.5-blue)](https://numpy.org/doc/1.18/) [![OpenCV 4.3.0](https://img.shields.io/badge/OpenCV-4.3.0-brightgreen)](https://docs.opencv.org/4.3.0/) [![MIT](https://img.shields.io/badge/License-MIT-3DA639.svg?logo=open-source-initiative)](LICENSE)

This repository provides the official implementation of the paper:
> **[Solving visual object ambiguities when pointing: an unsupervised learning approach](https://link.springer.com/article/10.1007/s00521-020-05109-w) (Neural Computing and Applications 2020)**<br>
> *[Doreen Jirak](https://scholar.google.com/citations?user=-HgMDDYAAAAJ&hl), †[David Biertimpel](https://www.linkedin.com/in/david-biertimpel/), ‡[Matthias Kerzel](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/people/kerzel.html) and ‡[Stefan Wermter](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/people/wermter.html) <br>
> *Istituto Italiano di Tecnologia, †University of Amsterdam, ‡University of Hamburg<br>
> pre-print : https://arxiv.org/abs/1912.06449

<img width="80%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/scenario_overview.png">

## Abstract
Whenever we are addressing a specific object or refer to a certain spatial location, we are using referential or deictic gestures usually accompanied by some verbal description. Particularly, pointing gestures are necessary to dissolve ambiguities in a scene and they are of crucial importance when verbal communication may fail due to environmental conditions or when two persons simply do not speak the same language. With the currently increasing advances of humanoid robots and their future integration in domestic domains, the development of gesture interfaces complementing human–robot interaction scenarios is of substantial interest. The implementation of an intuitive gesture scenario is still challenging because both the pointing intention and the corresponding object have to be correctly recognized in real time. The demand increases when considering pointing gestures in a cluttered environment, as is the case in households. Also, humans perform pointing in many different ways and those variations have to be captured. Research in this field often proposes a set of geometrical computations which do not scale well with the number of gestures and objects and use specific markers or a predefined set of pointing directions. In this paper, we propose an unsupervised learning approach to model the distribution of pointing gestures using a growing-when-required (GWR) network. We introduce an interaction scenario with a humanoid robot and define the so-called ambiguity classes. Our implementation for the hand and object detection is independent of any markers or skeleton models; thus, it can be easily reproduced. Our evaluation comparing a baseline computer vision approach with our GWR model shows that the pointing-object association is well learned even in cases of ambiguities resulting from close object proximity.

## Usage
The `demo.py` comes with only few parameters:
```Parameters
--gwr-model             Path to the GWR model. Not used when using pointing-array for prediction.
--skin-model            Path to the skin-color model used for hand detection.
--demo-video            Path to the demo video.
--use-pointing-array    If set, the pointing array approach is used. By default, the GWR network is used.
```

The Default parameters put in place allow running `python demo.py` for the GWR- and `python demo.py --use-pointing-array` for the pointing-array based approach. A demo run with all parameters specified looks as follows:
```Example-run
python demo.py --gwr-model "results/gwr_based_approach/gwr_models_and_results/normalized_for_demo_90_30e/" \
               --skin-model "resources/skin_color_segmentation/saved_histograms/skin_probabilities_crcb.npy" \
               --demo-video "resources/test_videos/amb1_o3_r1_m.webm" \
               --use-pointing-array             
```

Below an impression how the run after executing the commands should look like (left: GWR based pointing, right: pointing with a poining-array).
<img align="left" width="46%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/simple_scene_gwr_pointing_yellow.jpg">
<img  width="46%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/simple_scene_cv_pointing_green.jpg">

## Dependencies
The deictic gesture recognition is entirely based on NumPy, OpenCV and occasional SciPy functions. The full dependencies can be viewed in the `environment.yml`.

## <a name="Citing SVOAWP"></a> Citation
For citing our paper please use the following BibTeX entry:
```BibTeX
@Article{Jirak2020,
author={Jirak, Doreen
and Biertimpel, David
and Kerzel, Matthias
and Wermter, Stefan},
title={Solving visual object ambiguities when pointing: an unsupervised learning approach},
journal={Neural Computing and Applications},
year={2020},
month={Jun},
day={30},
issn={1433-3058},
doi={10.1007/s00521-020-05109-w},
url={https://doi.org/10.1007/s00521-020-05109-w}
}
```

## Further Visualizations
### Unambiguous

<img align="left" width="46%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/simple_scene_gwr_pointing_red.png">
<img  width="46%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/simple_scene_gwr_pointing_green.png">

---

## Ambiguity detection
<img align="left" width="44%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/un_amb_1.jpg">
<img  width="44%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/un_amb_2.jpg">
<img align="left" width="44%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/amb_1.png">
<img  width="44%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/amb_2.jpg">
<img width="80%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/union_of_bbs.png">

---

Note: The code this work is based on was developed during my bachelor thesis in summer 2018.
