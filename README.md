# Solving visual object ambiguities when pointing: an unsupervised learning approach
[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB.svg?logo=python)](https://www.python.org/) [![OpenCV 4.3.0](https://img.shields.io/badge/opencv-4.3.0-brightgreen)](https://pytorch.org/docs/1.4.0/) [![MIT](https://img.shields.io/badge/License-MIT-3DA639.svg?logo=open-source-initiative)](LICENSE)


This is the official implementation of the paper  
> **[Solving visual object ambiguities when pointing: an unsupervised learning approach](https://link.springer.com/article/10.1007/s00521-020-05109-w) (Neural Computing and Applications 2020)**<br>
> [Doreen Jirak]*(https://scholar.google.com/citations?user=-HgMDDYAAAAJ&hl), [David Biertimpel]†(https://www.linkedin.com/in/david-biertimpel/), [Matthias Kerzel]‡(https://www.inf.uni-hamburg.de/en/inst/ab/wtm/people/kerzel.html) and [Stefan Wermter]‡(https://www.inf.uni-hamburg.de/en/inst/ab/wtm/people/wermter.html) <br>
> *Istituto Italiano di Tecnologia, †University of Amsterdam, ‡University of Hamburg<br>
> pre-print : https://arxiv.org/abs/1912.06449

## Scenario
<img width="80%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/scenario_overview.png">
<img align="left" width="46%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/simple_scene_gwr_pointing_yellow.jpg">
<img  width="46%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/simple_scene_cv_pointing_green.jpg">


## <a name="Citing SVOAWP"></a> Citation
For citing our paper please use the following BibTeX entry.
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

## Recognition with GWR
### Unambiguous

<img align="left" width="46%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/simple_scene_gwr_pointing_red.png">
<img  width="46%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/simple_scene_gwr_pointing_green.png">

### Ambiguous
<img align="left" width="46%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/ambiguous_scene_gwr_pointing_yellow.jpg">
<img  width="46%"  src=>

---


## Recognition with pointing array
### Unambiguous
<img align="left" width="46%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/simple_scene_naive_pointing_yellow.png">
<img  width="46%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/ambiguous_scene_gwr_pointing_green.jpg">

### Ambiguous
<img align="left" width="46%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/ambiguous_scene_naive_pointing_yellow.png">
<img  width="46%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/ambiguous_scene_naive_pointing_red.png">

---

## Ambiguity detection
<img align="left" width="44%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/un_amb_1.jpg">
<img  width="44%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/un_amb_2.jpg">
<img align="left" width="44%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/amb_1.png">
<img  width="44%"  src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/amb_2.jpg">
<img width="80%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/union_of_bbs.png">

---

Disclaimer: The code on which this work is based was developed during my bachelor thesis in summer 2018.

