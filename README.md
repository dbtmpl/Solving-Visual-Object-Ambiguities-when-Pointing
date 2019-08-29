# Recognizing Deictic Gestures with Unsupervised Learning
## Supervisors and Examiners: [Dr. Doreen Jirak](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/people/jirak.html "University of Hamburg - Website") and [Dr. Matthias Kerzel](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/people/kerzel.html "University of Hamburg - Website")
**Official thesis title: Implementation and Evaluation of a Deictic Gesture Interface with the NICO robot**

In my Bachelor's thesis I implemented an interface for recognizing deictic (pointing) gestures with the [NICO (Neuro-Inspired COmpanion)](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/research/neurobotics/nico.html "University of Hamburg - Website")
robot. In the scenario, the experimenter points to objects and NICO gives feedback on which object is currently targeted. After implementing a *naive* solution using traditional computer-vision based methods, a [Growing When Required (GWR)](https://vision.unipv.it/IA2/aa2008-2009/A%20self-organising%20network%20that%20grows%20when%20required.pdf "Marsland et. al.") network is used to improve accuracy and enhance stability.
 
 
**Abstract:** In everyday interactions, people intuitively reference entities in their environment by pointing at them. These so-called deictic gestures allow directing other people's attention to a desired referent. In the field of Human-Robot-Interaction deictic gesture are of frequent interest as they enable people to apply familiar behavior to shift the robot's focus. However, despite being intuitive, deictic gestures possess an inherent ambiguity. Depending on the perspective and the target's proximity to other entities, the actual target of a deictic gesture may sometimes be difficult to identify, even for a human interaction partner. To this end, this thesis investigates whether we can create a natural deictic gesture interface with the humanoid robot NICO that is capable of recognizing a gesture's target also in ambiguous object constellations. In order to address this task we introduce two approaches: First, we approximate a pointing array from the hand posture of the experimenter. Subsequently, we predict the gesture's target by using a Growing When Required network (GWR). Finally, we create experimental set-ups to evaluate our approaches.

<div style="margin: auto; width: 80%" >

## Scenario

<img style="width: 80%; margin-bottom: 2%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/scenario_overview.png">

<img style="float: left; width: 44%; margin-right: 1%; margin-bottom: 2%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/scenario_blueprint_1.png">

<img style="float: right; width: 46%; margin-bottom: 2%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/scenario_blueprint_2.png">

<img style="width: 48%; margin-left: 25%; margin-bottom: 2%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/scenario_blueprint_3.png">


## Recognition with naive solution

### Unambiguous

<img style="float: left; width: 46%; margin-bottom: 5%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/simple_scene_naive_pointing_yellow.png">

<img style="float: right; width: 46%; margin-bottom: 5%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/simple_scene_naive_pointing_green.png">

### Ambiguous

<img style="float: left; width: 46%; margin-bottom: 5%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/ambiguous_scene_naive_pointing_yellow.png">

<img style="float: right; width: 46%; margin-bottom: 5%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/ambiguous_scene_naive_pointing_red.png">


## Recognition with GWR based solution

### Unambiguous

<img style="float: left; width: 46%; margin-bottom: 5%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/simple_scene_gwr_pointing_red.png">

<img style="float: right; width: 46%; margin-bottom: 5%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/simple_scene_gwr_pointing_green.png">

### Ambiguous

<img style="float: left; width: 46%; margin-bottom: 5%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/ambiguous_scene_gwr_pointing_yellow.jpg">

<img style="float: right; width: 46%; margin-bottom: 5%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/ambiguous_scene_gwr_pointing_green.jpg">

## Ambiguity detection

<img style="float: left; width: 44%; margin-right: 1%; margin-bottom: 2%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/un_amb_1.jpg">

<img style="float: right; width: 44%; margin-bottom: 2%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/un_amb_2.jpg">

<img style="float: left; width: 44%; margin-right: 1%; margin-bottom: 2%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/amb_1.png">

<img style="float: right; width: 44%; margin-bottom: 2%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/amb_2.jpg">

<img style="width: 80%; margin-left: 8%; margin-bottom: 2%" src="https://raw.githubusercontent.com/d4vidbiertmpl/Bachelors-thesis/master/demo_media/demo_images/union_of_bbs.png">

</div>