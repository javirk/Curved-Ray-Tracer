# PyTorch curved Ray Tracer

This is a ray tracer in Schwarzschild spacetime. Rays are sent from the camera to the scene, and at each time t, the 
position and velocity for t+1 is calculated following the orbit equation of said spacetime.

Results are acceptable. However, the geometry of spacetime only affects camera rays, which means that light is wrong. 
I will work on fixing this with photon mapping, which should work with a very similar approach.

The latest version of this repository produces images like:
![readme_1](https://github.com/javirk/Curved-Ray-Tracer/blob/master/readme_images/readme_1.png?raw=true)
![readme_2](https://github.com/javirk/Curved-Ray-Tracer/blob/master/readme_images/readme_2.png?raw=true)