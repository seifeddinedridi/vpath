VPath - Physically Based Volumetric Path Tracer
=

**VPath** is a CUDA path tracer that fully solves the radiative transfer equation, which makes it capable of rendering complex lighting effects in an unbiased manner. 

Features:
-
  
- Full Volumetric Scattering of light in homogeneous media
- Accurate Monte Carlo Sub-surface Scattering
- GPU Acceleration using CUDA architecture
- Diffuse BRDF
- Fresnel Reflection/Refraction
- Sphere primitives

Renders:
-
<p align="center">
<img src="https://github.com/D-POWER/vpath/blob/master/renders/sss-spheres.png?raw=true"/>
<img src="https://github.com/D-POWER/vpath/blob/master/renders/image192605.png?raw=true"/>
</p>

<b>[WARNING]</b> This project is meant as an experimental playground to test my ideas on volumetric rendering, so bear in mind that the code may contain bugs or present malfunction.
