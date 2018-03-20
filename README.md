# CSE274_Selected_Topics_in_Graphics

Project in sampling and reconstruction of visual appearance

by Asbjoern Lystrup and Marcus Loo Vergara

For our project we implemented soft shadows based on the paper *Axis-Aligned Filtering for Interactive Sampled Soft Shadows* (ADD LINK) by Soham Uday Mehta, Brandon Wang, and Ravi Ramamoorthi. The soft shadows are based on Monte Carlo sampling, and is hence physically accurate. It uses planar area light sources. The method starts off by casting a number of rays per pixel to obtain values defining the occlusion. These values are then used to find filter widths for blurring the noise, and an additional sample count to further enhance accuracy in the most complex areas of the shadows. Then the filtering is applied. The filtering is axis-aligned and done in image-space, providing great performance and making interactivity possible.

We started out by downloading and learning NVIDIA's OptiX, which is a real-time raytracing framework used by the paper. We looked at a couple of tutorials to understand the interaction between rays and geometry, and started to work on our implementation. 

The paper's method derives from recent work on frequency analysis and sheared filtering for offline soft shadows. The paper develops a theoretical analysis for axis-aligned filtering. After setting up the foundation of the implementation, we spent some time studying fourier spectrums to get a better idea of the paper's theory.

We continued working on our implementation and added occlusion calculation. Nine rays per pixel are sent toward random points on the light source to obtain distances between the light source and the closest and furthest occluder. We store these values in a buffer 
