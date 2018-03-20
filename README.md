# CSE274_Selected_Topics_in_Graphics

Project in sampling and reconstruction of visual appearance

by Asbjoern Lystrup and Marcus Loo Vergara

For our project we implemented soft shadows based on the paper *Axis-Aligned Filtering for Interactive Sampled Soft Shadows* (ADD LINK) by Soham Uday Mehta, Brandon Wang, and Ravi Ramamoorthi. The soft shadows are based on Monte Carlo sampling, and is hence physically accurate. It uses planar area light sources. The method starts off by casting a number of rays per pixel to obtain values defining the occlusion. These values are then used to find filter widths for blurring the noise, and an additional sample count to further enhance accuracy in the most complex areas of the shadows. Then the filtering is applied. The filtering is axis-aligned and done in image-space, providing great performance and making interaction possible.

We started out by downloading and learning NVIDIA's OptiX, which is a real-time raytracing framework used by the paper. We looked at a couple of tutorials to understand the interaction between rays and geometry, and started to work on our implementation. 

The paper's method derives from recent work on frequency analysis and sheared filtering for offline soft shadows. The paper develops a theoretical analysis for axis-aligned filtering. After setting up the foundation of the implementation, we spent some time studying fourier spectrums to get a better idea of the paper's theory.

We continued working on our implementation and added occlusion calculation and debugging functionality. For the occlusion calculation, nine rays per pixel are sent toward random points on the light source to obtain distances between the light source and the closest and furthest occluder. We approximate the distance between the light source and the pixels by using the center of the light source. In the same pass, we calculate the filter widths using the equations from the paper and store them in a float buffer. The debugging functionality we implemented let us use the arrow keys to move back and forth to look at different buffers, visualized by normalizing the buffer to make the greatest value correspond to white, and the lowest to black. It also let us see the minimum, average, and maximum value in text form, as well as the framerate.

We then implemented the axis-aligned filtering. We separate the filtering into two passes based on separable convolution; a pass that blurs the shadows horizontally, followed by a pass that blurs the first pass' result vertically. The computed filter widths correspond to standard deviations in a gaussian distribution used for the blurring.

We went back to the pass for the filter width calculation, and implemented adaptive sampling. The adaptive sampling is used to improve both the diffuse accuracy and the filter widths, and is based on the equations from the paper, which uses the occlusion distances. To compare our results with ground truth, we extended our debugging tools to generate three images upon a button press; a filtered result image, a ground truth image, and a disparity map.

As of this time, we had quite decent results, and we held a presentation of our project in class. Our filtering used gaussian offsets corresponding to image-space pixel positions. We had started writing code for the world-space based gaussian offsets, but it wasn't complete, so we continued to work on this. The approach we used was to have a
