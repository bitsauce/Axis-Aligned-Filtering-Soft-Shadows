# CSE274_Selected_Topics_in_Graphics

Project in sampling and reconstruction of visual appearance

by Asbjoern Lystrup and Marcus Loo Vergara

For our project we implemented soft shadows based on the paper *Axis-Aligned Filtering for Interactive Sampled Soft Shadows* (ADD LINK) by Soham Uday Mehta, Brandon Wang, and Ravi Ramamoorthi. The soft shadows are based on Monte Carlo sampling, and is hence physically accurate. The method starts off by casting a number of rays per pixel to obtain values defining the occlusion. These values are then used to find filter widths for blurring the noise, and an additional sample count to further enhance accuracy in the most complex areas of the shadows. 
