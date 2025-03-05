Welcome to my implementation of LiDAR-Based SLAM! This project was completed for a graduate course (ECE 276a) at UCSD.

Simultaneous localization and mapping (SLAM) is a fundamental problem seen in mobile robot autonomy everywhere these days. In this project, I implement SLAM to generate a motion model of a differential drive robot through designing my own odometry model and localization model using raw encoder, IMU, and LiDAR data. I create an occupancy grid map the environment based off this data and various coordinate transforms based off the location of the estimated body in time and the LiDAR data relative to the body. Using raw RGBD data, I also project RGB images onto a reconstructed 3D mapping of the environment, and then create a texture map of the environment following the trajectory of the robot. Finally, I utilize the GTSAM library to implement factor graph SLAM. Through this I improved the accuracy of the predicted trajectory through pose optimization and loop closures.

The main concepts implemented in this project are:
- Odometry model: Trajectory prediction of a differential drive robot
- Iterative Closest Points (ICP) algorithm: Optimization upon initial odometry trajectory
- Bresenham's algorithm: Occupancy grid construction and environment building
- Coordinate transforms: LiDAR points to world frame association, texture mapping 
- GTSAM Library: factor graph SLAM pose graph optimizations and loop closures

You may find all relevant code in this repository, please ensure you have the proper libraries installed. An easy way to accomplish this is through Conda.

A paper describing the equipment used, my implementation, and results can be found in the "LiDAR Based SLAM for Differential Drive Robots" pdf. Thank you! 
