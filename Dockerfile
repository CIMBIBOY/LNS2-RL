# Use Ubuntu 20.04 as base
FROM ubuntu:20.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Miniconda manually
RUN apt-get update && apt-get install -y \
    curl wget apt-transport-https ca-certificates gnupg gnupg2 x11-apps software-properties-common \
    build-essential git libboost-all-dev libeigen3-dev libopencv-dev python3-empy python3-pip \
    libxkbcommon-x11-0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-render-util0 libxrender1 libsm6 libxext6 libxrandr2 libxi6 libxtst6 \
    libxcomposite1 libxcursor1 libxdamage1 libxfixes3 libxss1 libxxf86vm1 libglx-mesa0 libglu1-mesa libgl1-mesa-glx \
    libgl1-mesa-dri libosmesa6 libosmesa6-dev mesa-utils && \
    wget https://github.com/Kitware/CMake/releases/download/v3.22.6/cmake-3.22.6-linux-x86_64.sh -O cmake.sh && \
    bash cmake.sh --skip-license --prefix=/usr/local && \
    rm cmake.sh && \
    # Install Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh && \
    /opt/conda/bin/conda init 

# Install ROS Noetic and essential ROS packages
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' \
&& curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
&& apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    python3-rosdep \
    ros-noetic-rospy \
    ros-noetic-roscpp \
    ros-noetic-std-msgs \
    ros-noetic-nav-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-tf \
    ros-noetic-visualization-msgs \
    ros-noetic-message-generation \
    ros-noetic-message-runtime \
    ros-noetic-rviz \
&& rosdep init && rosdep update \
&& apt-get clean && rm -rf /var/lib/apt/lists/*

# Export paths for conda and ROS
ENV PATH=/opt/conda/bin:$PATH
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN /opt/conda/bin/conda create -n lns2rl python=3.11 -y && \
    /opt/conda/bin/conda run -n lns2rl pip install -r /tmp/requirements.txt && \
    /opt/conda/bin/conda run -n lns2rl pip install torch==2.1.1+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 && \
    /opt/conda/bin/conda run -n lns2rl pip install -U "ray[default]" -f https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.html

# Set working directory
WORKDIR /app

# Set up ROS and init script on start
CMD ["/bin/bash", "-c", "source ~/.bashrc && chmod +x /lns2rl/init_script.sh && /lns2rl/init_script.sh"]