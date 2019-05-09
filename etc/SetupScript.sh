#!/bin/bash

#Also need java 8, python 2.7, pip

sudo apt --yes install git 
sudo apt --yes install libjsoncpp-dev
sudo apt --yes install python-pip 
sudo apt --yes install clang-6.0
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-6.0 1000
pip install --user twisted==18.9.0
pip install cffi

mkdir ~/Documents/CarlAgent

export K273_PATH=~/Documents/CarlAgent/k273
export GGPLIB_PATH=~/Documents/CarlAgent/ggplib
export GGP_BASE_PATH=$GGPLIB_PATH/ggp-base
export LD_LIBRARY_PATH=$K273_PATH/build/lib:$GGPLIB_PATH/src/cpp:$LD_LIBRARY_PATH
export CLASSPATH=$GGP_BASE_PATH/build/classes/main:$GGP_BASE_PATH/build/resources/main:$GGP_BASE_PATH/lib/Guava/guava-14.0.1.jar:$GGP_BASE_PATH/lib/Jython/jython.jar:$GGP_BASE_PATH/lib/Clojure/clojure.jar:$GGP_BASE_PATH/lib/Batik/batik-1.7.jar:$GGP_BASE_PATH/lib/FlyingSaucer/core-renderer.jar:$GGP_BASE_PATH/lib/javassist/javassist.jar:$GGP_BASE_PATH/lib/reflections/reflections-0.9.9-RC1.jar:$GGP_BASE_PATH/lib/Htmlparser/htmlparser-1.4.jar
export PYTHONPATH=$GGPLIB_PATH/src:$PYTHONPATH
export PATH=$GGPLIB_PATH/bin:$PATH

echo 'export K273_PATH=~/Documents/CarlAgent/k273
export GGPLIB_PATH=~/Documents/CarlAgent/ggplib
export GGP_BASE_PATH=$GGPLIB_PATH/ggp-base
export LD_LIBRARY_PATH=$K273_PATH/build/lib:$GGPLIB_PATH/src/cpp:$LD_LIBRARY_PATH
export CLASSPATH=$GGP_BASE_PATH/build/classes/main:$GGP_BASE_PATH/build/resources/main:$GGP_BASE_PATH/lib/Guava/guava-14.0.1.jar:$GGP_BASE_PATH/lib/Jython/jython.jar:$GGP_BASE_PATH/lib/Clojure/clojure.jar:$GGP_BASE_PATH/lib/Batik/batik-1.7.jar:$GGP_BASE_PATH/lib/FlyingSaucer/core-renderer.jar:$GGP_BASE_PATH/lib/javassist/javassist.jar:$GGP_BASE_PATH/lib/reflections/reflections-0.9.9-RC1.jar:$GGP_BASE_PATH/lib/Htmlparser/htmlparser-1.4.jar
export PYTHONPATH=$GGPLIB_PATH/src:$PYTHONPATH
export PATH=$GGPLIB_PATH/bin:$PATH' >> ~/.bashrc

cd ~/Documents/CarlAgent
git clone https://github.com/ggplib/k273

cd k273/src/cpp
make install
cd ../../..

git clone https://github.com/richemslie/ggplib.git
cd ggplib

git clone https://github.com/ggp-org/ggp-base
ln -s `pwd`/src/java/propnet_convert `pwd`/ggp-base/src/main/java
cd ggp-base
./gradlew classes assemble

cd ../src/cpp
make

cd ../ggplib/propnet
sed -i 's/-J//g' getpropnet.py
#Allocate as much memory as you want java to have when building propnets
#Here, we change from 8 to 3
sed -i 's/8G/3G/g' getpropnet.py

#Adding our repo...
cd $GGPLIB_PATH/..
git clone https://github.com/ribombee/GGP-CARL
pip install sklearn