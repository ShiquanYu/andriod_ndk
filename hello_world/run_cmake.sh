#! /bin/bash

cmake -DCMAKE_TOOLCHAIN_FILE=/home/crashdown/andriod_ndk/android-ndk-r21e-linux-x86_64/android-ndk-r21e/build/cmake/android.toolchain.cmake \
-DANDROID_ABI=armeabi-v7a with Neon \
-DANDROID_NATIVE_API_LEVEL=21 \
..
