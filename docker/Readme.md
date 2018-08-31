# Docker containers for Intel MKL-DNN versions of Deep Learning Frameworks

These containers rely heavily on the great support the [Anaconda](http://www.anaconda.com) distribution has for Intel optimizations.

To build the container:
`docker build -t "intel_optimized" .`

To run a built container:
`docker run -it intel_optimized`

