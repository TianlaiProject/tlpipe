===========================
Docker containers of tlpipe
===========================

This provides Dokerfile to build docker images of the tlpipe package.
You can create docker containers from the built images which have tlpipe
package and all its dependencies installed in a given unix/linux os.

Create a docker image
=====================

You can create a docker image from the given Dokerfile by run the following
command in a directory containing the Dockerfile ::

   $ docker build -t=tlpipe .

Create an run a container
=========================

You can create and run a docker container from the created docker image as ::

   $ docker run -it --name tlpipe tlpipe /bin/bash

or if you have already created the tlpipe container, run it as ::

   $ docker start tlpipe
   $ docker attach tlpipe


More info
=========

For more information about docker and docker container, please refer to
`<https://www.docker.com>`_.