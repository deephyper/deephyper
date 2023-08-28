Spack
*****

`Spack <https://spack.readthedocs.io/en/latest/>`_ is a package management tool designed to support multiple versions and configurations of software on a wide variety of platforms and environments. We use Spack to build from source some dependencies of DeepHyper.

Start by installing Spack on your system. The following command will install Spack in the current directory:

.. code-block:: console
    
    $ git clone -c feature.manyFiles=true https://github.com/spack/spack.git
    $ . ./spack/share/spack/setup-env.sh


.. _Redis Server Install:

Redis Server & RedisJSON
========================

Create a Spack environment where the redis-server will be installed:

.. code-block:: console

    $ spack env create redisjson
    $ spack env activate redisjson

Download the deephyper Spack package repository and add it to the environment:

.. code-block:: console

    $ git clone https://github.com/deephyper/deephyper-spack-packages.git
    $ spack repo add deephyper-spack-packages

Then, add the ``redisjson`` Spack package to the environment:

.. code-block:: console

    $ spack add redisjson

Finally, install:

.. code-block:: console
    
    $ spack install

Now that the Redis server is installed and the RedisJSON pluging is compiled you can start the Redis server using the appropriate ``redis.conf``. The following commands will create a ``redis.conf`` file which configures the server to listen on all interfaces, and also load the RedisJSON plugin:

.. code-block:: console

    $ touch redis.conf
    $ echo "bind 0.0.0.0" >> redis.conf
    $ cat $(spack find --path redisjson | grep -o "/.*/redisjson.*")/redis.conf >> redis.conf

Finally, start the Redis server:

.. code-block:: console

    $ redis-server redis.conf