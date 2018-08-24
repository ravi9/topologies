installdir=${1:-/usr/local/openmpi}
rm -rf openmpi-*
wget https://www.open-mpi.org/software/ompi/v3.1/downloads/openmpi-3.1.1.tar.gz
tar -xzf openmpi-*.tar.gz && cd openmpi-3.1.1
./configure --prefix=$installdir
make -j
sudo make install
echo "Installed openMPI into directory ${1}"

