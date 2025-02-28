sudo apt update
sudo apt install -y build-essential libssl-dev uuid-dev libgpgme-dev squashfs-tools libseccomp-dev wget pkg-config


distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
sudo wget -O /usr/share/keyrings/sylabs-archive-keyring.gpg https://apt.fury.io/sylabs/gpg.
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/sylabs-archive-keyring.gpg] https://apt.fury.io/sylabs/ $distribution main" | sudo tee /etc/apt/sources.list.d/sylabs.list

sudo apt update


singularity --version
apptainer version 1.2.5




Bootstrap: docker
From: ubuntu:20.04

%post
    apt-get update
    apt-get install -y wget build-essential cmake gfortran
    wget https://ftp.gromacs.org/pub/gromacs/gromacs-2021.8.tar.gz
    tar xzf gromacs-2021.8.tar.gz
    cd gromacs-2021.8
    mkdir build
    cd build
    cmake .. -DGMX_BUILD_SHARED_LIBS=ON -DGMX_MPI=OFF
    make -j $(nproc)
    make install
    export PATH="/usr/local/gromacs/bin:$PATH"
    gmx --version

%environment
    export PATH="/usr/local/gromacs/bin:$PATH"

%runscript
    gmx "$@"





sudo apptainer build gromacs2021.sif gromacs2021.def










+ apt-get install -y tzdata wget build-essential cmake gfortran libfftw3-de
Reading package lists... Done
Building dependency tree       
Reading state information... Done
E: Unable to locate package libfftw3-de
FATAL:   While performing build: while running engine: exit status 100







Bootstrap: docker
From: ubuntu:20.04

%post
    DEBIAN_FRONTEND=noninteractive
    TZ=Etc/UTC
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime
    apt-get update
    apt-get install -y tzdata wget build-essential cmake gfortran
    dpkg-reconfigure --frontend noninteractive tzdata
    wget https://ftp.gromacs.org/pub/gromacs/gromacs-2021.8.tar.gz
    tar xzf gromacs-2021.8.tar.gz
    cd gromacs-2021.8
    mkdir build
    cd build
    cmake .. -DGMX_BUILD_SHARED_LIBS=ON -DGMX_MPI=OFF
    make -j $(nproc)
    make install
    export PATH="/usr/local/gromacs/bin:$PATH"
    gmx --version

%environment
    export PATH="/usr/local/gromacs/bin:$PATH"

%runscript
    gmx "$@"




apt-get install -y libfftw3-dev


%post
    DEBIAN_FRONTEND=noninteractive
    TZ=Etc/UTC
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime
    apt-get update
    apt-get install -y tzdata wget build-essential cmake gfortran libfftw3-dev
    dpkg-reconfigure --frontend noninteractive tzdata
    wget https://ftp.gromacs.org/gromacs/gromacs-2021.5.tar.gz #Verify this link.
    tar xzf gromacs-2021.5.tar.gz
    cd gromacs-2021.5
    mkdir build
    cd build
    cmake .. -DGMX_BUILD_SHARED_LIBS=ON -DGMX_MPI=OFF
    make -j $(nproc)
    make install
    export PATH="/usr/local/gromacs/bin:$PATH"
    gmx --version


apt-get install -y tzdata wget build-essential cmake gfortran libfftw3-de

    rm -rf build #clean build
