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










+ DEBIAN_FRONTEND=noninteractive
+ TZ=Etc/UTC
+ ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime
+ dpkg-reconfigure --frontend noninteractive tzdata
dpkg-query: package 'tzdata' is not installed and no information is available
Use dpkg --info (= dpkg-deb --info) to examine archive files.
/usr/sbin/dpkg-reconfigure: tzdata is not installed
FATAL:   While performing build: while running engine: exit status 1





Bootstrap: docker
From: ubuntu:20.04

%post
    DEBIAN_FRONTEND=noninteractive
    TZ=Etc/UTC
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime
    dpkg-reconfigure --frontend noninteractive tzdata
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
