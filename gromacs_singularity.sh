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










-- pkg-config could not detect fftw3f, trying generic detection
Could not find fftw3f library named libfftw3f, please specify its location in CMAKE_PREFIX_PATH or FFTWF_LIBRARY by hand (e.g. -DFFTWF_LIBRARY='/path/to/libfftw3f.so')
CMake Error at cmake/gmxManageFFTLibraries.cmake:91 (MESSAGE):
  Cannot find FFTW 3 (with correct precision - libfftw3f for mixed-precision
  GROMACS or libfftw3 for double-precision GROMACS).  Either choose the right
  precision, choose another FFT(W) library (-DGMX_FFT_LIBRARY), enable the
  advanced option to let GROMACS build FFTW 3 for you
  (-DGMX_BUILD_OWN_FFTW=ON), or use the really slow GROMACS built-in fftpack
  library (-DGMX_FFT_LIBRARY=fftpack).
Call Stack (most recent call first):
  CMakeLists.txt:671 (include)


-- Configuring incomplete, errors occurred!
See also "/gromacs-2021.5/build/CMakeFiles/CMakeOutput.log".
See also "/gromacs-2021.5/build/CMakeFiles/CMakeError.log".
FATAL:   While performing build: while running engine: exit status 1






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
