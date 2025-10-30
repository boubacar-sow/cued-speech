# Pick a directory where you have write permission
cd $HOME

# Create a downloads folder if needed
mkdir -p ~/downloads && cd ~/downloads

# Download Boost 1.85.0 from the official SourceForge mirror (works fine)
wget https://sourceforge.net/projects/boost/files/boost/1.85.0/boost_1_85_0.tar.gz/download -O boost_1_85_0.tar.gz

# Verify file size (should be ~120 MB, NOT 11 KB)
ls -lh boost_1_85_0.tar.gz

# Extract
tar -xzf boost_1_85_0.tar.gz
cd boost_1_85_0

# Clean up from any prior attempts
./b2 --clean-all

# Prepare the build system for these libraries
./bootstrap.sh --with-libraries=system,thread,program_options,unit_test_framework

# Actually build and install them
./b2 --with-system --with-thread --with-program_options --with-unit_test_framework \
    variant=release link=shared,static threading=multi runtime-link=shared \
    install --prefix=$HOME/local/boost -j$(nproc)
