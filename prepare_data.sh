#!/bin/bash
cd data
URL=http://people.eecs.berkeley.edu/~akar/lsm/shapenet_release.tar.gz
CHECKSUM=61feff8480368e00eb928d4b10a40a40
FILE=shapenet_release.tar.gz

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

wget $URL -O $FILE
tar xvzf shapenet_release.tar.gz

echo "Processing voxels"
cd voxels
for i in `ls *.tar.gz`
do
    echo "Processing $i"
    tar xvzf $i > /dev/null
done

echo "Processing renderings"
cd ../renders
for i in `ls *.tar.gz`
do
    echo "Processing $i"
    tar xvzf $i > /dev/null && rm $i
done

cd ../..
echo "Done. Please run this command again to verify that checksum = $CHECKSUM."