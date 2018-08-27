#!/bin/bash

#dicom to nii
echo "------------------------------------------------------------------"
echo "              Converting DICOM to NIFTI                       "
echo "------------------------------------------------------------------"
/Applications/MATLAB_R*.app/bin/matlab -nodesktop -nodisplay -r "x = pwd; cd /Volumes/STORAGE/jackie_work/dicm2nii; dicm2nii(strcat(x,'/*WIP_T2W_GRASE*'),strcat(x,'/analysis'), 1); dicm2nii(strcat(x,'/*_T1_*'),strcat(x,'/analysis'), 1); exit"

#rename everything
cd ./analysis
mv *T1_1MM* T1.nii.gz
mv *T2W_GRASE* GRASE.nii.gz

#get rid of NaN voxels
fslmaths T1.nii.gz -nan T1.nii.gz
fslmaths GRASE.nii.gz -nan GRASE.nii.gz

#brain extraction
echo "------------------------------------------------------------------"
echo "                       Extracting Brains                       "
echo "------------------------------------------------------------------"
echo '...\r\c'
bet T1.nii.gz T1_bet.nii.gz -Z -R -f 0.3 
echo '......\r\c'
bet GRASE.nii.gz GRASE_bet.nii.gz -Z -R -B -F 
echo '......Done\r\c'
echo 

#fsl-fast segmentation on T1
echo "------------------------------------------------------------------"
echo "                  Performing T1 Segmentation                      "
echo "------------------------------------------------------------------"
echo '...\r\c'
fast -t 1 -n 3 -H 0.15 T1_bet.nii.gz
echo '...Done segmentation\r\c'
echo 

#register T1 segmentation to T2
echo "------------------------------------------------------------------"
echo "              Registering T1 segmentation to T2                   "
echo "------------------------------------------------------------------"
echo '...\r\c'
flirt -in GRASE_bet.nii.gz -ref T1_bet.nii.gz -omat T2toT1.mat;
echo '......\r\c'
convert_xfm -omat T1toT2.mat -inverse T2toT1.mat
echo '.........\r\c'
flirt -in T1_bet_seg.nii.gz -ref GRASE_bet.nii.gz -applyxfm -init T1toT2.mat -out T2_bet_regseg.nii.gz
echo '.........Completed\r\c'
echo 
