FROM deepmi/fastsurfer:latest
RUN apt-get update && apt-get install -y fsl
WORKDIR "/fastsurfer"
COPY convert_ixi.py .
ENV FSLDIR=/usr/share/fsl/5.0
ENV PATH=$PATH:$FSLDIR/bin
ENV LD_LIBRARY_PATH=/usr/lib/fsl/5.0:/usr/share/fsl/5.0/bin
ENV FSLOUTPUTTYPE=NIFTI_GZ
COPY MNI152_T1_1mm_brain.nii.gz .
ENTRYPOINT ["python", "convert_ixi.py"]