SET INSTANCE_NAME="f22-custom-vm"
SET IMAGE_FAMILY="pytorch-latest-gpu"
SET ZONE="us-central1-c"
SET INSTANCE_TYPE="n1-highmem-2"

gcloud compute instances create %INSTANCE_NAME% ^
  --zone=%ZONE% ^
  --image-family=%IMAGE_FAMILY% ^
  --machine-type=%INSTANCE_TYPE% ^
  --image-project=deeplearning-platform-release ^
  --maintenance-policy=TERMINATE ^
  --accelerator="type=nvidia-tesla-k80,count=1" ^
  --no-boot-disk-auto-delete ^
  --boot-disk-device-name=%INSTANCE_NAME%-disk ^
  --boot-disk-size=200GB ^
  --boot-disk-type=pd-standard ^
  --scopes=https://www.googleapis.com/auth/cloud-platform ^
  --metadata="install-nvidia-driver=True,proxy-mode=project_editors"

set /p DUMMY=Hit ENTER to continue...

