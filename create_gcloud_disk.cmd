REM Create Additional Storage Disks

SET INSTANCE_NAME="f22-custom-vm"
SET DISK_NAME="f22data"
SET ZONE="us-central1-c"

REM gcloud compute disks create %DISK_NAME% --zone=%ZONE% --size 100 --type pd-standard

gcloud compute instances attach-disk %INSTANCE_NAME% --zone=%ZONE% --disk %DISK_NAME%

REM gcloud compute instances detach-disk %INSTANCE_NAME% --zone=%ZONE% --disk %DISK_NAME%