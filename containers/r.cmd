
SET TAG=sp-redis
SET FILE=Dockerfile-redis
SET NAME=%TAG%-name

docker stop %NAME%

docker rm %NAME%

docker build -t %TAG% -f %FILE% .

SET VOLUME_PATH=%HOMEDRIVE%%HOMEPATH%\shared_volumes\sp\redis

echo "Using %VOLUME_PATH%."

docker run -it --name %NAME% -p 6379:6379 -v %VOLUME_PATH%:/data %TAG% --appendonly yes