@echo off
chcp 65001 >null 2>&1

echo    ____    __          
echo   / __/__ / /___ _____ 
echo  _\ \/ -_) __/ // / _ \
echo /___/\__/\__/\_,_/ .__/
echo                 /_/    

pip install pandas librosa numpy matplotlib >null 2>&1
echo Hoàn tất
pause >null 2>&1
del null