@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem Daol Vision - Safe Deploy Script v3
rem - GitHub main is the deploy source
rem - Runtime ROI/template files on Jetson are preserved
rem - deploy.bat itself is committed
rem - If this PC is behind GitHub, deployment stops safely
rem ============================================================

set "SSH_KEY=%USERPROFILE%\.ssh\deploy_for_gha"

rem ===== Equipment-specific values =====
set "BOARD_NAME=CN8-004"
set "TARGET=robot96@100.122.148.22"

rem ===== Git deploy source =====
set "GIT_REMOTE=origin"
set "GIT_BRANCH=main"
set "DEPLOY_REF=%GIT_REMOTE%/%GIT_BRANCH%"

echo.
echo ============================================
echo Deploy %BOARD_NAME%
echo TARGET     = %TARGET%
echo DEPLOY_REF = %DEPLOY_REF%
echo ============================================
echo.

call :PRECHECK_SYNC
if errorlevel 1 exit /b 1

call :COMMIT_AND_PUSH
if errorlevel 1 exit /b 1

call :DEPLOY_TO_TARGET
if errorlevel 1 exit /b 1

echo.
echo ===== %BOARD_NAME% DEPLOY COMPLETE =====
pause
exit /b 0


:PRECHECK_SYNC
echo [0/3] Check GitHub sync

git fetch %GIT_REMOTE% %GIT_BRANCH%
if errorlevel 1 (
    echo ERROR: git fetch failed.
    exit /b 1
)

set "REMOTE_AHEAD=0"
for /f %%A in ('git rev-list --count HEAD..%DEPLOY_REF%') do set "REMOTE_AHEAD=%%A"

if not "!REMOTE_AHEAD!"=="0" (
    echo.
    echo ERROR: This PC is behind GitHub by !REMOTE_AHEAD! commit^(s^).
    echo Run this first:
    echo     git pull --ff-only %GIT_REMOTE% %GIT_BRANCH%
    echo.
    exit /b 1
)

exit /b 0


:COMMIT_AND_PUSH
echo.
echo [1/3] Stage source files

git add deploy.bat 2>nul
git add .gitignore 2>nul
git add src
git add data/config
git add data/roi

rem ===== Keep local credentials out of Git =====
git rm -q --cached -f --ignore-unmatch data/config/email_config.json 2>nul
git rm -q --cached -f --ignore-unmatch data/config/google_drive_client.json 2>nul
git rm -q --cached -f --ignore-unmatch data/config/google_drive_token.json 2>nul

rem ===== Do not commit equipment runtime files =====
git reset -q -- data/roi/roi.json 2>nul
git reset -q -- data/roi/align_template.png 2>nul
git reset -q -- data/roi/profiles/*_roi.json 2>nul
git reset -q -- data/roi/profiles/align_template_*.png 2>nul
git reset -q -- data/roi/baseline_profile.json 2>nul
git reset -q -- data/roi/recipe_static.json.save 2>nul
git reset -q -- data/roi/recipes/recipe_auto.json 2>nul
git reset -q -- data/roi/templates 2>nul
git reset -q -- logs 2>nul
git reset -q -- data/logs 2>nul
git reset -q -- data/dataset 2>nul

echo.
echo [2/3] Commit if changed

git diff --cached --quiet
if errorlevel 1 (
    git commit -m "auto deploy %BOARD_NAME%"
    if errorlevel 1 exit /b 1
) else (
    echo No staged changes. Skip commit.
)

echo.
echo [3/3] Push %GIT_REMOTE% %GIT_BRANCH%
git push %GIT_REMOTE% %GIT_BRANCH%
exit /b %errorlevel%


:DEPLOY_TO_TARGET
echo.
echo Deploy %DEPLOY_REF% to %TARGET%

ssh -i "%SSH_KEY%" %TARGET% "cd ~/vision_project || exit 1; B=/tmp/vision_runtime_bak; rm -rf $B; mkdir -p $B/profiles $B/config $B/recipes || exit 10; if [ -f data/roi/roi.json ]; then cp -p data/roi/roi.json $B/roi.json || exit 11; fi; if [ -f data/roi/align_template.png ]; then cp -p data/roi/align_template.png $B/align_template.png || exit 11; fi; for f in data/roi/profiles/*_roi.json data/roi/profiles/align_template_*.png; do if [ -e \"$f\" ]; then cp -p \"$f\" $B/profiles/ || exit 12; fi; done; if [ -f data/roi/baseline_profile.json ]; then cp -p data/roi/baseline_profile.json $B/baseline_profile.json || exit 13; fi; if [ -f data/roi/recipe_static.json.save ]; then cp -p data/roi/recipe_static.json.save $B/recipe_static.json.save || exit 13; fi; if [ -f data/roi/recipes/recipe_auto.json ]; then cp -p data/roi/recipes/recipe_auto.json $B/recipes/recipe_auto.json || exit 13; fi; if [ -f data/config/email_config.json ]; then cp -p data/config/email_config.json $B/config/email_config.json || exit 14; fi; if [ -f data/config/google_drive_client.json ]; then cp -p data/config/google_drive_client.json $B/config/google_drive_client.json || exit 15; fi; if [ -f data/config/google_drive_token.json ]; then cp -p data/config/google_drive_token.json $B/config/google_drive_token.json || exit 16; fi; echo RUNTIME_BACKUP_OK; git fetch %GIT_REMOTE% %GIT_BRANCH% || exit 20; git reset --hard %DEPLOY_REF% || exit 21; mkdir -p data/roi/profiles data/roi/recipes data/config || exit 22; if [ -f $B/roi.json ]; then cp -p $B/roi.json data/roi/roi.json || exit 23; fi; if [ -f $B/align_template.png ]; then cp -p $B/align_template.png data/roi/align_template.png || exit 23; fi; for f in $B/profiles/*; do if [ -e \"$f\" ]; then cp -p \"$f\" data/roi/profiles/ || exit 24; fi; done; if [ -f $B/baseline_profile.json ]; then cp -p $B/baseline_profile.json data/roi/baseline_profile.json || exit 25; fi; if [ -f $B/recipe_static.json.save ]; then cp -p $B/recipe_static.json.save data/roi/recipe_static.json.save || exit 25; fi; if [ -f $B/recipes/recipe_auto.json ]; then cp -p $B/recipes/recipe_auto.json data/roi/recipes/recipe_auto.json || exit 25; fi; if [ -f $B/config/email_config.json ]; then cp -p $B/config/email_config.json data/config/email_config.json || exit 26; elif [ ! -f data/config/email_config.json ] && [ -f data/config/email_config.example.json ]; then cp data/config/email_config.example.json data/config/email_config.json || exit 26; fi; if [ -f $B/config/google_drive_client.json ]; then cp -p $B/config/google_drive_client.json data/config/google_drive_client.json || exit 27; fi; if [ -f $B/config/google_drive_token.json ]; then cp -p $B/config/google_drive_token.json data/config/google_drive_token.json || exit 28; fi; echo RUNTIME_RESTORE_OK; git rev-parse --short HEAD; echo DEPLOY_DONE_%BOARD_NAME%"

exit /b %errorlevel%
