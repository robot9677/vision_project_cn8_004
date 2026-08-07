@echo off

git add .
git rm -q --cached -f --ignore-unmatch data/config/email_config.json 2>nul
git commit -m "auto deploy"
git push origin main

ssh -i %USERPROFILE%\.ssh\deploy_for_gha robot96@172.30.1.92 "cd ~/vision_project || exit 1; mkdir -p /tmp/vision_runtime_bak/config; cp -f data/roi/roi.json /tmp/vision_runtime_bak/roi.json 2>/dev/null || true; cp -f data/roi/align_template.png /tmp/vision_runtime_bak/align_template.png 2>/dev/null || true; cp -f data/config/email_config.json /tmp/vision_runtime_bak/config/email_config.json 2>/dev/null || true; git fetch origin main && git reset --hard origin/main || exit 1; mkdir -p data/config; cp -f /tmp/vision_runtime_bak/roi.json data/roi/roi.json 2>/dev/null || true; cp -f /tmp/vision_runtime_bak/align_template.png data/roi/align_template.png 2>/dev/null || true; cp -f /tmp/vision_runtime_bak/config/email_config.json data/config/email_config.json 2>/dev/null || true; if [ ! -f data/config/email_config.json ] && [ -f data/config/email_config.example.json ]; then cp data/config/email_config.example.json data/config/email_config.json; fi; git rev-parse --short HEAD; ls -l data/roi/roi.json data/config/email_config.json 2>/dev/null || true"

echo.
echo ===== CODE DEPLOY COMPLETE =====
pause
