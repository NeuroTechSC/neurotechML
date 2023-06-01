@echo off
git status > git_status_output.txt
type git_status_output.txt
del git_status_output.txt
set /p commitMessage="Enter commit message: "
git add --all
git commit -am "%commitMessage%"
git push
pause